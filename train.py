"""
train.py
Phase 7/8 — Multi-utterance training for the Speech World Model.

Default behavior (unchanged):
- Trains EvidenceProjector + Slotizer + BeliefTransitionGRU with next-slot MSE.

Optional phone mode:
- Adds PhoneCTCHead on E_t (frame-level 50Hz features)
- Supervises with pseudo-labels from a frozen teacher phone CTC model
- Total loss: loss_pred + phone_loss_weight * loss_phone
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

# Add src/ to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from huper_features import EvidenceProjector
from slotizer import MeanPoolSlotizer, AttentionSlotizer
from belief_model import BeliefTransitionGRU
from pipeline import WavLMExtractor, BoundaryCache, TARGET_SR, D_RAW, D_PROJ
from phone_ctc import (
    PhoneCTCHead,
    PhoneTeacher,
    ctc_loss_from_logits,
    decode_logits_greedy,
    edit_distance,
    load_pseudo_label_cache,
    phone_cache_path,
    phone_error_rate,
    save_pseudo_label_cache,
)


@dataclass
class Config:
    # Dataset
    train_n: int = 32
    val_n: int = 8
    pooling: str = "mean"  # "mean" | "attn"

    # Model
    d_proj: int = D_PROJ
    d_gru: int = D_PROJ
    gru_layers: int = 1

    # Training
    lr: float = 1e-3
    max_epochs: int = 300
    patience: int = 50
    log_every: int = 10
    grad_clip: float = 1.0

    # Optional phone mode
    enable_phone_mode: bool = False
    phone_loss_weight: float = 2.0
    teacher_model_id: str = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
    phone_eval: bool = False
    phone_head_hidden_dim: int = 0

    # Paths
    cache_dir: str = "data/cache"
    ckpt_dir: str = "data/checkpoints"
    hist_path: str = "data/training_history.json"


def _fetch_librispeech_utterances(n: int, split: str) -> list[dict]:
    """Stream the first n utterances from LibriSpeech."""
    print(f"  Streaming LibriSpeech ({split})  [first {n} utterances] …")
    ds = load_dataset(
        "librispeech_asr",
        "clean",
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    utterances = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        audio = item["audio"]
        wav = np.array(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])

        if sr != TARGET_SR:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        utt_id = f"ls_{split.replace('.', '_')}_{i:04d}"
        utterances.append(
            {
                "id": utt_id,
                "waveform": wav,
                "sr": sr,
                "text": item.get("text", ""),
            }
        )
        if (i + 1) % 8 == 0 or i == n - 1:
            print(f"    loaded {i + 1}/{n} utterances …")

    return utterances


def build_dataset(cfg: Config, device: torch.device) -> tuple[list[dict], list[dict], dict]:
    """
    Build train/val datasets with cached WavLM features + boundaries.
    If phone mode is enabled, also attaches cached teacher pseudo labels.

    Returns
    -------
    train_items, val_items, phone_meta
      phone_meta is empty when phone mode is disabled.
    """
    cache_dir = Path(cfg.cache_dir)
    wavlm_ext = WavLMExtractor(device=device, cache_dir=cache_dir)
    bound_cache = BoundaryCache(cache_dir=cache_dir)

    teacher = None
    phone_meta: dict = {}
    if cfg.enable_phone_mode:
        print(f"\n--- Loading phone teacher: {cfg.teacher_model_id} ---")
        teacher = PhoneTeacher(cfg.teacher_model_id, device)
        phone_meta = {
            "labels": teacher.vocab.labels,
            "blank_id": teacher.vocab.blank_id,
            "teacher_model_id": cfg.teacher_model_id,
        }

    print("\n--- Loading train utterances ---")
    train_raw = _fetch_librispeech_utterances(cfg.train_n, "train.100")

    print("\n--- Loading val utterances ---")
    val_raw = _fetch_librispeech_utterances(cfg.val_n, "validation")

    def enrich(items: list[dict], tag: str) -> list[dict]:
        print(f"\n--- Extracting WavLM features [{tag}] ---")
        enriched = []

        for item in items:
            utt_id = item["id"]
            wav = item["waveform"]
            sr = item["sr"]

            layer24 = wavlm_ext.extract(wav, sr, utt_id)
            boundaries = bound_cache.get(wav, sr, utt_id)

            if len(boundaries) < 3:
                print(f"  [SKIP] {utt_id}  — only {len(boundaries)} boundaries")
                continue

            rec = {
                **item,
                "layer24": layer24,
                "boundaries": boundaries,
                "K": len(boundaries),
                "duration_s": len(wav) / sr,
            }

            if cfg.enable_phone_mode:
                assert teacher is not None
                cache_path = phone_cache_path(cache_dir, utt_id)

                if cache_path.exists():
                    bundle = load_pseudo_label_cache(cache_path)
                else:
                    bundle = teacher.pseudo_labels(wav, sr, target_T=layer24.shape[0])
                    save_pseudo_label_cache(cache_path, bundle)

                # Safety check: all utterances must share vocab/blank id.
                if bundle.vocab.blank_id != int(phone_meta["blank_id"]):
                    raise ValueError("Phone blank_id mismatch across utterances")
                if bundle.vocab.labels != phone_meta["labels"]:
                    raise ValueError("Phone label vocabulary mismatch across utterances")

                rec["phone_frame_ids"] = bundle.frame_ids.astype(np.int64)
                rec["phone_target_ids"] = bundle.target_ids.astype(np.int64)

            enriched.append(rec)

        if not enriched:
            raise RuntimeError(f"No enriched utterances for split={tag}")

        print(
            f"  {tag}: {len(enriched)} utterances enriched "
            f"(K range: {min(x['K'] for x in enriched)}–{max(x['K'] for x in enriched)})"
        )
        return enriched

    train_items = enrich(train_raw, "train")
    val_items = enrich(val_raw, "val")

    return train_items, val_items, phone_meta


def forward_utterance(
    item: dict,
    projector: EvidenceProjector,
    slotizer: nn.Module,
    belief_model: BeliefTransitionGRU,
    device: torch.device,
    pooling: str,
    phone_head: PhoneCTCHead | None = None,
    phone_blank_id: int | None = None,
    phone_loss_weight: float = 1.0,
    compute_phone_metrics: bool = False,
) -> dict:
    """Forward pass for one utterance with optional phone CTC path."""
    layer24 = item["layer24"]
    boundaries = item["boundaries"]

    x = torch.from_numpy(layer24).to(device)
    E_t = projector(x)  # (T, 256)

    if pooling == "mean":
        S = slotizer(E_t, boundaries)
    else:
        S, _ = slotizer(E_t, boundaries)

    S_batch = S.unsqueeze(0)
    _, pred, _ = belief_model(S_batch)
    target = S_batch[:, 1:, :].detach()

    loss_pred = F.mse_loss(pred, target)
    loss = loss_pred

    with torch.no_grad():
        cos_mean = float(F.cosine_similarity(pred.squeeze(0), target.squeeze(0), dim=-1).mean())

    out = {
        "loss": loss,
        "loss_pred": float(loss_pred.detach().cpu().item()),
        "loss_phone": float("nan"),
        "cos": cos_mean,
        "pred": pred,
        "phone_frame_acc": float("nan"),
        "phone_per": float("nan"),
    }

    if phone_head is not None and phone_blank_id is not None:
        logits_phone = phone_head(E_t)  # (T, V)
        target_ids = item["phone_target_ids"]
        loss_phone = ctc_loss_from_logits(logits_phone, target_ids, blank_id=phone_blank_id)
        loss = loss + float(phone_loss_weight) * loss_phone
        out["loss"] = loss
        out["loss_phone"] = float(loss_phone.detach().cpu().item())

        if compute_phone_metrics:
            frame_pred_ids, seq_pred_ids = decode_logits_greedy(logits_phone, blank_id=phone_blank_id)
            frame_ref = np.asarray(item["phone_frame_ids"], dtype=np.int64)
            seq_ref = np.asarray(item["phone_target_ids"], dtype=np.int64)
            T_cmp = min(len(frame_pred_ids), len(frame_ref))
            if T_cmp > 0:
                frame_acc = float((frame_pred_ids[:T_cmp] == frame_ref[:T_cmp]).mean())
            else:
                frame_acc = float("nan")
            out["phone_frame_acc"] = frame_acc
            out["phone_per"] = phone_error_rate(seq_pred_ids, seq_ref)

    return out


def evaluate(
    items: list[dict],
    projector: EvidenceProjector,
    slotizer: nn.Module,
    belief_model: BeliefTransitionGRU,
    device: torch.device,
    pooling: str,
    phone_head: PhoneCTCHead | None = None,
    phone_blank_id: int | None = None,
    phone_loss_weight: float = 1.0,
    compute_phone_metrics: bool = False,
) -> dict:
    """Evaluate model on split and return aggregate metrics."""
    projector.eval()
    slotizer.eval()
    belief_model.eval()
    if phone_head is not None:
        phone_head.eval()

    losses: list[float] = []
    cosines: list[float] = []
    phone_losses: list[float] = []
    phone_frame_accs: list[float] = []
    phone_pers: list[float] = []

    with torch.no_grad():
        for item in items:
            rec = forward_utterance(
                item,
                projector,
                slotizer,
                belief_model,
                device,
                pooling,
                phone_head=phone_head,
                phone_blank_id=phone_blank_id,
                phone_loss_weight=phone_loss_weight,
                compute_phone_metrics=compute_phone_metrics,
            )
            losses.append(float(rec["loss"].detach().cpu().item()))
            cosines.append(float(rec["cos"]))
            if not np.isnan(rec["loss_phone"]):
                phone_losses.append(float(rec["loss_phone"]))
            if not np.isnan(rec["phone_frame_acc"]):
                phone_frame_accs.append(float(rec["phone_frame_acc"]))
            if not np.isnan(rec["phone_per"]):
                phone_pers.append(float(rec["phone_per"]))

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "cos": float(np.mean(cosines)) if cosines else float("nan"),
        "phone_loss": float(np.mean(phone_losses)) if phone_losses else float("nan"),
        "phone_frame_acc": float(np.mean(phone_frame_accs)) if phone_frame_accs else float("nan"),
        "phone_per": float(np.mean(phone_pers)) if phone_pers else float("nan"),
    }


def train(
    cfg: Config,
    train_items: list[dict],
    val_items: list[dict],
    device: torch.device,
    phone_meta: dict,
) -> dict:
    """Train projector + slotizer + GRU, optionally with phone head."""
    projector = EvidenceProjector(d_in=D_RAW, d_out=cfg.d_proj).to(device)
    if cfg.pooling == "mean":
        slotizer: nn.Module = MeanPoolSlotizer().to(device)
    else:
        slotizer = AttentionSlotizer(cfg.d_proj).to(device)
    belief_model = BeliefTransitionGRU(d=cfg.d_gru, n_layers=cfg.gru_layers).to(device)

    phone_head: PhoneCTCHead | None = None
    phone_blank_id: int | None = None

    if cfg.enable_phone_mode:
        if not phone_meta:
            raise ValueError("Phone mode enabled but phone_meta is empty")
        phone_blank_id = int(phone_meta["blank_id"])
        hidden = cfg.phone_head_hidden_dim if cfg.phone_head_hidden_dim > 0 else None
        phone_head = PhoneCTCHead(
            d_in=cfg.d_proj,
            vocab_size=len(phone_meta["labels"]),
            hidden_dim=hidden,
        ).to(device)

    n_proj = sum(p.numel() for p in projector.parameters())
    n_gru = sum(p.numel() for p in belief_model.parameters())
    n_slot = sum(p.numel() for p in slotizer.parameters())
    n_phone = sum(p.numel() for p in phone_head.parameters()) if phone_head is not None else 0

    print("\n  Trainable parameters:")
    print(f"    EvidenceProjector    : {n_proj:>10,}")
    print(f"    BeliefTransitionGRU  : {n_gru:>10,}")
    print(f"    Slotizer             : {n_slot:>10,}")
    if phone_head is not None:
        print(f"    PhoneCTCHead         : {n_phone:>10,}")
    print(f"    Total                : {n_proj + n_gru + n_slot + n_phone:>10,}")

    params = list(projector.parameters()) + list(slotizer.parameters()) + list(belief_model.parameters())
    if phone_head is not None:
        params += list(phone_head.parameters())

    optimizer = torch.optim.AdamW(params, lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=20,
    )

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "belief_model_best.pt"

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_cos": [],
        "val_cos": [],
        "train_phone_loss": [],
        "val_phone_loss": [],
        "train_phone_frame_acc": [],
        "val_phone_frame_acc": [],
        "train_phone_per": [],
        "val_phone_per": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    stale_count = 0

    print(
        f"\n  Training on {device}  |  lr={cfg.lr}  |  max_epochs={cfg.max_epochs}"
        f"  |  patience={cfg.patience}  |  train_n={len(train_items)}  val_n={len(val_items)}"
    )
    if cfg.enable_phone_mode:
        print(
            f"  Active losses: loss_pred + {cfg.phone_loss_weight:.3f} * loss_phone_ctc"
            f"  |  teacher={cfg.teacher_model_id}"
        )
    else:
        print("  Active losses: loss_pred only")

    hdr = (
        f"  {'Epoch':>5}  {'TrLoss':>8}  {'VaLoss':>8}  {'TrCos':>7}  {'VaCos':>7}"
        f"  {'TrPhAcc':>8}  {'VaPhAcc':>8}  {'TrPER':>7}  {'VaPER':>7}  {'LR':>10}"
    )
    print()
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for epoch in range(1, cfg.max_epochs + 1):
        projector.train()
        slotizer.train()
        belief_model.train()
        if phone_head is not None:
            phone_head.train()

        tr_losses: list[float] = []
        tr_cosines: list[float] = []
        tr_phone_losses: list[float] = []
        tr_phone_accs: list[float] = []
        tr_phone_pers: list[float] = []

        idx_order = np.random.permutation(len(train_items))
        for idx in idx_order:
            item = train_items[idx]
            optimizer.zero_grad()

            rec = forward_utterance(
                item,
                projector,
                slotizer,
                belief_model,
                device,
                cfg.pooling,
                phone_head=phone_head,
                phone_blank_id=phone_blank_id,
                phone_loss_weight=cfg.phone_loss_weight,
                compute_phone_metrics=cfg.phone_eval and cfg.enable_phone_mode,
            )
            loss_t = rec["loss"]
            loss_t.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=cfg.grad_clip)
            optimizer.step()

            tr_losses.append(float(loss_t.detach().cpu().item()))
            tr_cosines.append(float(rec["cos"]))
            if not np.isnan(rec["loss_phone"]):
                tr_phone_losses.append(float(rec["loss_phone"]))
            if not np.isnan(rec["phone_frame_acc"]):
                tr_phone_accs.append(float(rec["phone_frame_acc"]))
            if not np.isnan(rec["phone_per"]):
                tr_phone_pers.append(float(rec["phone_per"]))

        train_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        train_cos = float(np.mean(tr_cosines)) if tr_cosines else float("nan")
        train_phone_loss = float(np.mean(tr_phone_losses)) if tr_phone_losses else float("nan")
        train_phone_acc = float(np.mean(tr_phone_accs)) if tr_phone_accs else float("nan")
        train_phone_per = float(np.mean(tr_phone_pers)) if tr_phone_pers else float("nan")

        val_rec = evaluate(
            val_items,
            projector,
            slotizer,
            belief_model,
            device,
            cfg.pooling,
            phone_head=phone_head,
            phone_blank_id=phone_blank_id,
            phone_loss_weight=cfg.phone_loss_weight,
            compute_phone_metrics=cfg.phone_eval and cfg.enable_phone_mode,
        )

        val_loss = val_rec["loss"]
        val_cos = val_rec["cos"]

        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_cos"].append(train_cos)
        history["val_cos"].append(val_cos)
        history["train_phone_loss"].append(train_phone_loss)
        history["val_phone_loss"].append(val_rec["phone_loss"])
        history["train_phone_frame_acc"].append(train_phone_acc)
        history["val_phone_frame_acc"].append(val_rec["phone_frame_acc"])
        history["train_phone_per"].append(train_phone_per)
        history["val_phone_per"].append(val_rec["phone_per"])
        history["lr"].append(current_lr)

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            stale_count = 0
            payload = {
                "epoch": epoch,
                "projector": projector.state_dict(),
                "slotizer": slotizer.state_dict(),
                "belief_model": belief_model.state_dict(),
                "val_loss": val_loss,
                "cfg": asdict(cfg),
            }
            if phone_head is not None:
                payload["phone_head"] = phone_head.state_dict()
                payload["phone_meta"] = phone_meta
            torch.save(payload, best_ckpt)
        else:
            stale_count += 1

        if epoch % cfg.log_every == 0 or epoch == 1:
            tr_pa = train_phone_acc if not np.isnan(train_phone_acc) else float("nan")
            va_pa = val_rec["phone_frame_acc"] if not np.isnan(val_rec["phone_frame_acc"]) else float("nan")
            tr_per = train_phone_per if not np.isnan(train_phone_per) else float("nan")
            va_per = val_rec["phone_per"] if not np.isnan(val_rec["phone_per"]) else float("nan")
            print(
                f"  {epoch:>5}  {train_loss:>8.5f}  {val_loss:>8.5f}  "
                f"{train_cos:>7.4f}  {val_cos:>7.4f}  "
                f"{tr_pa:>8.4f}  {va_pa:>8.4f}  {tr_per:>7.4f}  {va_per:>7.4f}  "
                f"{current_lr:>10.2e}"
            )

        if stale_count >= cfg.patience:
            print(f"\n  Early stop at epoch {epoch}  (val loss stale for {cfg.patience} epochs)")
            break

    print("\n  Training complete.")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Checkpoint: {best_ckpt}")

    hist_path = Path(cfg.hist_path)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"  History saved -> {hist_path}")

    return history


def print_summary(cfg: Config, train_items: list[dict], val_items: list[dict], history: dict) -> None:
    val_loss_final = history["val_loss"][-1]
    val_cos_final = history["val_cos"][-1]
    best_val_loss = min(history["val_loss"])
    mean_K_train = np.mean([x["K"] for x in train_items])
    mean_K_val = np.mean([x["K"] for x in val_items])

    print()
    print("-" * 58)
    print("  Training Summary")
    print("-" * 58)
    print(f"  Train utterances     : {len(train_items)}")
    print(f"  Val utterances       : {len(val_items)}")
    print(f"  Mean slots / utt     : {mean_K_train:.1f} (train)  {mean_K_val:.1f} (val)")
    print(f"  Best val loss        : {best_val_loss:.6f}")
    print(f"  Final val loss       : {val_loss_final:.6f}")
    print(f"  Final val cos        : {val_cos_final:.4f}")

    if cfg.enable_phone_mode:
        ph_acc = history["val_phone_frame_acc"][-1]
        ph_per = history["val_phone_per"][-1]
        print(f"  Final val phone acc  : {ph_acc:.4f}" if not np.isnan(ph_acc) else "  Final val phone acc  : n/a")
        print(f"  Final val phone PER  : {ph_per:.4f}" if not np.isnan(ph_per) else "  Final val phone PER  : n/a")

    print(f"  Epochs trained       : {len(history['val_loss'])}")
    print("-" * 58)


def parse_args() -> Config:
    ap = argparse.ArgumentParser(description="Train Speech World Model")
    ap.add_argument("--train_n", type=int, default=32)
    ap.add_argument("--val_n", type=int, default=8)
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "attn"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--log_every", type=int, default=10)

    # Phone mode switches
    ap.add_argument("--enable_phone_mode", action="store_true", help="Enable phone CTC head training")
    ap.add_argument("--phone_loss_weight", type=float, default=2.0)
    ap.add_argument("--teacher_model_id", type=str, default="vitouphy/wav2vec2-xls-r-300m-timit-phoneme")
    ap.add_argument("--phone_eval", action="store_true", help="Compute phone metrics during epochs")
    ap.add_argument("--phone_head_hidden_dim", type=int, default=0)

    args = ap.parse_args()

    return Config(
        train_n=args.train_n,
        val_n=args.val_n,
        pooling=args.pooling,
        lr=args.lr,
        max_epochs=args.epochs,
        patience=args.patience,
        log_every=args.log_every,
        enable_phone_mode=args.enable_phone_mode,
        phone_loss_weight=args.phone_loss_weight,
        teacher_model_id=args.teacher_model_id,
        phone_eval=args.phone_eval,
        phone_head_hidden_dim=args.phone_head_hidden_dim,
    )


if __name__ == "__main__":
    print("█" * 60)
    print("  TRAINING")
    print("  Speech World Model: belief training (+ optional phone mode)")
    print("█" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Device: {device}")

    cfg = parse_args()
    print("\n  Config:")
    for k, v in asdict(cfg).items():
        print(f"    {k:22s}: {v}")

    t0 = time.time()

    print("\n" + "═" * 52)
    print("  STEP 1 — Dataset")
    print("═" * 52)
    train_items, val_items, phone_meta = build_dataset(cfg, device)

    print("\n" + "═" * 52)
    print("  STEP 2 — Training loop")
    print("═" * 52)
    history = train(cfg, train_items, val_items, device, phone_meta)

    print_summary(cfg, train_items, val_items, history)

    elapsed = time.time() - t0
    print(f"\n  Total wall time: {elapsed / 60:.1f} min")
