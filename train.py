"""
train.py
Phase 7 — Multi-utterance training for the Speech World Model.

Trains EvidenceProjector + BeliefTransitionGRU on a small LibriSpeech subset.
WavLM-Large is frozen throughout; its layer-24 outputs are cached to disk.

Primary loss:  MSE( Ŝ_{k+1}, S_{k+1} )   — next-slot prediction
Auxiliary:     cosine(Ŝ_{k+1}, S_{k+1})  — monitored, not backpropagated

Active losses vs placeholders
──────────────────────────────
  [ACTIVE]      loss_pred   — MSE next-slot prediction (weight=1.0)
  [TODO/Phase8] loss_phone  — phone CTC on E_t  (requires phone labels)
  [TODO/Phase8] loss_asr    — ASR CTC on upsampled B_k (requires transcripts)

Usage
─────
  python train.py                         # default config
  python train.py --train_n 64 --val_n 8  # larger split
  python train.py --epochs 500 --lr 5e-4
"""

# --- Imports ---
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


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Dataset
    train_n:    int   = 32          # number of train utterances
    val_n:      int   = 8           # number of val utterances
    pooling:    str   = "mean"      # "mean" | "attn"

    # Model
    d_proj:     int   = D_PROJ      # EvidenceProjector output dim
    d_gru:      int   = D_PROJ      # BeliefTransitionGRU hidden dim
    gru_layers: int   = 1

    # Training
    lr:         float = 1e-3
    max_epochs: int   = 300
    patience:   int   = 50          # early stop patience on val loss
    log_every:  int   = 10
    grad_clip:  float = 1.0

    # Paths
    cache_dir:  str = "data/cache"
    ckpt_dir:   str = "data/checkpoints"
    hist_path:  str = "data/training_history.json"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset — LibriSpeech via HuggingFace streaming
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_librispeech_utterances(n: int, split: str) -> list[dict]:
    """
    Stream the first n utterances from LibriSpeech.

    split examples: "train.clean.100", "validation.clean", "test.clean"

    Each returned dict has:
      "id"       : str
      "waveform" : np.ndarray  (N_samples,) float32 @ 16kHz
      "sr"       : int
      "text"     : str
    """
    print(f"  Streaming LibriSpeech ({split})  [first {n} utterances] …")
    ds = load_dataset(
        "librispeech_asr", "clean",
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    utterances = []
    for i, item in enumerate(ds):
        if i >= n:
            break
        audio = item["audio"]
        wav   = np.array(audio["array"], dtype=np.float32)
        sr    = int(audio["sampling_rate"])

        # Resample to TARGET_SR if needed
        if sr != TARGET_SR:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
            sr  = TARGET_SR

        # Stable utterance id from dataset fields
        utt_id = f"ls_{split.replace('.', '_')}_{i:04d}"

        utterances.append({
            "id":       utt_id,
            "waveform": wav,
            "sr":       sr,
            "text":     item.get("text", ""),
        })
        if (i + 1) % 8 == 0 or i == n - 1:
            print(f"    loaded {i+1}/{n} utterances …")

    return utterances


def build_dataset(cfg: Config, device: torch.device) -> tuple[
    list[dict], list[dict]
]:
    """
    Load train + val utterances from LibriSpeech streaming.

    For each utterance:
      - extract WavLM layer-24 features (cached to disk)
      - detect syllable boundaries (cached to disk)

    Returns (train_items, val_items).
    Each item is a dict:
      "id"          : str
      "waveform"    : np.ndarray
      "sr"          : int
      "text"        : str
      "layer24"     : np.ndarray  (T, 1024)
      "boundaries"  : list[tuple[int,int]]
      "K"           : int
      "duration_s"  : float
    """
    cache_dir  = Path(cfg.cache_dir)
    wavlm_ext  = WavLMExtractor(device=device, cache_dir=cache_dir)
    bound_cache = BoundaryCache(cache_dir=cache_dir)

    # ── Fetch raw utterances ──────────────────────────────────────────────────
    print("\n--- Loading train utterances ---")
    train_raw = _fetch_librispeech_utterances(cfg.train_n, "train.100")

    print("\n--- Loading val utterances ---")
    val_raw   = _fetch_librispeech_utterances(cfg.val_n,   "validation")

    # ── Extract + cache features ─────────────────────────────────────────────
    def enrich(items: list[dict], tag: str) -> list[dict]:
        print(f"\n--- Extracting WavLM features [{tag}] ---")
        enriched = []
        for item in items:
            utt_id = item["id"]
            wav    = item["waveform"]
            sr     = item["sr"]

            layer24    = wavlm_ext.extract(wav, sr, utt_id)           # (T, 1024)
            boundaries = bound_cache.get(wav, sr, utt_id)             # list[tuple]

            if len(boundaries) < 3:
                print(f"  [SKIP] {utt_id}  — only {len(boundaries)} boundaries")
                continue

            enriched.append({
                **item,
                "layer24":    layer24,
                "boundaries": boundaries,
                "K":          len(boundaries),
                "duration_s": len(wav) / sr,
            })

        print(f"  {tag}: {len(enriched)} utterances enriched "
              f"(K range: {min(x['K'] for x in enriched)}–{max(x['K'] for x in enriched)})")
        return enriched

    train_items = enrich(train_raw, "train")
    val_items   = enrich(val_raw,   "val")

    return train_items, val_items


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers — forward pass + loss
# ─────────────────────────────────────────────────────────────────────────────

def forward_utterance(
    item:        dict,
    projector:   EvidenceProjector,
    slotizer:    nn.Module,
    belief_model: BeliefTransitionGRU,
    device:      torch.device,
    pooling:     str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run one utterance through projector → slotizer → GRU.

    Returns
    -------
    loss     : scalar  MSE(pred, target)
    cos_mean : scalar  mean cosine(pred, target)
    pred     : (1, K-1, 256)
    """
    layer24    = item["layer24"]        # (T, 1024) numpy
    boundaries = item["boundaries"]    # list[tuple]

    # Projector: (T, 1024) → (T, 256)  [gradients flow here]
    x   = torch.from_numpy(layer24).to(device)   # (T, 1024)
    E_t = projector(x)                            # (T, 256)

    # Slotizer: (T, 256) → (K, 256)
    if pooling == "mean":
        S = slotizer(E_t, boundaries)             # (K, 256)
    else:
        S, _ = slotizer(E_t, boundaries)          # (K, 256)

    # BeliefTransitionGRU: (1, K, 256) → pred (1, K-1, 256)
    S_batch = S.unsqueeze(0)                      # (1, K, 256)
    _, pred, _ = belief_model(S_batch)            # pred: (1, K-1, 256)

    target = S_batch[:, 1:, :].detach()           # (1, K-1, 256)

    # Primary loss: MSE next-slot prediction
    loss_pred = F.mse_loss(pred, target)

    # ── Placeholder hooks for future losses ──────────────────────────────────
    # loss_phone = ctc_loss(E_t, phone_labels)   # [TODO Phase 8 — needs labels]
    # loss_asr   = ctc_loss(B_upsampled, words)  # [TODO Phase 8 — needs transcripts]
    # total_loss = 1.0 * loss_pred + 1.0 * loss_phone + 0.5 * loss_asr
    # ─────────────────────────────────────────────────────────────────────────

    loss = loss_pred   # only primary loss active in Phase 7

    # Cosine (detached, just for monitoring)
    with torch.no_grad():
        cos_mean = float(
            F.cosine_similarity(pred.squeeze(0), target.squeeze(0), dim=-1).mean()
        )

    return loss, torch.tensor(cos_mean), pred


def evaluate(
    items:        list[dict],
    projector:    EvidenceProjector,
    slotizer:     nn.Module,
    belief_model: BeliefTransitionGRU,
    device:       torch.device,
    pooling:      str,
) -> tuple[float, float]:
    """Return mean val loss and mean val cosine."""
    projector.eval()
    slotizer.eval()
    belief_model.eval()
    losses, cosines = [], []

    with torch.no_grad():
        for item in items:
            loss, cos, _ = forward_utterance(
                item, projector, slotizer, belief_model, device, pooling
            )
            losses.append(float(loss))
            cosines.append(float(cos))

    return float(np.mean(losses)), float(np.mean(cosines))


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    cfg:          Config,
    train_items:  list[dict],
    val_items:    list[dict],
    device:       torch.device,
) -> dict:
    """
    Train EvidenceProjector + BeliefTransitionGRU.
    Returns history dict with per-epoch metrics.
    """
    # ── Model initialisation ─────────────────────────────────────────────────
    projector    = EvidenceProjector(d_in=D_RAW, d_out=cfg.d_proj).to(device)
    if cfg.pooling == "mean":
        slotizer: nn.Module = MeanPoolSlotizer().to(device)
    else:
        slotizer = AttentionSlotizer(cfg.d_proj).to(device)
    belief_model = BeliefTransitionGRU(d=cfg.d_gru, n_layers=cfg.gru_layers).to(device)

    n_proj   = sum(p.numel() for p in projector.parameters())
    n_gru    = sum(p.numel() for p in belief_model.parameters())
    n_slot   = sum(p.numel() for p in slotizer.parameters())
    print(f"\n  Trainable parameters:")
    print(f"    EvidenceProjector    : {n_proj:>10,}")
    print(f"    BeliefTransitionGRU  : {n_gru:>10,}")
    print(f"    Slotizer             : {n_slot:>10,}")
    print(f"    Total                : {n_proj + n_gru + n_slot:>10,}")

    # ── Optimiser ────────────────────────────────────────────────────────────
    params = (
        list(projector.parameters())
        + list(slotizer.parameters())
        + list(belief_model.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20,
    )

    # ── Checkpoint dir ────────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "belief_model_best.pt"

    # ── Training history ─────────────────────────────────────────────────────
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "train_cos":  [], "val_cos":  [],
        "lr":         [],
    }
    best_val_loss  = float("inf")
    stale_count    = 0

    print(f"\n  Training on {device}  |  lr={cfg.lr}  |  max_epochs={cfg.max_epochs}"
          f"  |  patience={cfg.patience}  |  train_n={len(train_items)}  val_n={len(val_items)}")
    print(f"  Active losses: [ACTIVE] loss_pred (MSE next-slot)  "
          f"| [TODO] loss_phone  | [TODO] loss_asr")
    print()
    hdr = f"  {'Epoch':>5}  {'TrLoss':>8}  {'VaLoss':>8}  {'TrCos':>7}  {'VaCos':>7}  {'LR':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for epoch in range(1, cfg.max_epochs + 1):
        projector.train()
        slotizer.train()
        belief_model.train()

        epoch_losses, epoch_cosines = [], []
        # Shuffle train items each epoch
        idx_order = np.random.permutation(len(train_items))

        for idx in idx_order:
            item = train_items[idx]
            optimizer.zero_grad()

            loss, cos, _ = forward_utterance(
                item, projector, slotizer, belief_model, device, cfg.pooling
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=cfg.grad_clip)
            optimizer.step()

            epoch_losses.append(loss.item())
            epoch_cosines.append(float(cos))

        train_loss = float(np.mean(epoch_losses))
        train_cos  = float(np.mean(epoch_cosines))

        val_loss, val_cos = evaluate(
            val_items, projector, slotizer, belief_model, device, cfg.pooling
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_cos"].append(train_cos)
        history["val_cos"].append(val_cos)
        history["lr"].append(current_lr)

        # Checkpoint best model
        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            stale_count   = 0
            torch.save({
                "epoch":         epoch,
                "projector":     projector.state_dict(),
                "slotizer":      slotizer.state_dict(),
                "belief_model":  belief_model.state_dict(),
                "val_loss":      val_loss,
                "cfg":           asdict(cfg),
            }, best_ckpt)
        else:
            stale_count += 1

        if epoch % cfg.log_every == 0 or epoch == 1:
            print(f"  {epoch:>5}  {train_loss:>8.5f}  {val_loss:>8.5f}  "
                  f"{train_cos:>7.4f}  {val_cos:>7.4f}  {current_lr:>10.2e}")

        if stale_count >= cfg.patience:
            print(f"\n  Early stop at epoch {epoch}  (val loss stale for {cfg.patience} epochs)")
            break

    print(f"\n  Training complete.")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Checkpoint: {best_ckpt}")

    # Save history
    hist_path = Path(cfg.hist_path)
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hist_path, "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"  History saved → {hist_path}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(
    cfg:         Config,
    train_items: list[dict],
    val_items:   list[dict],
    history:     dict,
) -> None:
    val_loss_final = history["val_loss"][-1]
    val_cos_final  = history["val_cos"][-1]
    best_val_loss  = min(history["val_loss"])
    mean_K_train   = np.mean([x["K"] for x in train_items])
    mean_K_val     = np.mean([x["K"] for x in val_items])

    print()
    print("─" * 52)
    print("  Phase 7 Training Summary")
    print("─" * 52)
    print(f"  ├── Train utterances     : {len(train_items)}")
    print(f"  ├── Val utterances       : {len(val_items)}")
    print(f"  ├── Mean slots / utt     : {mean_K_train:.1f} (train)  {mean_K_val:.1f} (val)")
    print(f"  ├── Best val loss        : {best_val_loss:.6f}")
    print(f"  ├── Final val loss       : {val_loss_final:.6f}")
    print(f"  ├── Final val cos        : {val_cos_final:.4f}")
    print(f"  ├── Epochs trained       : {len(history['val_loss'])}")
    print(f"  ├── Active losses        : MSE next-slot [ACTIVE]")
    print(f"  ├── Placeholder losses   : phone CTC, ASR CTC [TODO Phase 8]")
    print(f"  └── Next step            : Phase 8 — add phone/ASR CTC heads")
    print("─" * 52)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> Config:
    ap = argparse.ArgumentParser(description="Train Speech World Model — Phase 7")
    ap.add_argument("--train_n",  type=int,   default=32)
    ap.add_argument("--val_n",    type=int,   default=8)
    ap.add_argument("--pooling",  type=str,   default="mean", choices=["mean", "attn"])
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--epochs",   type=int,   default=300)
    ap.add_argument("--patience", type=int,   default=50)
    ap.add_argument("--log_every",type=int,   default=10)
    args = ap.parse_args()

    return Config(
        train_n    = args.train_n,
        val_n      = args.val_n,
        pooling    = args.pooling,
        lr         = args.lr,
        max_epochs = args.epochs,
        patience   = args.patience,
        log_every  = args.log_every,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("█" * 60)
    print("  PHASE 7 — TRAINING")
    print("  Speech World Model: multi-utterance belief training")
    print("█" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Device: {device}")

    cfg = parse_args()
    print(f"\n  Config:")
    for k, v in asdict(cfg).items():
        print(f"    {k:15s}: {v}")

    t0 = time.time()

    # Step 1: Build dataset (streaming + feature extraction + caching)
    print("\n" + "═" * 52)
    print("  STEP 1 — Dataset")
    print("═" * 52)
    train_items, val_items = build_dataset(cfg, device)

    # Step 2: Train
    print("\n" + "═" * 52)
    print("  STEP 2 — Training loop")
    print("═" * 52)
    history = train(cfg, train_items, val_items, device)

    # Step 3: Summary
    print_summary(cfg, train_items, val_items, history)

    elapsed = time.time() - t0
    print(f"\n  Total wall time: {elapsed / 60:.1f} min")
    print(f"\n  Run verify.py to inspect the trained model.")
