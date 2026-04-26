"""
streaming_pipeline_eval.py
Phase 8 — Integrated Offline vs Streaming Pipeline Evaluation.

Goes deeper than streaming_eval.py: instead of stopping at frame-level cosine
against the oracle, this script pushes both feature sources all the way through
the downstream HuPER pipeline — slotizer → BeliefTransitionGRU — and compares
outputs at every layer.

Layers compared
---------------
  1. Frame-level  cos(E_stream_t, E_offline_t)       mean over [:T_cmp]
  2. Slot-level   cos(S_stream_k, S_offline_k)       mean over [:K_cmp]
  3. Belief-level cos(B_stream_k, B_offline_k)       mean over [:K_cmp]
  4. Pred-level   cos(P_stream_k, P_offline_k)       mean over [:K_cmp-1]

Both paths share:
  - the same EvidenceProjector weights (loaded from checkpoint if present)
  - the same BeliefTransitionGRU weights
  - the same syllable boundaries (detected once on the full waveform)

This isolates the windowed-attention degradation from any weight or boundary
mismatch.

Usage
-----
  python streaming_pipeline_eval.py
  python streaming_pipeline_eval.py --n_utt 2 --lookaheads 40
  python streaming_pipeline_eval.py --n_utt 3 --lookaheads 0,40,160

Note: for frame-level-only evaluation with figures, use streaming_eval.py instead.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoFeatureExtractor, WavLMModel

sys.path.insert(0, str(Path(__file__).parent / "src"))
from huper_features import EvidenceProjector
from slotizer import MeanPoolSlotizer
from belief_model import BeliefTransitionGRU
from syllable_clock import detect_syllables
from streaming_encoder import WAVLM_ID, TARGET_SR, D_RAW, D_PROJ
from feature_sources import (
    FeatureBundle,
    extract_offline_features,
    extract_streaming_features,
)

ROOT_DIR   = Path(__file__).parent
CKPT_PATH  = ROOT_DIR / "data" / "checkpoints" / "belief_model_best.pt"
OUT_DIR    = ROOT_DIR / "data" / "figures_phase8_final"
SAMPLE_WAV = ROOT_DIR / "data" / "samples" / "librispeech_sample.wav"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_utterances(n: int) -> list[dict]:
    """Stream first n utterances from LibriSpeech validation set."""
    try:
        print(f"  Streaming LibriSpeech validation [first {n} utterances]…")
        ds = load_dataset(
            "librispeech_asr", "clean", split="validation",
            streaming=True, trust_remote_code=True,
        )
        items = []
        for i, item in enumerate(ds):
            if i >= n:
                break
            audio = item["audio"]
            wav   = np.array(audio["array"], dtype=np.float32)
            sr    = int(audio["sampling_rate"])
            if sr != TARGET_SR:
                wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
            items.append({
                "id":         f"val_{i:04d}",
                "waveform":   wav,
                "sr":         TARGET_SR,
                "text":       item.get("text", ""),
                "duration_s": len(wav) / TARGET_SR,
            })
        print(f"  Loaded {len(items)} utterances from LibriSpeech")
        return items
    except Exception as exc:
        # Fall back to the bundled sample wav if dataset unavailable
        print(f"  WARNING: LibriSpeech stream failed ({exc})")
        print(f"  Falling back to bundled sample: {SAMPLE_WAV}")
        if not SAMPLE_WAV.exists():
            raise FileNotFoundError(f"Bundled sample not found: {SAMPLE_WAV}") from exc
        wav, _ = librosa.load(str(SAMPLE_WAV), sr=TARGET_SR, mono=True)
        wav = wav.astype(np.float32)
        return [{"id": "local_sample", "waveform": wav, "sr": TARGET_SR,
                 "text": "", "duration_s": len(wav) / TARGET_SR}] * min(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Downstream rollout helper
# ─────────────────────────────────────────────────────────────────────────────

def roll_downstream(
    E_np:       np.ndarray,              # (T_raw, 256) float32
    boundaries: list[tuple[int, int]],   # at 50Hz, from detect_syllables
    T_max:      int,                     # clip frames to this length
    slotizer:   MeanPoolSlotizer,
    belief_gru: BeliefTransitionGRU,
    device:     torch.device,
) -> dict:
    """
    Run slotizer + BeliefTransitionGRU on one feature sequence.

    Parameters
    ----------
    E_np       : (T_raw, 256) — feature frames
    boundaries : syllable slot boundaries (shared between offline and streaming)
    T_max      : clip E_t to T_max frames before slotizing
    slotizer   : MeanPoolSlotizer (stateless)
    belief_gru : BeliefTransitionGRU (weights already loaded)
    device     : torch.device

    Returns
    -------
    dict with keys:
      S    : (K, 256) numpy — slot evidence
      B    : (K, 256) numpy — belief states
      pred : (K-1, 256) numpy — next-slot predictions
      K    : int
    """
    # Clip to T_max
    E_np_clipped = E_np[:T_max]                                # (T, 256)
    T = E_np_clipped.shape[0]

    E_t = torch.from_numpy(E_np_clipped).to(device)           # (T, 256)

    # Clip boundaries to valid frame range
    bounds_clipped = [
        (s, min(e, T - 1))
        for s, e in boundaries
        if s < T
    ]

    if len(bounds_clipped) == 0:
        # Degenerate — no valid slots
        empty = np.empty((0, D_PROJ), dtype=np.float32)
        return {"S": empty, "B": empty, "pred": empty, "K": 0}

    # Slotize: (T, 256) → (K, 256)
    S = slotizer(E_t, bounds_clipped)                          # (K, 256) tensor

    # Belief rollout: (1, K, 256) → B, pred, L
    with torch.no_grad():
        B_tensor, pred_tensor, _ = belief_gru(S.unsqueeze(0)) # (1,K,256) each

    return {
        "S":    S.cpu().numpy(),                   # (K, 256)
        "B":    B_tensor.squeeze(0).cpu().numpy(), # (K, 256)
        "pred": pred_tensor.squeeze(0).cpu().numpy(), # (K-1, 256)
        "K":    S.shape[0],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-utterance evaluation
# ─────────────────────────────────────────────────────────────────────────────

def compare_bundles(
    offline:    dict,   # roll_downstream output for offline path
    stream:     dict,   # roll_downstream output for streaming path
    fb_off:     FeatureBundle,
    fb_str:     FeatureBundle,
) -> dict:
    """
    Compute all layer-wise cosine similarities between offline and streaming.

    Returns a flat dict of scalar metrics.
    """
    T_cmp = min(fb_off.T, fb_str.T)
    K_cmp = min(offline["K"], stream["K"])

    # ── Frame cosine ─────────────────────────────────────────────────────────
    if T_cmp >= 2:
        Eo = torch.from_numpy(fb_off.E_t[:T_cmp])
        Es = torch.from_numpy(fb_str.E_t[:T_cmp])
        frame_cos = float(F.cosine_similarity(Eo, Es, dim=-1).mean())
    else:
        frame_cos = float("nan")

    # ── Slot cosine ───────────────────────────────────────────────────────────
    if K_cmp >= 2:
        So = torch.from_numpy(offline["S"][:K_cmp])
        Ss = torch.from_numpy(stream["S"][:K_cmp])
        slot_cos = float(F.cosine_similarity(So, Ss, dim=-1).mean())
    else:
        slot_cos = float("nan")

    # ── Belief cosine ─────────────────────────────────────────────────────────
    if K_cmp >= 2:
        Bo = torch.from_numpy(offline["B"][:K_cmp])
        Bs = torch.from_numpy(stream["B"][:K_cmp])
        belief_cos = float(F.cosine_similarity(Bo, Bs, dim=-1).mean())
    else:
        belief_cos = float("nan")

    # ── Prediction cosine ────────────────────────────────────────────────────
    K_pred = min(offline["pred"].shape[0], stream["pred"].shape[0])
    if K_pred >= 2:
        Po = torch.from_numpy(offline["pred"][:K_pred])
        Ps = torch.from_numpy(stream["pred"][:K_pred])
        pred_cos = float(F.cosine_similarity(Po, Ps, dim=-1).mean())
    else:
        pred_cos = float("nan")

    return {
        "T_offline":       fb_off.T,
        "T_stream":        fb_str.T,
        "T_cmp":           T_cmp,
        "T_drift":         fb_str.T - fb_off.T,
        "K_offline":       offline["K"],
        "K_stream":        stream["K"],
        "K_cmp":           K_cmp,
        "frame_cos":       frame_cos,
        "slot_cos":        slot_cos,
        "belief_cos":      belief_cos,
        "pred_cos":        pred_cos,
        "latency_ms":      fb_str.latency_ms,
        "wall_s_offline":  fb_off.wall_s,
        "wall_s_stream":   fb_str.wall_s,
        "rtf_stream":      fb_str.wall_s / fb_str.config.get("n_samples", 1) * TARGET_SR,
        "lookahead_ms":    fb_str.config.get("right_lookahead_ms", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(n_utt: int, chunk_ms: int, left_ms: int,
        lookaheads: list[int], out_path: Path) -> None:

    print("█" * 64)
    print("  PHASE 8 — INTEGRATED PIPELINE EVALUATION")
    print("  Offline vs Streaming: frame → slot → belief → prediction")
    print("█" * 64)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Device      : {device}")
    print(f"  Utterances  : {n_utt}")
    print(f"  Chunk       : {chunk_ms}ms   left_context: {left_ms}ms")
    print(f"  Lookaheads  : {lookaheads}")

    # ── Load models ──────────────────────────────────────────────────────────
    print("\n--- Loading WavLM-Large (one-time) ---")
    feat_extractor = AutoFeatureExtractor.from_pretrained(WAVLM_ID)
    wavlm = WavLMModel.from_pretrained(WAVLM_ID)   # type: ignore[arg-type]
    wavlm = wavlm.to(device)
    wavlm.eval()
    n_params = sum(p.numel() for p in wavlm.parameters())
    print(f"  WavLM-Large  ({n_params/1e6:.0f}M params)  device={device}")

    projector   = EvidenceProjector(D_RAW, D_PROJ)
    slotizer    = MeanPoolSlotizer()
    belief_gru  = BeliefTransitionGRU(d=D_PROJ)

    # Attempt to load trained weights
    use_trained = False
    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, weights_only=True, map_location="cpu")
        projector.load_state_dict(ckpt["projector"])
        belief_gru.load_state_dict(ckpt["belief_model"])
        use_trained = True
        print(f"  Loaded trained weights from {CKPT_PATH.name}  "
              f"(epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f})")
    else:
        print(f"  WARNING: {CKPT_PATH.name} not found — using random-init weights.")
        print(f"  Frame/slot/belief cosines still valid; pred cosine will be arbitrary.")

    projector  = projector.to(device).eval()
    belief_gru = belief_gru.to(device).eval()

    # ── Load utterances ───────────────────────────────────────────────────────
    print("\n--- Loading utterances ---")
    utterances = load_utterances(n_utt)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    all_records: list[dict] = []
    print("\n--- Running evaluation ---")
    print(f"  {'utt':12s}  {'la_ms':6s}  {'frame_cos':9s}  "
          f"{'slot_cos':8s}  {'belief_cos':10s}  {'pred_cos':8s}  "
          f"{'T_off':5s}  {'T_str':5s}  {'K_off':5s}  {'K_str':5s}  "
          f"{'lat_ms':6s}  {'RTF':5s}")
    print("  " + "─" * 108)

    for utt in utterances:
        waveform   = utt["waveform"]
        sr         = utt["sr"]
        duration_s = utt["duration_s"]
        utt_id     = utt["id"]

        # Offline features (computed once per utterance)
        fb_off = extract_offline_features(
            waveform, sr, projector, wavlm, feat_extractor, device,
        )

        # Syllable boundaries (computed once, shared between all paths)
        boundaries, seg_method = detect_syllables(waveform, sr)
        T_max = min(fb_off.T, 10000)   # generous upper bound

        # Offline downstream rollout
        ds_off = roll_downstream(
            fb_off.E_t, boundaries, T_max,
            slotizer, belief_gru, device,
        )

        for la in lookaheads:
            fb_str = extract_streaming_features(
                waveform, sr, projector, wavlm, feat_extractor, device,
                chunk_ms=chunk_ms,
                left_context_ms=left_ms,
                right_lookahead_ms=la,
            )

            T_cmp_limit = min(fb_off.T, fb_str.T)
            ds_str = roll_downstream(
                fb_str.E_t, boundaries, T_cmp_limit,
                slotizer, belief_gru, device,
            )

            metrics = compare_bundles(ds_off, ds_str, fb_off, fb_str)
            metrics.update({
                "utt_id":     utt_id,
                "duration_s": duration_s,
                "seg_method": seg_method,
                "use_trained_weights": use_trained,
            })
            all_records.append(metrics)

            rtf = fb_str.wall_s / duration_s
            print(
                f"  {utt_id:12s}  "
                f"la={la:3d}ms  "
                f"{metrics['frame_cos']:9.4f}  "
                f"{metrics['slot_cos']:8.4f}  "
                f"{metrics['belief_cos']:10.4f}  "
                f"{metrics['pred_cos']:8.4f}  "
                f"{metrics['T_offline']:5d}  "
                f"{metrics['T_stream']:5d}  "
                f"{metrics['K_offline']:5d}  "
                f"{metrics['K_stream']:5d}  "
                f"{la+chunk_ms:6.0f}  "
                f"{rtf:5.3f}"
            )

    # ── Aggregate by lookahead ────────────────────────────────────────────────
    print()
    print("─" * 64)
    print("  Aggregate (mean over utterances)")
    print("─" * 64)
    print(f"  {'la_ms':6s}  {'frame_cos':9s}  {'slot_cos':8s}  "
          f"{'belief_cos':10s}  {'pred_cos':8s}  {'lat_ms':6s}")
    print("  " + "─" * 64)

    aggregates: dict[int, dict] = {}
    for la in lookaheads:
        recs = [r for r in all_records if r["lookahead_ms"] == la]
        def mean_metric(key: str) -> float:
            vals = [r[key] for r in recs if not np.isnan(r[key])]
            return float(np.mean(vals)) if vals else float("nan")

        agg = {
            "n_utt":       len(recs),
            "lookahead_ms": la,
            "latency_ms":  la + chunk_ms,
            "frame_cos":   mean_metric("frame_cos"),
            "slot_cos":    mean_metric("slot_cos"),
            "belief_cos":  mean_metric("belief_cos"),
            "pred_cos":    mean_metric("pred_cos"),
        }
        aggregates[la] = agg
        print(
            f"  la={la:3d}ms  "
            f"{agg['frame_cos']:9.4f}  "
            f"{agg['slot_cos']:8.4f}  "
            f"{agg['belief_cos']:10.4f}  "
            f"{agg['pred_cos']:8.4f}  "
            f"{agg['latency_ms']:6.0f}ms"
        )

    # ── Save results ──────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "meta": {
            "n_utt":        n_utt,
            "chunk_ms":     chunk_ms,
            "left_ms":      left_ms,
            "lookaheads":   lookaheads,
            "use_trained_weights": use_trained,
            "device":       str(device),
        },
        "per_utterance": all_records,
        "aggregate":     {str(k): v for k, v in aggregates.items()},
    }
    # Strip numpy arrays from per_utterance records before JSON serialisation
    clean_records = []
    for r in all_records:
        clean_records.append({
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in r.items()
        })
    output["per_utterance"] = clean_records

    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n  Results saved → {out_path}")

    # ── Key answers ───────────────────────────────────────────────────────────
    print()
    print("─" * 64)
    print("  Key answers")
    print("─" * 64)
    if 40 in aggregates:
        a = aggregates[40]
        print(f"  Q1. Frame cosine  (la=40ms) : {a['frame_cos']:.4f}")
        print(f"  Q2. Slot cosine   (la=40ms) : {a['slot_cos']:.4f}")
        print(f"  Q3. Belief cosine (la=40ms) : {a['belief_cos']:.4f}")
        print(f"  Q4. Pred cosine   (la=40ms) : {a['pred_cos']:.4f}")
        print(f"  → Downstream belief states are "
              f"{'HIGHLY ALIGNED' if a['belief_cos'] > 0.9 else 'MODERATELY ALIGNED' if a['belief_cos'] > 0.7 else 'POORLY ALIGNED'} "
              f"with offline at la=40ms")
    if 0 in aggregates and 40 in aggregates:
        gap = aggregates[40]["belief_cos"] - aggregates[0]["belief_cos"]
        print(f"  Q5. Belief cos gain 0→40ms  : {gap:+.4f}")
    if not use_trained:
        print()
        print("  NOTE: Random-init weights used — pred_cos values are not meaningful.")
        print("        Place belief_model_best.pt at data/checkpoints/ for trained comparison.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Integrated offline-vs-streaming HuPER pipeline evaluation"
    )
    ap.add_argument("--n_utt",      type=int,   default=2,
                    help="Number of LibriSpeech val utterances (default: 2)")
    ap.add_argument("--chunk",      type=int,   default=320,
                    help="Chunk size in ms (default: 320)")
    ap.add_argument("--left",       type=int,   default=640,
                    help="Left context in ms (default: 640)")
    ap.add_argument("--lookaheads", type=str,   default="40",
                    help="Comma-separated lookahead values in ms (default: '40')")
    ap.add_argument("--out",        type=Path,
                    default=ROOT_DIR / "data" / "figures_phase8_final" / "streaming_pipeline_eval.json",
                    help="Output JSON path")
    args = ap.parse_args()

    lookaheads = [int(x.strip()) for x in args.lookaheads.split(",")]

    run(
        n_utt      = args.n_utt,
        chunk_ms   = args.chunk,
        left_ms    = args.left,
        lookaheads = lookaheads,
        out_path   = args.out,
    )
