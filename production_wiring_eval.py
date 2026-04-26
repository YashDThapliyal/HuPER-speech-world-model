"""
production_wiring_eval.py
Phase 8 rerun through the production pipeline API.

Replaces the direct feature_sources + roll_downstream harness with calls to
SpeechWorldModelPipeline.run_full(mode=...). Produces the same four
cosine metrics (frame, slot, belief, pred) so the numbers are directly
comparable to streaming_pipeline_eval.json.

Both modes share the same EvidenceProjector and BeliefTransitionGRU weights
(loaded from checkpoint if present) so the comparison isolates the
windowed-attention effect, not weight drift.

Usage
-----
  python production_wiring_eval.py                        # 2 utts, la=40ms
  python production_wiring_eval.py --n_utt 2 --lookaheads 40
  python production_wiring_eval.py --n_utt 3 --lookaheads 0,40,160
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent / "src"))
from pipeline import build_pipeline, TARGET_SR

ROOT_DIR  = Path(__file__).parent
CKPT_PATH = ROOT_DIR / "data" / "checkpoints" / "belief_model_best.pt"
OUT_JSON  = ROOT_DIR / "data" / "figures_phase8_final" / "production_wiring_eval.json"
SAMPLE_WAV = ROOT_DIR / "data" / "samples" / "librispeech_sample.wav"


# ─────────────────────────────────────────────────────────────────────────────
# Utterance loading (mirrors streaming_pipeline_eval.load_utterances)
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
        print(f"  WARNING: LibriSpeech stream failed ({exc})")
        print(f"  Falling back to bundled sample: {SAMPLE_WAV}")
        if not SAMPLE_WAV.exists():
            raise FileNotFoundError(f"Bundled sample not found: {SAMPLE_WAV}") from exc
        wav, _ = librosa.load(str(SAMPLE_WAV), sr=TARGET_SR, mono=True)
        wav = wav.astype(np.float32)
        return [{"id": "local_sample", "waveform": wav, "sr": TARGET_SR,
                 "text": "", "duration_s": len(wav) / TARGET_SR}] * min(n, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Cosine helpers
# ─────────────────────────────────────────────────────────────────────────────

def cos_mean(A: np.ndarray, B: np.ndarray) -> float:
    """Mean frame-wise cosine similarity, comparing up to min(A.rows, B.rows)."""
    K = min(A.shape[0], B.shape[0])
    if K < 2:
        return float("nan")
    a = torch.from_numpy(A[:K])
    b = torch.from_numpy(B[:K])
    return float(F.cosine_similarity(a, b, dim=-1).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(n_utt: int, lookaheads: list[int]) -> None:
    print("█" * 64)
    print("  PRODUCTION WIRING EVAL — Phase 8 via run_full(mode=...)")
    print("  Offline vs Streaming through SpeechWorldModelPipeline")
    print("█" * 64)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Device      : {device}")
    print(f"  Utterances  : {n_utt}")
    print(f"  Lookaheads  : {lookaheads}")

    # Build production pipeline
    pipeline = build_pipeline(pooling="mean", device=device)

    # Load trained weights if available (identical logic to streaming_pipeline_eval)
    use_trained = False
    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, weights_only=True, map_location="cpu")
        pipeline.projector.load_state_dict(ckpt["projector"])
        pipeline.belief_model.load_state_dict(ckpt["belief_model"])
        pipeline.projector.to(device).eval()
        pipeline.belief_model.to(device).eval()
        use_trained = True
        print(f"  Loaded trained weights from {CKPT_PATH.name}  "
              f"(epoch={ckpt['epoch']}  val_loss={ckpt['val_loss']:.6f})")
    else:
        print(f"  WARNING: {CKPT_PATH.name} not found — using random-init weights.")

    # Load utterances
    print("\n--- Loading utterances ---")
    utterances = load_utterances(n_utt)

    # Evaluate — write temp wav per utterance (run_full takes a path)
    all_records: list[dict] = []
    print("\n--- Running evaluation via run_full() ---")
    print(f"  {'utt':12s}  {'la_ms':6s}  {'frame_cos':9s}  "
          f"{'slot_cos':8s}  {'belief_cos':10s}  {'pred_cos':8s}  "
          f"{'T_off':5s}  {'T_str':5s}  {'K_off':5s}  {'K_str':5s}  "
          f"{'lat_ms':6s}  {'RTF':5s}")
    print("  " + "─" * 104)

    wav_dir = ROOT_DIR / "data" / "cache" / "prod_eval_wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    for utt in utterances:
        wav_path = wav_dir / f"{utt['id']}.wav"
        sf.write(str(wav_path), utt["waveform"], utt["sr"])

        # Offline run
        r_off = pipeline.run_full(wav_path, utt_id=utt["id"] + "_off", mode="offline")

        for la in lookaheads:
            r_str = pipeline.run_full(
                wav_path,
                utt_id       = utt["id"] + f"_str_la{la}",
                mode         = "streaming",
                lookahead_ms = la,
            )

            rec = {
                "utt_id":             utt["id"],
                "duration_s":         utt["duration_s"],
                "lookahead_ms":       la,
                "T_offline":          r_off.T,
                "T_stream":           r_str.T,
                "T_drift":            r_str.T - r_off.T,
                "K_offline":          r_off.K,
                "K_stream":           r_str.K,
                "frame_cos":          cos_mean(r_off.E_t,         r_str.E_t),
                "slot_cos":           cos_mean(r_off.slots,       r_str.slots),
                "belief_cos":         cos_mean(r_off.beliefs,     r_str.beliefs),
                "pred_cos":           cos_mean(r_off.predictions, r_str.predictions),
                "latency_ms":         r_str.metadata["latency_ms"],
                "use_trained_weights": use_trained,
                "source":             "production_pipeline_api",
                "driver":             "SpeechWorldModelPipeline.run_full",
            }
            all_records.append(rec)

            rtf = r_str.metadata["source_wall_s"] / utt["duration_s"]
            print(
                f"  {utt['id']:12s}  "
                f"la={la:3d}ms  "
                f"{rec['frame_cos']:9.4f}  "
                f"{rec['slot_cos']:8.4f}  "
                f"{rec['belief_cos']:10.4f}  "
                f"{rec['pred_cos']:8.4f}  "
                f"{rec['T_offline']:5d}  "
                f"{rec['T_stream']:5d}  "
                f"{rec['K_offline']:5d}  "
                f"{rec['K_stream']:5d}  "
                f"{rec['latency_ms']:6.0f}  "
                f"{rtf:5.3f}"
            )

    # Aggregate by lookahead
    print()
    print("─" * 64)
    print("  Aggregate (mean over utterances)")
    print("─" * 64)
    print(f"  {'la_ms':6s}  {'frame_cos':9s}  {'slot_cos':8s}  "
          f"{'belief_cos':10s}  {'pred_cos':8s}  {'lat_ms':6s}")
    print("  " + "─" * 64)

    aggregates: dict[str, dict] = {}
    for la in lookaheads:
        recs = [r for r in all_records if r["lookahead_ms"] == la]

        def mean_metric(key: str) -> float:
            vals = [r[key] for r in recs if not np.isnan(float(r[key]))]
            return float(np.mean(vals)) if vals else float("nan")

        agg = {
            "n_utt":        len(recs),
            "lookahead_ms": la,
            "latency_ms":   la + 320,   # chunk_ms default is 320
            "frame_cos":    mean_metric("frame_cos"),
            "slot_cos":     mean_metric("slot_cos"),
            "belief_cos":   mean_metric("belief_cos"),
            "pred_cos":     mean_metric("pred_cos"),
        }
        aggregates[str(la)] = agg
        print(
            f"  la={la:3d}ms  "
            f"{agg['frame_cos']:9.4f}  "
            f"{agg['slot_cos']:8.4f}  "
            f"{agg['belief_cos']:10.4f}  "
            f"{agg['pred_cos']:8.4f}  "
            f"{agg['latency_ms']:6.0f}ms"
        )

    # Save results
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "meta": {
            "n_utt":        n_utt,
            "lookaheads":   lookaheads,
            "use_trained_weights": use_trained,
            "device":       str(device),
            "harness":      "production_wiring_eval.py",
            "driver":       "SpeechWorldModelPipeline.run_full(mode=...)",
        },
        "per_utterance": all_records,
        "aggregate":     aggregates,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(output, fh, indent=2)
    print(f"\n  Results saved → {OUT_JSON}")

    # Key answers
    print()
    print("─" * 64)
    print("  Key results (la=40ms)")
    print("─" * 64)
    if "40" in aggregates:
        agg40 = aggregates["40"]
        print(f"  frame_cos  = {agg40['frame_cos']:.4f}")
        print(f"  slot_cos   = {agg40['slot_cos']:.4f}")
        print(f"  belief_cos = {agg40['belief_cos']:.4f}")
        print(f"  pred_cos   = {agg40['pred_cos']:.4f}")
        print(f"  latency    = {agg40['latency_ms']:.0f}ms")


def main() -> None:
    ap = argparse.ArgumentParser(description="Production wiring eval via run_full()")
    ap.add_argument("--n_utt", type=int, default=2,
                    help="Number of utterances to evaluate")
    ap.add_argument("--lookaheads", type=str, default="40",
                    help="Comma-separated lookahead values in ms (default: 40)")
    args = ap.parse_args()
    lookaheads = [int(x) for x in args.lookaheads.split(",")]
    run(args.n_utt, lookaheads)


if __name__ == "__main__":
    main()
