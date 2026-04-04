"""
streaming_eval.py
Phase 8 — Streaming HuPER Encoder: evaluation vs offline oracle.

For each of N LibriSpeech utterances, compares:
  E_offline : (T, 256)  full-context WavLM, full utterance (oracle)
  E_stream  : (T', 256) windowed WavLM, causal or bounded-lookahead

Streaming configs evaluated
────────────────────────────
  causal       right_lookahead =   0ms  (strictly causal)
  lookahead40  right_lookahead =  40ms
  lookahead80  right_lookahead =  80ms
  lookahead160 right_lookahead = 160ms

All configs share:
  chunk_size_ms = 320ms   hop_size_ms = 320ms
  left_context_ms = 640ms

Metrics
───────
  1. Framewise cosine similarity  mean cos(E_stream_t, E_offline_t)
  2. MSE between aligned frames
  3. Boundary instability  local cos dip at chunk boundaries
  4. Algorithmic latency (ms)
  5. Real-time factor  wall_time / audio_duration

Usage
─────
  python streaming_eval.py
  python streaming_eval.py --n_utt 4     # faster
  python streaming_eval.py --chunk 160   # smaller chunks

Note: for polished figures and the full findings writeup, run make_phase8_report.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoFeatureExtractor, WavLMModel
import librosa

sys.path.insert(0, str(Path(__file__).parent / "src"))
from huper_features import EvidenceProjector
from streaming_encoder import (
    StreamingHuPEREncoder, StreamingConfig,
    make_eval_configs, extract_offline_oracle,
    WAVLM_ID, TARGET_SR, D_RAW, D_PROJ, FRAME_STRIDE,
)

ROOT_DIR = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_utterances(n: int) -> list[dict]:
    """Stream first n utterances from LibriSpeech validation split."""
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
    print(f"  Loaded {len(items)} utterances")
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Per-utterance evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_utterance(
    item:      dict,
    encoder:   StreamingHuPEREncoder,
    cfg:       StreamingConfig,
    E_offline: np.ndarray,   # (T, 256) oracle
) -> dict:
    """Run streaming encoder on one utterance and return alignment metrics."""
    waveform      = item["waveform"]
    duration_s    = item["duration_s"]
    chunk_samples = cfg.chunk_samples
    n_chunks      = int(np.ceil(len(waveform) / chunk_samples))

    encoder.reset()
    t0 = time.perf_counter()
    for i in range(n_chunks):
        encoder.push_audio(waveform[i * chunk_samples : (i + 1) * chunk_samples])
    encoder.flush()
    wall_s = time.perf_counter() - t0

    E_stream = encoder.get_emitted_features()   # (T', 256)

    # Alignment: compare first T_cmp frames
    T_cmp = min(E_stream.shape[0], E_offline.shape[0])
    if T_cmp < 2:
        return _empty_result(item["id"], cfg, duration_s, wall_s)

    Es = torch.from_numpy(E_stream[:T_cmp])     # (T_cmp, 256)
    Eo = torch.from_numpy(E_offline[:T_cmp])    # (T_cmp, 256)

    cos_per_frame = F.cosine_similarity(Es, Eo, dim=-1).numpy()   # (T_cmp,)
    mse_per_frame = ((E_stream[:T_cmp] - E_offline[:T_cmp]) ** 2).mean(axis=1)

    # Boundary instability: compare cos at boundary frames vs neighbours
    boundary_frames = [
        i * chunk_samples // FRAME_STRIDE
        for i in range(1, n_chunks)
        if i * chunk_samples // FRAME_STRIDE < T_cmp
    ]
    if boundary_frames:
        boundary_cos    = float(np.mean([cos_per_frame[min(f, T_cmp-1)] for f in boundary_frames]))
        non_boundary    = np.delete(cos_per_frame, [min(f, T_cmp-1) for f in boundary_frames if f < T_cmp])
        non_boundary_cos = float(non_boundary.mean()) if len(non_boundary) > 0 else float(cos_per_frame.mean())
        boundary_drop   = non_boundary_cos - boundary_cos   # positive = dip at boundary
    else:
        boundary_cos     = float(cos_per_frame.mean())
        non_boundary_cos = boundary_cos
        boundary_drop    = 0.0

    rtf = wall_s / duration_s

    return {
        "utt_id":           item["id"],
        "mode":             cfg.mode,
        "lookahead_ms":     cfg.right_lookahead_ms,
        "latency_ms":       cfg.algorithmic_latency_ms,
        "T_offline":        E_offline.shape[0],
        "T_stream":         E_stream.shape[0],
        "T_cmp":            T_cmp,
        "duration_s":       duration_s,
        "mean_cos":         float(cos_per_frame.mean()),
        "min_cos":          float(cos_per_frame.min()),
        "mse_mean":         float(mse_per_frame.mean()),
        "boundary_cos":     boundary_cos,
        "non_boundary_cos": non_boundary_cos,
        "boundary_drop":    boundary_drop,
        "rtf":              rtf,
        "wall_s":           wall_s,
        "cos_per_frame":    cos_per_frame,   # (T_cmp,) for plotting
        "n_chunks":         n_chunks,
        "chunk_frames":     chunk_samples // FRAME_STRIDE,
        "E_stream":         E_stream[:T_cmp],
        "E_offline":        E_offline[:T_cmp],
    }


def _empty_result(utt_id: str, cfg: StreamingConfig, dur: float, wall: float) -> dict:
    return {
        "utt_id": utt_id, "mode": cfg.mode,
        "lookahead_ms": cfg.right_lookahead_ms, "latency_ms": cfg.algorithmic_latency_ms,
        "T_offline": 0, "T_stream": 0, "T_cmp": 0, "duration_s": dur,
        "mean_cos": 0.0, "min_cos": 0.0, "mse_mean": 0.0,
        "boundary_cos": 0.0, "non_boundary_cos": 0.0, "boundary_drop": 0.0,
        "rtf": wall / dur, "wall_s": wall,
        "cos_per_frame": np.array([]), "n_chunks": 0, "chunk_frames": 0,
        "E_stream": np.empty((0, D_PROJ)), "E_offline": np.empty((0, D_PROJ)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(n_utt: int, chunk_ms: int) -> None:
    print("█" * 60)
    print("  PHASE 8 — STREAMING EVALUATION")
    print("  StreamingHuPEREncoder vs offline oracle")
    print("█" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Device: {device}")
    print(f"  Utterances: {n_utt}   chunk_size_ms: {chunk_ms}ms")

    # ── Load utterances ───────────────────────────────────────────────────────
    print("\n--- Loading utterances ---")
    utterances = load_utterances(n_utt)
    print(f"  Loaded {len(utterances)} utterances  "
          f"(dur range: {min(u['duration_s'] for u in utterances):.1f}–"
          f"{max(u['duration_s'] for u in utterances):.1f}s)")

    # ── Load WavLM + projector once ───────────────────────────────────────────
    print("\n--- Loading WavLM-Large (one-time) ---")
    feat_extractor = AutoFeatureExtractor.from_pretrained(WAVLM_ID)
    wavlm = WavLMModel.from_pretrained(WAVLM_ID)   # type: ignore[arg-type]
    wavlm = wavlm.to(device)
    wavlm.eval()
    n_params = sum(p.numel() for p in wavlm.parameters())
    print(f"  WavLM-Large loaded  ({n_params/1e6:.0f}M params)  device={device}")

    projector = EvidenceProjector(D_RAW, D_PROJ).to(device)
    projector.eval()
    print(f"  EvidenceProjector  ({sum(p.numel() for p in projector.parameters()):,} params)")

    # ── Build streaming configs ───────────────────────────────────────────────
    configs = make_eval_configs(
        chunk_size_ms   = chunk_ms,
        left_context_ms = chunk_ms * 2,   # 2× chunk for left context
    )
    print(f"\n  Streaming configs:")
    for cfg in configs:
        print(f"    la={cfg.right_lookahead_ms:3d}ms  "
              f"window={cfg.window_size_ms}ms  "
              f"latency={cfg.algorithmic_latency_ms:.0f}ms")

    # Build one encoder per config (all share the same WavLM)
    encoders = [
        StreamingHuPEREncoder(cfg, device, projector=projector,
                              wavlm_model=wavlm, feat_extractor=feat_extractor)
        for cfg in configs
    ]

    # ── Compute offline oracle features for all utterances ───────────────────
    print("\n--- Computing offline oracle features ---")
    oracle_encoder = StreamingHuPEREncoder(
        configs[0], device, projector=projector,
        wavlm_model=wavlm, feat_extractor=feat_extractor,
    )

    E_offline_all: list[np.ndarray] = []
    for utt in utterances:
        t0     = time.perf_counter()
        E_off  = extract_offline_oracle(utt["waveform"], device, projector, oracle_encoder)
        dt     = time.perf_counter() - t0
        E_offline_all.append(E_off)
        print(f"  {utt['id']}  T={E_off.shape[0]:4d}  dur={utt['duration_s']:.2f}s  "
              f"wall={dt:.2f}s")

    # ── Run streaming evaluation ──────────────────────────────────────────────
    all_results:   dict[int, list[dict]] = {cfg.right_lookahead_ms: [] for cfg in configs}

    for utt, E_off in zip(utterances, E_offline_all):
        print(f"\n  [{utt['id']}]  dur={utt['duration_s']:.2f}s  T_oracle={E_off.shape[0]}")
        for cfg, enc in zip(configs, encoders):
            res = eval_utterance(utt, enc, cfg, E_off)
            all_results[cfg.right_lookahead_ms].append(res)
            print(f"    la={cfg.right_lookahead_ms:3d}ms  "
                  f"T_stream={res['T_stream']:4d}  "
                  f"cos={res['mean_cos']:.4f}  "
                  f"mse={res['mse_mean']:.5f}  "
                  f"bdrop={res['boundary_drop']:.4f}  "
                  f"RTF={res['rtf']:.3f}")

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    agg: dict[int, dict] = {}
    for la, results in all_results.items():
        agg[la] = {
            "latency_ms":     configs[[c.right_lookahead_ms for c in configs].index(la)].algorithmic_latency_ms,
            "mean_cos_mean":  float(np.mean([r["mean_cos"]  for r in results])),
            "mean_cos_std":   float(np.std( [r["mean_cos"]  for r in results])),
            "mse_mean":       float(np.mean([r["mse_mean"]  for r in results])),
            "boundary_drop":  float(np.mean([r["boundary_drop"] for r in results])),
            "rtf_mean":       float(np.mean([r["rtf"]       for r in results])),
        }

    # ── Print summary ─────────────────────────────────────────────────────────
    best_la  = max(agg.keys(), key=lambda la: agg[la]["mean_cos_mean"])
    worst_la = min(agg.keys(), key=lambda la: agg[la]["mean_cos_mean"])

    print()
    print("─" * 62)
    print("  Phase 8 Summary")
    print("─" * 62)
    print(f"  ├── Num utterances evaluated  : {n_utt}")
    print(f"  ├── Streaming modes tested    : causal(0ms), 40ms, 80ms, 160ms")
    print(f"  ├── Chunk size                : {chunk_ms}ms")
    print(f"  ├── Left context              : {chunk_ms*2}ms")
    print()
    for la in sorted(agg.keys()):
        a = agg[la]
        print(f"  │  la={la:3d}ms  cos={a['mean_cos_mean']:.4f}±{a['mean_cos_std']:.4f}  "
              f"mse={a['mse_mean']:.5f}  "
              f"bdrop={a['boundary_drop']:.4f}  "
              f"latency={a['latency_ms']:.0f}ms  "
              f"RTF={a['rtf_mean']:.3f}")
    print()
    print(f"  ├── Best mean cosine to oracle : la={best_la}ms  "
          f"cos={agg[best_la]['mean_cos_mean']:.4f}")
    print(f"  ├── Worst mean cosine to oracle: la={worst_la}ms  "
          f"cos={agg[worst_la]['mean_cos_mean']:.4f}")
    print(f"  ├── Best latency setting       : la=0ms  "
          f"latency={agg[0]['latency_ms']:.0f}ms  cos={agg[0]['mean_cos_mean']:.4f}")
    print(f"  ├── Mean RTF (causal)          : {agg[0]['rtf_mean']:.3f}  "
          f"({'< 1 = real-time capable' if agg[0]['rtf_mean'] < 1 else '> 1 = slower than real-time'})")
    print(f"  └── For figures: run make_phase8_report.py")
    print()

    gain_160 = agg[160]["mean_cos_mean"] - agg[0]["mean_cos_mean"]
    gain_80  = agg[80]["mean_cos_mean"]  - agg[0]["mean_cos_mean"]

    print("  Answers to key questions:")
    print(f"  │")
    usable = agg[0]['mean_cos_mean'] > 0.5
    print(f"  │ Q1. Can the streaming encoder emit useful low-latency features?")
    print(f"  │     cos(causal, oracle) = {agg[0]['mean_cos_mean']:.4f}  "
          f"→  {'YES — features are correlated with oracle' if usable else 'MARGINAL — features differ significantly'}")
    print(f"  │")
    print(f"  │ Q2. Quality loss under strict causal inference (0ms lookahead)?")
    print(f"  │     cosine drop vs best = {agg[best_la]['mean_cos_mean'] - agg[0]['mean_cos_mean']:.4f}  "
          f"(absolute)  MSE={agg[0]['mse_mean']:.5f}")
    print(f"  │")
    print(f"  │ Q3. Does small bounded lookahead help?")
    print(f"  │     +40ms → +{agg[40]['mean_cos_mean'] - agg[0]['mean_cos_mean']:.4f}  "
          f" +80ms → +{gain_80:.4f}   +160ms → +{gain_160:.4f}")
    print(f"  └─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_utt",  type=int, default=8,
                    help="Number of LibriSpeech val utterances to evaluate")
    ap.add_argument("--chunk",  type=int, default=320,
                    help="Chunk size in ms")
    args = ap.parse_args()
    run(args.n_utt, args.chunk)
