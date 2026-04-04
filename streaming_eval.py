"""
streaming_eval.py
Phase 8 — Streaming HuPER Encoder: evaluation vs offline oracle.

For each of 8 LibriSpeech utterances, compares:
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
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoFeatureExtractor, WavLMModel
import librosa

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent / "src"))
from huper_features import EvidenceProjector
from streaming_encoder import (
    StreamingHuPEREncoder, StreamingConfig,
    make_eval_configs, extract_offline_oracle,
    WAVLM_ID, TARGET_SR, D_RAW, D_PROJ, FRAME_STRIDE,
)

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
FIG_DIR    = ROOT_DIR / "data" / "figures_phase8"
CACHE_DIR  = ROOT_DIR / "data" / "cache"
SAMPLE_WAV = ROOT_DIR / "data" / "samples" / "librispeech_sample.wav"

DARK_BG  = "#0e1117"
PANEL_BG = "#1a1d23"
SPINE_C  = "#333333"
TICK_C   = "#666666"
TEXT_C   = "#cccccc"


# ─────────────────────────────────────────────────────────────────────────────
# Styling helpers
# ─────────────────────────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:  # type: ignore[name-defined]
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TICK_C)
    ax.spines[:].set_color(SPINE_C)
    if title:  ax.set_title(title, color="white", fontsize=10, pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_C, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_C, fontsize=9)


def _save(fig: plt.Figure, name: str) -> None:  # type: ignore[name-defined]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / name
    fig.savefig(out, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_utterances(n: int) -> list[dict]:
    """
    Stream first n utterances from LibriSpeech validation split.
    Returns list of dicts: {id, waveform, sr, text, duration_s}.
    """
    print(f"  Streaming LibriSpeech validation [first {n} utterances] …")
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
            sr  = TARGET_SR
        items.append({
            "id":         f"val_{i:04d}",
            "waveform":   wav,
            "sr":         sr,
            "text":       item.get("text", ""),
            "duration_s": len(wav) / sr,
        })
        if (i + 1) % 4 == 0:
            print(f"    loaded {i+1}/{n} …")
    return items


# ─────────────────────────────────────────────────────────────────────────────
# Per-utterance evaluation
# ─────────────────────────────────────────────────────────────────────────────

def eval_utterance(
    item:        dict,
    encoder:     StreamingHuPEREncoder,
    cfg:         StreamingConfig,
    E_offline:   np.ndarray,             # (T, 256) oracle
) -> dict:
    """
    Run streaming encoder on one utterance and compute alignment metrics.

    Returns a dict with all per-utterance results.
    """
    waveform      = item["waveform"]
    duration_s    = item["duration_s"]
    chunk_samples = cfg.chunk_samples
    n_chunks      = int(np.ceil(len(waveform) / chunk_samples))

    encoder.reset()

    # Push audio chunk by chunk
    t0 = time.perf_counter()
    for i in range(n_chunks):
        chunk = waveform[i * chunk_samples : (i + 1) * chunk_samples]
        encoder.push_audio(chunk)
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
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    0:   "#4fc3f7",   # causal — blue
    40:  "#81c784",   # 40ms   — green
    80:  "#ffd740",   # 80ms   — amber
    160: "#ff7043",   # 160ms  — orange
}
LABELS = {0: "causal (0ms)", 40: "look-40ms", 80: "look-80ms", 160: "look-160ms"}


def viz8_1_frame_similarity(
    results_by_la: dict[int, dict],
    utt_id:        str,
    chunk_frames:  int,
) -> None:
    """Viz 8-1 — Framewise cosine similarity vs offline, one utterance."""
    fig, ax = plt.subplots(figsize=(16, 5), facecolor=DARK_BG)
    fig.suptitle(
        f"Viz 8-1 — Streaming vs Offline Framewise Cosine  [{utt_id}]",
        color="white", fontsize=12, fontweight="bold",
    )

    for la, res in sorted(results_by_la.items()):
        cos = res["cos_per_frame"]
        if len(cos) == 0:
            continue
        t = np.arange(len(cos)) * FRAME_STRIDE / TARGET_SR
        ax.plot(t, cos, color=COLORS[la], lw=0.8, alpha=0.85,
                label=f"{LABELS[la]}  μ={res['mean_cos']:.3f}")

    # Mark chunk boundaries
    ref_res = next(iter(results_by_la.values()))
    T_total = ref_res["T_cmp"]
    dur_s   = T_total * FRAME_STRIDE / TARGET_SR
    for i in range(1, ref_res["n_chunks"] + 1):
        t_bound = i * chunk_frames * FRAME_STRIDE / TARGET_SR
        if t_bound < dur_s:
            ax.axvline(t_bound, color="#444444", lw=0.6, linestyle="--", alpha=0.7)

    ax.axhline(1.0, color=SPINE_C, lw=0.5, linestyle=":")
    ax.set_xlim(0, dur_s)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=8, framealpha=0.3, facecolor=PANEL_BG,
              labelcolor="white", edgecolor=SPINE_C)
    _style_ax(ax, xlabel="Time (s)", ylabel="Cosine similarity to oracle")
    ax.text(0.99, 0.04, "dashed = chunk boundaries",
            transform=ax.transAxes, color="#666666", fontsize=7, ha="right")

    _save(fig, "viz8_1_frame_similarity.png")


def viz8_2_latency_quality(agg: dict[int, dict]) -> None:
    """Viz 8-2 — Latency vs quality trade-off across streaming modes."""
    la_vals    = sorted(agg.keys())
    mean_cos   = [agg[la]["mean_cos_mean"]  for la in la_vals]
    latencies  = [agg[la]["latency_ms"]     for la in la_vals]
    rtf_vals   = [agg[la]["rtf_mean"]       for la in la_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=DARK_BG,
                                    gridspec_kw={"wspace": 0.35})
    fig.suptitle("Viz 8-2 — Latency / Quality Trade-off",
                 color="white", fontsize=12, fontweight="bold")

    # Left: quality vs lookahead
    colors_list = [COLORS[la] for la in la_vals]
    bars1 = ax1.bar(np.arange(len(la_vals)), mean_cos, color=colors_list, width=0.55)
    ax1.set_xticks(np.arange(len(la_vals)))
    ax1.set_xticklabels([f"{la}ms" for la in la_vals], color=TEXT_C, fontsize=9)
    ax1.set_ylim(0, 1.05)
    for bar, v in zip(bars1, mean_cos):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                 f"{v:.3f}", ha="center", va="bottom", color="white", fontsize=8)
    _style_ax(ax1, title="Feature quality vs lookahead",
              xlabel="Right lookahead (ms)", ylabel="Mean cosine to oracle")

    # Right: latency + RTF
    x = np.arange(len(la_vals))
    ax2.bar(x - 0.2, latencies, color=colors_list, width=0.35, label="Latency (ms)")
    ax2r = ax2.twinx()
    ax2r.plot(x, rtf_vals, color="#ce93d8", marker="o", lw=1.5, ms=6, label="RTF")
    ax2r.set_ylim(0, max(rtf_vals) * 1.5)
    ax2r.tick_params(colors=TICK_C)
    ax2r.set_ylabel("Real-time factor", color=TEXT_C, fontsize=9)
    ax2r.spines[:].set_color(SPINE_C)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{la}ms" for la in la_vals], color=TEXT_C, fontsize=9)
    _style_ax(ax2, title="Latency and RTF vs lookahead",
              xlabel="Right lookahead (ms)", ylabel="Algorithmic latency (ms)")

    _save(fig, "viz8_2_latency_quality.png")


def viz8_3_feature_heatmap(res_causal: dict, res_la160: dict) -> None:
    """Viz 8-3 — Feature heatmap: offline / streaming / absolute difference."""
    E_off  = res_causal["E_offline"]    # (T_cmp, 256)
    E_c    = res_causal["E_stream"]     # (T_cmp, 256)
    E_la   = res_la160["E_stream"]      # (T_cmp2, 256)
    T      = min(E_off.shape[0], E_c.shape[0], E_la.shape[0], 200)

    Eo, Ec, El = E_off[:T].T, E_c[:T].T, E_la[:T].T   # (256, T)
    diff_c  = np.abs(Eo - Ec)
    diff_la = np.abs(Eo - El)

    vmin, vmax = np.percentile(Eo, 2), np.percentile(Eo, 98)
    dmax = max(float(diff_c.max()), float(diff_la.max()), 1e-6)

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), facecolor=DARK_BG,
                             gridspec_kw={"hspace": 0.45})
    fig.suptitle("Viz 8-3 — Feature Heatmap Comparison  (first 200 frames)",
                 color="white", fontsize=12, fontweight="bold")

    def _imshow(ax: plt.Axes, data: np.ndarray, title: str,  # type: ignore[name-defined]
                cmap: str, vn: float, vx: float) -> None:
        ax.imshow(data, aspect="auto", origin="lower", cmap=cmap,
                  vmin=vn, vmax=vx, interpolation="nearest",
                  extent=(0, T * FRAME_STRIDE / TARGET_SR, 0, 256))
        _style_ax(ax, title=title, xlabel="Time (s)", ylabel="Dim")

    _imshow(axes[0], Eo, "Offline oracle E_t (full context)", "RdBu_r", vmin, vmax)
    _imshow(axes[1], Ec, f"Streaming causal (0ms lookahead)  μcos={res_causal['mean_cos']:.3f}",
            "RdBu_r", vmin, vmax)
    _imshow(axes[2], diff_c, "|offline − causal|", "hot", 0, dmax)
    _imshow(axes[3], diff_la, f"|offline − 160ms lookahead|  μcos={res_la160['mean_cos']:.3f}",
            "hot", 0, dmax)

    _save(fig, "viz8_3_feature_heatmap.png")


def viz8_4_boundary_zoom(res: dict) -> None:
    """Viz 8-4 — Zoomed cosine similarity around several chunk boundaries."""
    cos         = res["cos_per_frame"]    # (T_cmp,)
    chunk_frames = res["chunk_frames"]
    n_show      = 4
    half        = 12   # frames on each side of boundary

    if len(cos) < half * 2 or chunk_frames < 1:
        return

    # Pick up to n_show boundaries
    boundaries = [i * chunk_frames for i in range(1, res["n_chunks"] + 1)
                  if half < i * chunk_frames < len(cos) - half][:n_show]
    if not boundaries:
        return

    n_cols = len(boundaries)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4), facecolor=DARK_BG,
                             gridspec_kw={"wspace": 0.35})
    if n_cols == 1:
        axes = [axes]   # type: ignore[assignment]
    fig.suptitle(
        f"Viz 8-4 — Chunk Boundary Effect  [{res['utt_id']}  la={res['lookahead_ms']}ms]"
        f"  boundary_drop={res['boundary_drop']:.4f}",
        color="white", fontsize=11, fontweight="bold",
    )

    for ax, bf in zip(axes, boundaries):
        lo, hi = bf - half, bf + half
        x = np.arange(lo, hi)
        ax.plot(x, cos[lo:hi], color=COLORS.get(res["lookahead_ms"], "#4fc3f7"),
                lw=1.2, marker="o", ms=3)
        ax.axvline(bf, color="#ff7043", lw=1.0, linestyle="--",
                   label=f"boundary k={bf}")
        ax.axhline(res["mean_cos"], color="white", lw=0.6, linestyle=":",
                   label=f"mean={res['mean_cos']:.3f}")
        ax.set_ylim(-0.1, 1.05)
        ax.legend(fontsize=7, framealpha=0.3, facecolor=PANEL_BG,
                  labelcolor="white", edgecolor=SPINE_C)
        t_bound = bf * FRAME_STRIDE / TARGET_SR
        _style_ax(ax, title=f"Boundary at t={t_bound:.2f}s",
                  xlabel="Frame index", ylabel="Cosine to oracle")

    _save(fig, "viz8_4_boundary_zoom.png")


def viz8_5_aggregate(
    agg:         dict[int, dict],
    all_results: dict[int, list[dict]],
) -> None:
    """Viz 8-5 — Aggregate cosine per mode (boxplot) + RTF bar chart."""
    la_vals = sorted(agg.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=DARK_BG,
                                    gridspec_kw={"wspace": 0.35})
    fig.suptitle("Viz 8-5 — Aggregate Results Across Utterances",
                 color="white", fontsize=12, fontweight="bold")

    # Boxplot of per-utterance mean cosine for each mode
    data_cos = [[r["mean_cos"] for r in all_results[la]] for la in la_vals]
    bp = ax1.boxplot(
        data_cos,
        patch_artist=True,
        medianprops=dict(color="white", lw=1.5),
        whiskerprops=dict(color=SPINE_C),
        capprops=dict(color=SPINE_C),
        flierprops=dict(marker="o", color=SPINE_C, ms=4),
    )
    for patch, la in zip(bp["boxes"], la_vals):
        patch.set_facecolor(COLORS[la])
        patch.set_alpha(0.8)

    ax1.set_xticks(np.arange(1, len(la_vals) + 1))
    ax1.set_xticklabels([LABELS[la] for la in la_vals], color=TEXT_C, fontsize=8)
    ax1.set_ylim(0, 1.05)
    _style_ax(ax1, title="Cosine to oracle — distribution over utterances",
              xlabel="Streaming mode", ylabel="Mean cosine (per utterance)")

    # RTF bar chart
    rtf_vals = [agg[la]["rtf_mean"] for la in la_vals]
    colors_l  = [COLORS[la] for la in la_vals]
    ax2.bar(np.arange(len(la_vals)), rtf_vals, color=colors_l, width=0.55)
    ax2.axhline(1.0, color="white", lw=0.8, linestyle=":", label="RTF = 1 (real-time)")
    ax2.set_xticks(np.arange(len(la_vals)))
    ax2.set_xticklabels([LABELS[la] for la in la_vals], color=TEXT_C, fontsize=8)
    ax2.legend(fontsize=8, framealpha=0.3, facecolor=PANEL_BG,
               labelcolor="white", edgecolor=SPINE_C)
    _style_ax(ax2, title="Real-time factor per mode  (< 1 = faster than real time)",
              xlabel="Streaming mode", ylabel="RTF")

    _save(fig, "viz8_5_aggregate.png")


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

    # ── Visualizations ────────────────────────────────────────────────────────
    print("\n--- Generating visualizations ---")

    # Representative utterance = the first one
    rep_utt     = utterances[0]
    chunk_frames = configs[0].chunk_samples // FRAME_STRIDE

    # Build per-la result dicts for the representative utterance
    rep_by_la = {la: all_results[la][0] for la in sorted(all_results.keys())}

    viz8_1_frame_similarity(rep_by_la, rep_utt["id"], chunk_frames)
    viz8_2_latency_quality(agg)
    viz8_3_feature_heatmap(
        res_causal = all_results[0][0],
        res_la160  = all_results[160][0],
    )
    viz8_4_boundary_zoom(all_results[0][0])   # causal mode for boundary zoom
    viz8_5_aggregate(agg, all_results)

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
    print(f"  ├── Figures                   : {FIG_DIR}")
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
