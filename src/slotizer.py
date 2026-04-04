"""
Module: slotizer
Phase: 5
Goal: compress E_t (T, 256) @ 50Hz → S_k (K, 256) @ ~5Hz.
      Every frame belongs to exactly one syllable slot.
      The Slotizer summarises what was heard in each syllable as a single vector.

What this teaches:
- Mean pooling gives every frame equal weight within a slot
- Attention pooling uses a learned query to up-weight the most informative frames
  (typically the vowel nucleus — peak energy — or the onset transient)
- The choice of pooling strategy is Phase 5's architectural decision;
  Phase 6 BeliefTransition receives S_k regardless of which strategy produced it
- The PCA slot trajectory shows sequential structure the GRU can exploit
"""

# --- Imports ---
from __future__ import annotations

import pickle
from pathlib import Path

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, WavLMModel

matplotlib.use("Agg")

# --- Constants ---
TARGET_SR  = 16_000
HOP        = 320          # 20ms — 50Hz frame rate
WAVLM_ID   = "microsoft/wavlm-large"
D_PROJ     = 256

DARK_BG  = "#0e1117"
PANEL_BG = "#1a1d23"
SPINE_C  = "#333333"
TICK_C   = "#666666"
TEXT_C   = "#cccccc"

VIZ_DIR  = Path(__file__).parent.parent / "data" / "visualizations" / "phase5"
DATA_DIR = Path(__file__).parent.parent / "data"


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def log_shape(name: str, x: torch.Tensor | np.ndarray) -> None:
    if isinstance(x, torch.Tensor):
        print(f"  {name:40s}: {tuple(x.shape)}  dtype={x.dtype}  device={x.device}")
    else:
        print(f"  {name:40s}: shape={x.shape}  dtype={x.dtype}")


def _style_ax(
    ax: plt.Axes,   # type: ignore[name-defined]
    title: str = "", xlabel: str = "", ylabel: str = "",
) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TICK_C)
    ax.spines[:].set_color(SPINE_C)
    if title:  ax.set_title(title, color="white", fontsize=10, pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_C, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_C, fontsize=9)


def pca_2d(X: np.ndarray) -> np.ndarray:
    """Pure-numpy 2-component PCA.  X: (N, D) → (N, 2)"""
    Xc   = X - X.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(Xc.T))
    top2 = vecs[:, np.argsort(vals)[::-1][:2]]
    return Xc @ top2


# ─────────────────────────────────────────────────────────────────────────────
# Evidence projector (mirrors Phase 3 / Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceProjector(nn.Module):
    def __init__(self, d_in: int = 1024, d_out: int = D_PROJ) -> None:
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.proj(x))   # (T, 256)


def compute_evidence(
    waveform: np.ndarray, sr: int, device: torch.device
) -> torch.Tensor:
    """WavLM layer-24 → EvidenceProjector → E_t (T, 256) on device."""
    feat_ex = AutoFeatureExtractor.from_pretrained(WAVLM_ID)
    wavlm   = WavLMModel.from_pretrained(WAVLM_ID).to(device)   # type: ignore[arg-type]
    wavlm.eval()

    inputs = feat_ex(waveform, sampling_rate=sr,
                     return_tensors="pt", padding=True)
    with torch.no_grad():
        out = wavlm(inputs.input_values.to(device), output_hidden_states=True)

    layer24 = out.hidden_states[24].squeeze(0)    # type: ignore[index]  # (T, 1024)
    log_shape("layer24", layer24)

    proj = EvidenceProjector().to(device)
    proj.eval()
    with torch.no_grad():
        E_t = proj(layer24)                        # (T, 256)
    log_shape("E_t", E_t)
    return E_t


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — MeanPoolSlotizer
# ─────────────────────────────────────────────────────────────────────────────

class MeanPoolSlotizer(nn.Module):
    """
    Baseline slotizer: every frame in a syllable contributes equally.

        S_k = mean( E_t[start_k : end_k+1], dim=0 )

    No learnable parameters.  Output: (K, 256)
    """

    def forward(
        self,
        E_t: torch.Tensor,                      # (T, 256)
        boundaries: list[tuple[int, int]],
    ) -> torch.Tensor:
        slots = [
            E_t[s: e + 1].mean(dim=0)           # (256,)
            for s, e in boundaries
        ]
        return torch.stack(slots)               # (K, 256)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — AttentionSlotizer
# ─────────────────────────────────────────────────────────────────────────────

class AttentionSlotizer(nn.Module):
    """
    Learnable-query attention slotizer.

    A single query vector q (256,) scores each frame within the syllable.
    The softmax of those scores becomes a weighted average over the frames.

        scores_k = softmax( E_t[start:end] @ q / sqrt(256) )   # (n_frames,)
        S_k      = scores_k ⊤ E_t[start:end]                   # (256,)

    The query learns WHICH frame type matters: onset, vowel nucleus, or coda.
    Initialised from N(0, 1/sqrt(d)) so attention starts roughly uniform.

    Returns S (K, 256) and attn_weights list[np.ndarray] for inspection.
    """

    def __init__(self, d: int = D_PROJ) -> None:
        super().__init__()
        self.scale = d ** 0.5
        # Xavier-style init — keeps softmax from being too peaked at start
        self.q = nn.Parameter(torch.randn(d) / (d ** 0.5))

    def forward(
        self,
        E_t: torch.Tensor,                      # (T, 256)
        boundaries: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, list[np.ndarray]]:
        slots:   list[torch.Tensor]  = []
        weights: list[np.ndarray]    = []

        for s, e in boundaries:
            chunk  = E_t[s: e + 1]              # (n_k, 256)
            scores = chunk @ self.q / self.scale # (n_k,)
            w      = F.softmax(scores, dim=0)   # (n_k,)
            slot   = w @ chunk                  # (256,)
            slots.append(slot)
            weights.append(w.detach().cpu().numpy())

        return torch.stack(slots), weights       # (K, 256)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Compare mean vs attention
# ─────────────────────────────────────────────────────────────────────────────

def compare_pooling(
    S_mean: torch.Tensor,    # (K, 256)
    S_attn: torch.Tensor,    # (K, 256)
) -> tuple[np.ndarray, float, int]:
    """
    Compute per-slot cosine similarity between mean-pooled and attn-pooled slots.

    Returns
    -------
    cos_sims      : np.ndarray  shape (K,)  per-slot cosine similarity
    mean_cos      : float       mean across all slots
    worst_slot    : int         slot index with lowest cosine similarity
    """
    S_m_n = F.normalize(S_mean, dim=-1)   # (K, 256)
    S_a_n = F.normalize(S_attn, dim=-1)   # (K, 256)
    cos_sims = (S_m_n * S_a_n).sum(dim=-1).cpu().numpy()   # (K,)
    mean_cos  = float(cos_sims.mean())
    worst_slot = int(cos_sims.argmin())
    return cos_sims, mean_cos, worst_slot


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 1 — Full compression: E_t → S_mean → S_attn
# ─────────────────────────────────────────────────────────────────────────────

def viz5_1_compression(
    E_t: np.ndarray,
    S_mean: np.ndarray,
    S_attn: np.ndarray,
    boundaries: list[tuple[int, int]],
    duration_s: float,
) -> None:
    """
    Three stacked heatmaps on a shared colourscale.
    Boundary lines on the top (E_t) panel show which frames collapse into each slot.
    """
    T, D = E_t.shape
    K    = S_mean.shape[0]

    # Z-score each matrix globally for a shared colourscale
    def znorm(X: np.ndarray) -> np.ndarray:
        return (X - X.mean()) / (X.std() + 1e-8)

    E_z = znorm(E_t)
    Sm_z = znorm(S_mean)
    Sa_z = znorm(S_attn)

    vmin, vmax = -2.0, 2.0

    fig, axes = plt.subplots(
        3, 1, figsize=(18, 12), facecolor=DARK_BG,
        gridspec_kw={"hspace": 0.45, "height_ratios": [5, 2, 2]},
    )
    fig.suptitle(
        f"Viz 5-1 — Slotizer Compression: ({T}, {D}) → ({K}, {D})",
        color="white", fontsize=13, fontweight="bold", y=1.0,
    )

    # ── Top: E_t (329, 256) with boundary overlays ───────────────────────────
    ax0 = axes[0]
    ax0.imshow(E_z.T, aspect="auto", origin="lower",
               extent=(0.0, float(T), 0.0, float(D)),
               cmap="RdBu_r", vmin=vmin, vmax=vmax,
               interpolation="nearest")

    for s, e in boundaries:
        ax0.axvline(s, color="#ffffff", lw=0.5, alpha=0.45, linestyle="--")
    ax0.axvline(boundaries[-1][1] + 1, color="#ffffff",
                lw=0.5, alpha=0.45, linestyle="--")

    # Slot index labels at top of panel
    for k, (s, e) in enumerate(boundaries):
        mid = (s + e) / 2
        ax0.text(mid, D - 8, str(k), color="#cccccc", fontsize=5.5,
                 ha="center", va="top")

    _style_ax(ax0,
              title=f"E_t  raw evidence  ({T} frames, 256 dims, ~50 Hz)  "
                    "— white dashes = syllable boundaries",
              xlabel="Frame index",
              ylabel="Dim")
    ax0.set_xlim(0, T)

    # ── Middle: S_mean (34, 256) ──────────────────────────────────────────────
    ax1 = axes[1]
    im1 = ax1.imshow(Sm_z.T, aspect="auto", origin="lower",
                     extent=(0.0, float(K), 0.0, float(D)),
                     cmap="RdBu_r", vmin=vmin, vmax=vmax,
                     interpolation="nearest")
    _style_ax(ax1,
              title=f"S_mean  mean-pooled slots  ({K} slots, 256 dims, ~5 Hz)",
              xlabel="Slot index k",
              ylabel="Dim")
    ax1.set_xlim(0, K)

    # ── Bottom: S_attn (34, 256) ──────────────────────────────────────────────
    ax2 = axes[2]
    ax2.imshow(Sa_z.T, aspect="auto", origin="lower",
               extent=(0.0, float(K), 0.0, float(D)),
               cmap="RdBu_r", vmin=vmin, vmax=vmax,
               interpolation="nearest")
    _style_ax(ax2,
              title=f"S_attn  attention-pooled slots  ({K} slots, 256 dims, ~5 Hz)",
              xlabel="Slot index k",
              ylabel="Dim")
    ax2.set_xlim(0, K)

    cbar = fig.colorbar(im1, ax=axes.tolist(),   # type: ignore[arg-type]
                        pad=0.01, fraction=0.008)
    cbar.set_label("Activation (z-scored)", color=TEXT_C, fontsize=8)
    cbar.ax.tick_params(colors=TICK_C)

    out = VIZ_DIR / "viz5_1_compression.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 2 — Attention weights deep dive: 4 representative syllables
# ─────────────────────────────────────────────────────────────────────────────

def _pick_four_syllables(
    boundaries: list[tuple[int, int]],
    waveform: np.ndarray,
    sr: int,
) -> list[int]:
    """
    Return 4 syllable indices: shortest, longest, most vowel-energy, least vowel-energy.
    """
    durations = np.array([e - s + 1 for s, e in boundaries])
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr, hop_length=HOP, n_mels=80, fmin=80, fmax=7600,
    )
    # Vowel-range energy per syllable (bands 8-40 ≈ 400Hz-4kHz)
    vowel_energy = mel[8:40, :].sum(axis=0)   # (T,)
    syl_vowel = np.array([
        vowel_energy[s: e + 1].mean() for s, e in boundaries
    ])

    # Pick candidates that have at least 3 frames to be interesting
    valid = np.where(durations >= 3)[0]
    if len(valid) < 4:
        valid = np.arange(len(boundaries))

    shortest    = valid[np.argmin(durations[valid])]
    longest     = valid[np.argmax(durations[valid])]
    most_vowel  = valid[np.argmax(syl_vowel[valid])]
    least_vowel = valid[np.argmin(syl_vowel[valid])]

    # Deduplicate while preserving order
    seen: set[int] = set()
    picks: list[int] = []
    for idx in [shortest, longest, most_vowel, least_vowel]:
        if int(idx) not in seen:
            picks.append(int(idx))
            seen.add(int(idx))
        if len(picks) == 4:
            break
    # Pad with remaining valid slots if needed
    for idx in valid:
        if len(picks) == 4:
            break
        if int(idx) not in seen:
            picks.append(int(idx))
            seen.add(int(idx))

    return picks[:4]


def viz5_2_attention(
    waveform: np.ndarray,
    sr: int,
    boundaries: list[tuple[int, int]],
    attn_weights: list[np.ndarray],
    labels: list[str],          # ["shortest","longest","most-vowel","least-vowel"]
    syl_indices: list[int],
) -> None:
    """
    4 columns, 2 rows each:
    Row 0: attention weight bar chart for each syllable
    Row 1: waveform segment for that syllable
    """
    n_cols = len(syl_indices)
    fig, axes = plt.subplots(
        2, n_cols, figsize=(4.5 * n_cols, 7), facecolor=DARK_BG,
        gridspec_kw={"hspace": 0.55, "wspace": 0.35},
    )
    fig.suptitle(
        "Viz 5-2 — Attention Weights: does attention peak at the vowel nucleus?",
        color="white", fontsize=13, fontweight="bold",
    )

    palette = ["#4fc3f7", "#ff9800", "#81c784", "#e040fb"]

    for col, (k, label) in enumerate(zip(syl_indices, labels)):
        s, e = boundaries[k]
        dur_ms  = (e - s + 1) * HOP / TARGET_SR * 1000
        n_frames = e - s + 1

        # Waveform slice
        s_samp  = s * HOP
        e_samp  = min((e + 1) * HOP, len(waveform))
        wav_sl  = waveform[s_samp:e_samp]

        w = attn_weights[k]      # (n_frames,)
        c = palette[col % len(palette)]

        # Row 0: attention bars
        ax0 = axes[0, col]
        frames_x = np.arange(n_frames)
        ax0.bar(frames_x, w, color=c, alpha=0.80, edgecolor=PANEL_BG, linewidth=0.3)
        # Mark argmax (peak attention frame)
        peak = int(w.argmax())
        ax0.axvline(peak, color="white", lw=1.2, linestyle="--", alpha=0.8)
        ax0.text(peak + 0.2, w.max() * 0.95,
                 f"peak\nf{peak}", color="white", fontsize=7, va="top")
        ax0.set_xlim(-0.5, n_frames - 0.5)
        ax0.set_ylim(0, w.max() * 1.35)
        _style_ax(ax0,
                  title=f"k={k}  [{label}]\n{dur_ms:.0f} ms  /  {n_frames} frames",
                  xlabel="Frame within syllable",
                  ylabel="Attn weight" if col == 0 else "")
        # Entropy annotation
        entropy = float(-np.sum(w * np.log(w + 1e-12)))
        uniform_ent = float(np.log(n_frames)) if n_frames > 1 else 0.0
        ax0.text(0.98, 0.98,
                 f"entropy={entropy:.2f}\n(uniform={uniform_ent:.2f})",
                 transform=ax0.transAxes, color=TEXT_C, fontsize=7,
                 va="top", ha="right")

        # Row 1: waveform
        ax1 = axes[1, col]
        t_ms = np.linspace(0, dur_ms, len(wav_sl))
        ax1.plot(t_ms, wav_sl, color=c, lw=0.9)
        ax1.axhline(0, color=SPINE_C, lw=0.5)
        # Mark the peak attention frame on waveform too
        peak_ms = peak * HOP / TARGET_SR * 1000
        ax1.axvline(peak_ms, color="white", lw=1.0, linestyle="--", alpha=0.7)
        ax1.set_xlim(0, dur_ms)
        _style_ax(ax1,
                  title="Waveform",
                  xlabel="ms",
                  ylabel="Amp" if col == 0 else "")

    out = VIZ_DIR / "viz5_2_attention.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 3 — Mean vs attention disagreement per slot
# ─────────────────────────────────────────────────────────────────────────────

def viz5_3_disagreement(
    cos_sims: np.ndarray,
    boundaries: list[tuple[int, int]],
    duration_s: float,
) -> None:
    """Bar chart of per-slot cosine similarity; green/yellow/red by threshold."""
    K = len(cos_sims)
    durations_ms = np.array(
        [(e - s + 1) * HOP / TARGET_SR * 1000 for s, e in boundaries]
    )

    bar_colors = [
        "#66bb6a" if c > 0.95 else
        "#ffd740" if c > 0.80 else
        "#ff7043"
        for c in cos_sims
    ]

    fig, axes = plt.subplots(
        2, 1, figsize=(16, 7), facecolor=DARK_BG,
        gridspec_kw={"hspace": 0.55, "height_ratios": [3, 2]},
    )
    fig.suptitle(
        "Viz 5-3 — Mean vs Attention Pooling: per-slot cosine similarity",
        color="white", fontsize=13, fontweight="bold",
    )

    # ── Top: cosine similarity bars ───────────────────────────────────────────
    ax0 = axes[0]
    x = np.arange(K)
    ax0.bar(x, cos_sims, color=bar_colors, edgecolor=PANEL_BG, linewidth=0.4)
    ax0.axhline(0.95, color="#66bb6a", lw=1.0, linestyle="--", alpha=0.6,
                label="0.95 (very similar)")
    ax0.axhline(0.80, color="#ffd740", lw=1.0, linestyle="--", alpha=0.6,
                label="0.80 (moderate diff)")
    ax0.axhline(cos_sims.mean(), color="white", lw=1.0, linestyle=":",
                alpha=0.7, label=f"mean = {cos_sims.mean():.3f}")
    ax0.set_xlim(-0.5, K - 0.5)
    ax0.set_ylim(max(0.0, cos_sims.min() - 0.05), 1.02)
    ax0.set_xticks(x)
    ax0.set_xticklabels([str(i) for i in x], fontsize=7, color=TEXT_C)
    _style_ax(ax0,
              title=f"Cosine(S_mean_k, S_attn_k) per slot  "
                    f"[green>0.95 | yellow 0.80–0.95 | red<0.80]",
              xlabel="Slot index k",
              ylabel="Cosine similarity")
    ax0.legend(fontsize=8, framealpha=0.3, facecolor=PANEL_BG,
               labelcolor="white", edgecolor=SPINE_C)

    # Mark worst slot
    worst = int(cos_sims.argmin())
    ax0.annotate(
        f"k={worst}\ncos={cos_sims[worst]:.3f}",
        xy=(worst, cos_sims[worst]),
        xytext=(worst + 1.5, cos_sims[worst] - 0.03),
        color="#ff7043", fontsize=8, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#ff7043", lw=1.0),
    )

    # ── Bottom: slot durations — longer slots tend to disagree more ───────────
    ax1 = axes[1]
    scatter = ax1.scatter(x, durations_ms, c=cos_sims,
                          cmap="RdYlGn", vmin=0.80, vmax=1.0,
                          s=50, zorder=3)
    ax1.plot(x, durations_ms, color="#444444", lw=0.8, zorder=2)
    plt.colorbar(scatter, ax=ax1, pad=0.01,
                 label="Cosine sim").ax.tick_params(colors=TICK_C)
    _style_ax(ax1,
              title="Slot duration (ms)  —  longer syllables show more disagreement",
              xlabel="Slot index k",
              ylabel="Duration (ms)")
    ax1.set_xlim(-0.5, K - 0.5)

    out = VIZ_DIR / "viz5_3_disagreement.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 4 — Slot trajectory in PCA space
# ─────────────────────────────────────────────────────────────────────────────

def viz5_4_trajectory(S_mean: np.ndarray) -> None:
    """
    PCA (K, 256) → (K, 2), draw as a labelled arrow-connected trajectory.
    Color encodes temporal order: blue=early, red=late.
    """
    K   = S_mean.shape[0]
    Z   = pca_2d(S_mean)      # (K, 2)

    # Colour ramp: early → blue, late → red
    cmap   = plt.cm.coolwarm   # type: ignore[attr-defined]
    colors = [cmap(i / (K - 1)) for i in range(K)]

    fig, ax = plt.subplots(figsize=(10, 8), facecolor=DARK_BG)
    fig.suptitle(
        "Viz 5-4 — Slot Trajectory in PCA Space  (S_mean, K=34 slots)",
        color="white", fontsize=13, fontweight="bold",
    )

    # Arrows between consecutive slots
    for i in range(K - 1):
        dx = Z[i + 1, 0] - Z[i, 0]
        dy = Z[i + 1, 1] - Z[i, 1]
        ax.annotate(
            "",
            xy=(Z[i + 1, 0], Z[i + 1, 1]),
            xytext=(Z[i, 0], Z[i, 1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=(*colors[i][:3], 0.55),
                lw=1.0,
                mutation_scale=8,
            ),
        )

    # Scatter dots
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=np.arange(K), cmap="coolwarm",
                    s=60, zorder=5, edgecolors="#333333", linewidths=0.5)
    plt.colorbar(sc, ax=ax, pad=0.01,
                 label="Slot index k (blue=early, red=late)").ax.tick_params(
                     colors=TICK_C)

    # Label every 5th slot
    for i in range(0, K, 5):
        ax.text(Z[i, 0] + 0.02, Z[i, 1] + 0.02,
                f"k={i}", color="white", fontsize=8, fontweight="bold")

    # Also label first and last
    ax.text(Z[0, 0] + 0.02, Z[0, 1] - 0.05,
            "start", color="#4fc3f7", fontsize=8)
    ax.text(Z[-1, 0] + 0.02, Z[-1, 1] - 0.05,
            "end", color="#ff7043", fontsize=8)

    _style_ax(ax, title="", xlabel="PC 1", ylabel="PC 2")
    ax.text(
        0.02, 0.03,
        "Each point = one syllable slot S_k.\n"
        "Arrows show temporal order.\n"
        "A coherent path (not random scatter) means\n"
        "the GRU can predict the next slot from the current one.",
        transform=ax.transAxes,
        color="#888888", fontsize=8, va="bottom",
    )

    out = VIZ_DIR / "viz5_4_trajectory.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo / __main__
# ─────────────────────────────────────────────────────────────────────────────

def run_demo() -> None:
    print("\n" + "█" * 60)
    print("  PHASE 5 — SLOTIZER")
    print("  Speech World Model: HuPER Implementation")
    print("█" * 60)

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).parent.parent
    audio_path   = project_root / "data" / "samples" / "librispeech_sample.wav"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ── Step 1: Load inputs ───────────────────────────────────────────────────
    print("\n--- Step 1: Load inputs ---")
    waveform, _sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    sr: int = int(_sr)
    duration_s = len(waveform) / sr

    with open(DATA_DIR / "syllable_boundaries.pkl", "rb") as fh:
        bdata = pickle.load(fh)
    boundaries: list[tuple[int, int]] = bdata["boundaries"]
    K = len(boundaries)
    T = len(waveform) // HOP
    print(f"  Audio    : {duration_s:.2f} s")
    print(f"  T frames : {T}  @ 50Hz")
    print(f"  K slots  : {K}  @ {K/duration_s:.1f} Hz")
    print(f"  Ratio    : {T}/{K} = {T/K:.1f}×  (frames per slot avg)")

    print("\n  Computing E_t (WavLM + EvidenceProjector) …")
    E_t_gpu = compute_evidence(waveform, sr, device)    # (T, 256) on device
    E_t     = E_t_gpu.cpu().numpy()                     # numpy for viz
    log_shape("E_t (numpy)", torch.from_numpy(E_t))

    # ── Step 2: MeanPoolSlotizer ──────────────────────────────────────────────
    print("\n--- Step 2: MeanPoolSlotizer ---")
    mean_slotizer = MeanPoolSlotizer()
    with torch.no_grad():
        S_mean_t = mean_slotizer(E_t_gpu.cpu(), boundaries)   # (K, 256)
    log_shape("S_mean", S_mean_t)
    S_mean = S_mean_t.numpy()

    print("\n  First 3 slot vector L2 norms:")
    for k in range(3):
        norm = float(S_mean_t[k].norm())
        print(f"    S_mean[{k}]  norm = {norm:.4f}")

    # ── Step 3: AttentionSlotizer ─────────────────────────────────────────────
    print("\n--- Step 3: AttentionSlotizer ---")
    attn_slotizer = AttentionSlotizer(D_PROJ)
    with torch.no_grad():
        S_attn_t, attn_weights = attn_slotizer(E_t_gpu.cpu(), boundaries)
    log_shape("S_attn", S_attn_t)
    S_attn = S_attn_t.numpy()

    print("\n  First 3 syllable attention distributions:")
    for k in range(3):
        s, e = boundaries[k]
        w = attn_weights[k]
        print(f"    Syllable k={k}  frames [{s}–{e}]  n={e-s+1}")
        print(f"      weights: [{', '.join(f'{v:.3f}' for v in w)}]")
        print(f"      peak at frame {w.argmax()} within slot  |  "
              f"entropy = {-np.sum(w * np.log(w + 1e-12)):.3f}  "
              f"(uniform = {np.log(max(1, e-s+1)):.3f})")

    # ── Step 4: Compare ───────────────────────────────────────────────────────
    print("\n--- Step 4: Mean vs attention comparison ---")
    cos_sims, mean_cos, worst_slot = compare_pooling(S_mean_t, S_attn_t)
    print(f"  Mean cosine similarity (all slots) : {mean_cos:.4f}")
    print(f"  Worst slot k={worst_slot}  "
          f"(cos = {cos_sims[worst_slot]:.4f})  "
          f"— biggest disagreement between strategies")
    print(f"  Best  slot k={int(cos_sims.argmax())}  "
          f"(cos = {cos_sims.max():.4f})")

    n_green  = (cos_sims > 0.95).sum()
    n_yellow = ((cos_sims > 0.80) & (cos_sims <= 0.95)).sum()
    n_red    = (cos_sims <= 0.80).sum()
    print(f"  green (>0.95): {n_green}  |  yellow (0.80–0.95): {n_yellow}"
          f"  |  red (<0.80): {n_red}")

    # ── Visualizations ────────────────────────────────────────────────────────
    print("\n--- Viz 5-1: Full compression heatmaps ---")
    viz5_1_compression(E_t, S_mean, S_attn, boundaries, duration_s)

    print("--- Viz 5-2: Attention weights deep dive ---")
    syl_indices = _pick_four_syllables(boundaries, waveform, sr)
    syl_labels  = ["shortest", "longest", "most-vowel", "least-vowel"]
    # Trim labels to match deduplicated picks
    syl_labels = syl_labels[:len(syl_indices)]
    viz5_2_attention(waveform, sr, boundaries, attn_weights,
                     syl_labels, syl_indices)

    print("--- Viz 5-3: Mean vs attention disagreement ---")
    viz5_3_disagreement(cos_sims, boundaries, duration_s)

    print("--- Viz 5-4: Slot trajectory in PCA space ---")
    viz5_4_trajectory(S_mean)

    # ── Step 6: Save outputs ──────────────────────────────────────────────────
    print("\n--- Step 6: Saving outputs ---")
    mean_path = DATA_DIR / "slots_mean.pkl"
    attn_path = DATA_DIR / "slots_attn.pkl"
    with open(mean_path, "wb") as fh:
        pickle.dump({"S": S_mean, "boundaries": boundaries,
                     "K": K, "T": T, "duration_s": duration_s}, fh)
    with open(attn_path, "wb") as fh:
        pickle.dump({"S": S_attn, "boundaries": boundaries,
                     "K": K, "T": T, "duration_s": duration_s,
                     "attn_weights": attn_weights}, fh)
    print(f"  slots_mean.pkl → {mean_path}")
    print(f"  slots_attn.pkl → {attn_path}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'─' * 40}")
    print(f"  Slotizer Summary")
    print(f"  ├── Input  : E_t ({T}, {D_PROJ}) @ 50Hz")
    print(f"  ├── Output : S   ({K}, {D_PROJ}) @ {K/duration_s:.1f}Hz")
    print(f"  ├── Compression        : {T/K:.1f}×")
    print(f"  ├── Mean vs attn cosine: {mean_cos:.3f}")
    print(f"  └── Max disagreement   : k={worst_slot}  "
          f"(cos={cos_sims[worst_slot]:.3f})")
    print(f"{'─' * 40}\n")

    print("  Visualizations saved to:")
    for f in sorted(VIZ_DIR.iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    run_demo()
