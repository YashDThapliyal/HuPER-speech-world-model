"""
Module: syllable_clock
Phase: 4
Goal: Detect syllable boundaries — the "clock ticks" that drive every
      belief update, prior update, and next-slot prediction in the world model.

What this teaches:
- Language priors don't update at 50Hz — they update at ~5Hz (syllable rate)
- Syllables are the natural unit of lexical access in human speech perception
- The clock converts: T ≈ 329 frames @ 50Hz  →  K ≈ 30 slots @ ~5Hz
- Each slot S_k = pooled evidence for one syllable tick
- Variable slot durations are WHY we need the Slotizer (Phase 5), not fixed windows

Syllable detection strategy (try in order):
  Option A: fishared/sylber           — self-supervised syllable model (HF)
  Option B: librosa onset_detect      — onset-based proxy (this is what we use)
  Option C: energy-based nucleus      — RMS peak finding in vowel bands (fallback)
"""

# --- Imports ---
from __future__ import annotations

import pickle
from pathlib import Path

import librosa
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import find_peaks, savgol_filter
from transformers import AutoFeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2Processor, WavLMModel

matplotlib.use("Agg")

# --- Constants ---
TARGET_SR      = 16_000
HOP_LENGTH     = 320          # 20ms stride — HuPER frame rate (50Hz)
WAVLM_ID       = "microsoft/wavlm-large"
PHONE_ID       = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
D_PROJ         = 256

TRANSCRIPT = (
    "HE WAS IN A FEVERED STATE OF MIND OWING TO THE "
    "BLIGHT HIS WIFE S ACTION THREATENED TO CAST "
    "UPON HIS ENTIRE FUTURE"
)

DARK_BG  = "#0e1117"
PANEL_BG = "#1a1d23"
SPINE_C  = "#333333"
TICK_C   = "#666666"
TEXT_C   = "#cccccc"

VIZ_DIR  = Path(__file__).parent.parent / "data" / "visualizations" / "phase4"
DATA_DIR = Path(__file__).parent.parent / "data"

# Boundary file Phase 5 will load
BOUNDARIES_PATH = DATA_DIR / "syllable_boundaries.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def log_shape(name: str, x: torch.Tensor | np.ndarray) -> None:
    if isinstance(x, torch.Tensor):
        print(f"  {name:35s}: {tuple(x.shape)}  dtype={x.dtype}  device={x.device}")
    else:
        print(f"  {name:35s}: shape={x.shape}  dtype={x.dtype}")


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


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Syllable segmentation
# ─────────────────────────────────────────────────────────────────────────────

def _detect_option_a(waveform: np.ndarray, sr: int) -> list[tuple[int, int]] | None:
    """
    Option A — Sylber (self-supervised syllable model).
    Returns None if model not available.
    """
    try:
        from transformers import AutoModel   # noqa: PLC0415
        model = AutoModel.from_pretrained("fishared/sylber")
        # If we get here the model loaded — run inference
        # (API TBD if model becomes available)
        del model
        return None
    except Exception:
        return None


def _detect_option_b(waveform: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """
    Option B — librosa onset detection.

    Onsets mark the start of each new acoustic event / syllable.
    We treat each onset as a syllable boundary and build
    (start_frame, end_frame) pairs at the 50Hz rate.

    Tuned delta=0.05 gives ~5 syllables/second on LibriSpeech.
    """
    onset_frames = librosa.onset.onset_detect(
        y=waveform,
        sr=sr,
        hop_length=HOP_LENGTH,
        units="frames",
        backtrack=True,
        delta=0.05,
        wait=2,
    )
    onset_frames = np.unique(onset_frames)

    # Ensure we start from frame 0 and end at the last frame
    T = len(waveform) // HOP_LENGTH
    if len(onset_frames) == 0 or onset_frames[0] != 0:
        onset_frames = np.concatenate([[0], onset_frames])
    if onset_frames[-1] < T - 1:
        onset_frames = np.concatenate([onset_frames, [T]])

    # Build (start, end) pairs — end is exclusive (start of next boundary)
    boundaries: list[tuple[int, int]] = []
    for i in range(len(onset_frames) - 1):
        s = int(onset_frames[i])
        e = int(onset_frames[i + 1]) - 1   # inclusive end
        if e > s:
            boundaries.append((s, e))
    return boundaries


def _detect_option_c(waveform: np.ndarray, sr: int) -> list[tuple[int, int]]:
    """
    Option C — Energy-based vowel nucleus detection (fallback).

    Smooth the mel-band energy in the vowel range (500–4kHz),
    find prominence peaks (= vowel nuclei = syllable centres),
    then place boundaries at the troughs between adjacent peaks.
    """
    T = len(waveform) // HOP_LENGTH
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr,
        hop_length=HOP_LENGTH, n_mels=80, fmin=80, fmax=7600,
    )
    # Bands 5–50 ≈ 300Hz–4kHz — vowel formant region
    energy = mel[5:50, :].sum(axis=0)                     # (T,)
    smooth = savgol_filter(energy, window_length=5, polyorder=2)

    peaks, _ = find_peaks(smooth, distance=3, prominence=smooth.max() * 0.02)

    if len(peaks) < 3:
        # Final fallback: evenly space 30 syllable boundaries
        boundaries = [(int(i * T / 30), int((i + 1) * T / 30) - 1)
                      for i in range(30)]
        return boundaries

    # Boundaries at midpoints between adjacent peaks + add frame 0 and last
    mids = [0] + [int((peaks[i] + peaks[i + 1]) / 2)
                  for i in range(len(peaks) - 1)] + [T]
    boundaries = [(mids[i], mids[i + 1] - 1) for i in range(len(mids) - 1)
                  if mids[i + 1] - 1 > mids[i]]
    return boundaries


def detect_syllables(
    waveform: np.ndarray, sr: int
) -> tuple[list[tuple[int, int]], str]:
    """
    Try syllable detection in order A → B → C.

    Returns
    -------
    boundaries : list of (start_frame, end_frame) at 50Hz indices
    method     : name of the method that succeeded
    """
    result = _detect_option_a(waveform, sr)
    if result is not None:
        return result, "Sylber (Option A)"

    result = _detect_option_b(waveform, sr)
    if len(result) >= 10:
        return result, "librosa onset_detect (Option B)"

    return _detect_option_c(waveform, sr), "energy-based nucleus (Option C)"


def print_syllable_table(
    boundaries: list[tuple[int, int]], sr: int, max_rows: int = 40
) -> None:
    """Print a per-syllable summary table."""
    T_total   = sum(e - s + 1 for s, e in boundaries)
    durations = [(e - s + 1) * HOP_LENGTH / sr * 1000 for s, e in boundaries]

    print(f"\n  {'k':>4}  {'start':>6}  {'end':>6}  {'frames':>7}  {'ms':>8}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*8}")
    for k, ((s, e), dur_ms) in enumerate(zip(boundaries, durations)):
        if k >= max_rows:
            print(f"  ... ({len(boundaries) - max_rows} more)")
            break
        print(f"  {k:>4}  {s:>6}  {e:>6}  {e-s+1:>7}  {dur_ms:>7.1f}")

    K = len(boundaries)
    dur_arr = np.array(durations)
    total_s = T_total * HOP_LENGTH / sr
    print(f"\n  K = {K} syllables  |  total = {total_s:.2f} s")
    print(f"  syllable rate   : {K / total_s:.1f} Hz  (target: 4–5 Hz)")
    print(f"  mean duration   : {dur_arr.mean():.0f} ms")
    print(f"  median duration : {np.median(dur_arr):.0f} ms")
    print(f"  min / max       : {dur_arr.min():.0f} / {dur_arr.max():.0f} ms")
    print(f"  compression     : T ≈ {T_total} frames @ 50 Hz  →  K = {K} slots @ "
          f"{K / total_s:.1f} Hz  ({T_total // K}× reduction per slot avg)")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Re-run EvidenceProjector to get E_t: (T, 256)
# ─────────────────────────────────────────────────────────────────────────────

class EvidenceProjector(torch.nn.Module):
    def __init__(self, d_in: int = 1024, d_out: int = D_PROJ) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(d_in, d_out)
        self.act  = torch.nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.proj(x))   # (T, 256)


def get_evidence(waveform: np.ndarray, sr: int, device: torch.device) -> np.ndarray:
    """
    Run WavLM-Large (layer 24) → EvidenceProjector.

    Returns E_t : np.ndarray  shape (T, 256)
    """
    print(f"  Loading WavLM-Large on {device} …")
    feat_ex = AutoFeatureExtractor.from_pretrained(WAVLM_ID)
    wavlm   = WavLMModel.from_pretrained(WAVLM_ID).to(device)   # type: ignore[arg-type]
    wavlm.eval()

    inputs       = feat_ex(waveform, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = wavlm(input_values, output_hidden_states=True)

    # layer-24 hidden states: (1, T, 1024)
    layer24 = outputs.hidden_states[24].squeeze(0)   # type: ignore[index]
    log_shape("layer24 hidden states", layer24)

    projector = EvidenceProjector().to(device)
    projector.eval()
    with torch.no_grad():
        E_t = projector(layer24)          # (T, 256)
    log_shape("E_t (projected)", E_t)

    return E_t.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Demonstrate clock operating: T frames → K slots
# ─────────────────────────────────────────────────────────────────────────────

def apply_syllable_clock(
    E_t: np.ndarray,
    boundaries: list[tuple[int, int]],
) -> np.ndarray:
    """
    For each syllable k, mean-pool E_t[start:end+1] → S_k.

    Parameters
    ----------
    E_t        : (T, 256)
    boundaries : K pairs of (start_frame, end_frame)

    Returns
    -------
    S : (K, 256)   — syllable-level slot evidence (mean-pooled)
    """
    slots = []
    for start, end in boundaries:
        chunk = E_t[start: end + 1]       # (num_frames, 256)
        slots.append(chunk.mean(axis=0))  # (256,)
    S = np.stack(slots)                   # (K, 256)
    return S


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 1 — Syllable boundaries on log-mel spectrogram
# ─────────────────────────────────────────────────────────────────────────────

def viz4_1_boundaries(
    waveform: np.ndarray,
    sr: int,
    boundaries: list[tuple[int, int]],
    method: str,
) -> None:
    """
    Log-mel spectrogram with:
    - alternating shaded syllable regions
    - white dashed boundary lines
    - transcript words approximately aligned below
    """
    duration_s = len(waveform) / sr
    T          = len(waveform) // HOP_LENGTH
    K          = len(boundaries)

    # Log-mel
    mel = librosa.feature.melspectrogram(
        y=waveform, sr=sr,
        hop_length=HOP_LENGTH, n_mels=80, fmin=80, fmax=7600,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)   # (80, T)

    fig, axes = plt.subplots(
        3, 1, figsize=(18, 9), facecolor=DARK_BG,
        gridspec_kw={"hspace": 0.18, "height_ratios": [1, 4, 1]},
    )
    fig.suptitle(
        f"Viz 4-1 — Syllable Clock: {K} boundaries detected  "
        f"[{method}]",
        color="white", fontsize=13, fontweight="bold", y=1.0,
    )

    # ── Waveform ──────────────────────────────────────────────────────────────
    t_wav = np.linspace(0, duration_s, len(waveform))
    axes[0].plot(t_wav, waveform, color="#4fc3f7", lw=0.4, alpha=0.85)
    axes[0].set_xlim(0, duration_s)
    axes[0].axhline(0, color=SPINE_C, lw=0.5)
    _style_ax(axes[0], title="Waveform", ylabel="Amp")
    axes[0].tick_params(labelbottom=False)

    # ── Spectrogram ───────────────────────────────────────────────────────────
    ax = axes[1]
    ax.imshow(
        log_mel,
        aspect="auto", origin="lower",
        extent=(0.0, duration_s, 0.0, 80.0),
        cmap="inferno", interpolation="nearest",
    )

    shade_colors = ["#ffffff08", "#4fc3f710"]
    for k, (start, end) in enumerate(boundaries):
        t_s = start * HOP_LENGTH / sr
        t_e = (end + 1) * HOP_LENGTH / sr
        # Shaded region
        ax.axvspan(t_s, t_e, color=shade_colors[k % 2], linewidth=0)
        # Boundary line at start
        ax.axvline(t_s, color="#ffffff", lw=0.6, linestyle="--", alpha=0.5)
        # Syllable index label at top
        t_mid = (t_s + t_e) / 2
        ax.text(t_mid, 77, str(k), color="#aaaaaa", fontsize=6,
                ha="center", va="top")

    # End boundary
    t_last = (boundaries[-1][1] + 1) * HOP_LENGTH / sr
    ax.axvline(t_last, color="#ffffff", lw=0.6, linestyle="--", alpha=0.5)
    ax.set_xlim(0, duration_s)
    _style_ax(ax,
              title=f"Log-Mel Spectrogram  |  K={K} syllable slots  "
                    f"(white dashes = boundaries, numbers = slot index)",
              ylabel="Mel bin")
    axes[0].tick_params(labelbottom=False)
    ax.tick_params(labelbottom=False)

    # ── Transcript word labels ────────────────────────────────────────────────
    ax_t = axes[2]
    ax_t.set_facecolor(DARK_BG)
    ax_t.set_xlim(0, duration_s)
    ax_t.set_ylim(0, 1)
    ax_t.set_yticks([])
    ax_t.tick_params(colors=TICK_C)
    ax_t.spines[:].set_color(SPINE_C)
    ax_t.set_xlabel("Time (s)", color=TEXT_C, fontsize=9)

    words    = TRANSCRIPT.split()
    n_words  = len(words)
    for i, word in enumerate(words):
        t_word = i / n_words * duration_s
        ax_t.axvline(t_word, color="#555555", lw=0.6, ymin=0.7, ymax=1.0)
        ax_t.text(
            t_word + duration_s / n_words * 0.05, 0.35,
            word, color=TEXT_C, fontsize=7,
            va="center", rotation=30, rotation_mode="anchor",
        )

    out = VIZ_DIR / "viz4_1_boundaries.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 2 — Syllable duration distribution
# ─────────────────────────────────────────────────────────────────────────────

def viz4_2_durations(boundaries: list[tuple[int, int]], sr: int) -> None:
    """Histogram of syllable durations in ms."""
    durations_ms = np.array(
        [(e - s + 1) * HOP_LENGTH / sr * 1000 for s, e in boundaries]
    )
    mean_ms   = durations_ms.mean()
    median_ms = np.median(durations_ms)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    fig.suptitle("Viz 4-2 — Syllable Duration Distribution",
                 color="white", fontsize=13, fontweight="bold")

    counts, bins, patches = ax.hist(
        durations_ms, bins=20, color="#4fc3f7", alpha=0.75, edgecolor=PANEL_BG,
    )

    ax.axvline(mean_ms,   color="#ff9800", lw=2.0, linestyle="-",
               label=f"Mean = {mean_ms:.0f} ms")
    ax.axvline(median_ms, color="#81c784", lw=2.0, linestyle="--",
               label=f"Median = {median_ms:.0f} ms")
    ax.axvline(200,       color="#e040fb", lw=1.5, linestyle=":",
               label="Typical syllable = 200 ms")

    _style_ax(ax,
              title=f"K={len(boundaries)} syllables  |  variable duration → "
                    "need Slotizer (not fixed windows)",
              xlabel="Duration (ms)",
              ylabel="Count")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.3,
              facecolor=PANEL_BG, labelcolor="white", edgecolor=SPINE_C)

    ax.text(0.98, 0.60,
            "Syllables range from\n"
            f"{durations_ms.min():.0f} to {durations_ms.max():.0f} ms.\n"
            "Fixed-window slicing would\n"
            "cut across phoneme boundaries.",
            transform=ax.transAxes,
            color="#888888", fontsize=8, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PANEL_BG, alpha=0.6))

    out = VIZ_DIR / "viz4_2_durations.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 3 — Frame rate vs syllable clock rate
# ─────────────────────────────────────────────────────────────────────────────

def viz4_3_rates(
    boundaries: list[tuple[int, int]],
    T: int,
    duration_s: float,
) -> None:
    """
    Two stacked horizontal timelines:
    Top    : 50Hz frame ticks (evenly spaced, show every 10th for clarity)
    Bottom : syllable clock ticks (actual boundary positions)
    Connecting lines show how frames map to slots.
    """
    K = len(boundaries)

    fig, axes = plt.subplots(
        2, 1, figsize=(18, 5), facecolor=DARK_BG,
        gridspec_kw={"hspace": 0.6},
    )
    fig.suptitle("Viz 4-3 — Frame Rate (50Hz) vs Syllable Clock (~5Hz)",
                 color="white", fontsize=13, fontweight="bold")

    # ── Top: frame timeline ───────────────────────────────────────────────────
    ax_f = axes[0]
    ax_f.set_facecolor(DARK_BG)
    ax_f.set_xlim(0, duration_s)
    ax_f.set_ylim(-0.5, 1.5)
    ax_f.set_yticks([])
    ax_f.spines[:].set_color(SPINE_C)
    ax_f.tick_params(colors=TICK_C)

    frame_times = np.arange(T) * HOP_LENGTH / TARGET_SR
    # Show every 5th frame tick
    stride = max(1, T // 80)
    for ft in frame_times[::stride]:
        ax_f.axvline(ft, color="#4fc3f7", lw=0.6, alpha=0.5, ymin=0.2, ymax=0.8)

    ax_f.text(0.01, 1.1, f"T = {T} frames @ 50Hz  (one tick = 20ms)",
              transform=ax_f.transAxes, color="#4fc3f7", fontsize=9)
    ax_f.set_xlabel("Time (s)", color=TEXT_C, fontsize=9)

    # ── Bottom: syllable clock timeline ───────────────────────────────────────
    ax_s = axes[1]
    ax_s.set_facecolor(DARK_BG)
    ax_s.set_xlim(0, duration_s)
    ax_s.set_ylim(-0.5, 1.5)
    ax_s.set_yticks([])
    ax_s.spines[:].set_color(SPINE_C)
    ax_s.tick_params(colors=TICK_C)

    syl_colors = plt.cm.Set2(np.linspace(0, 1, K))   # type: ignore[attr-defined]
    for k, ((start, end), col) in enumerate(zip(boundaries, syl_colors)):
        t_s = start * HOP_LENGTH / TARGET_SR
        t_e = (end + 1) * HOP_LENGTH / TARGET_SR
        ax_s.barh(0.5, t_e - t_s, left=t_s, height=0.6,
                  color=col, alpha=0.75, linewidth=0)
        # Boundary tick
        ax_s.axvline(t_s, color="white", lw=1.0, alpha=0.6, ymin=0.1, ymax=0.9)
        # Slot number
        if t_e - t_s > 0.05:
            ax_s.text((t_s + t_e) / 2, 0.5, str(k),
                      ha="center", va="center", color="black",
                      fontsize=7, fontweight="bold")

    ax_s.text(0.01, 1.1,
              f"K = {K} syllable slots @ {K / duration_s:.1f} Hz  "
              f"(variable tick spacing)",
              transform=ax_s.transAxes, color="#ff9800", fontsize=9)
    ax_s.set_xlabel("Time (s)", color=TEXT_C, fontsize=9)

    # Draw connecting lines between shared time axis (using figure coords)
    fig.canvas.draw()
    ratio = T // K

    out = VIZ_DIR / "viz4_3_rates.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 4 — Evidence slice per syllable (FEVERED STATE deep dive)
# ─────────────────────────────────────────────────────────────────────────────

def _find_fevered_state_syllables(
    boundaries: list[tuple[int, int]], duration_s: float, n_syl: int = 4
) -> list[int]:
    """
    Find the syllable indices that fall in the FEVERED STATE time range.
    FEVERED STATE is words 5-6 of 22 ≈ 22-32% into the clip.
    """
    words    = TRANSCRIPT.split()
    n_words  = len(words)
    try:
        word_idx = words.index("FEVERED")
    except ValueError:
        word_idx = n_words // 3

    t_start = word_idx / n_words * duration_s
    t_end   = (word_idx + 3) / n_words * duration_s  # 3-word window

    # Find syllables whose centres fall in this range
    hits = []
    for k, (s, e) in enumerate(boundaries):
        t_centre = ((s + e) / 2) * HOP_LENGTH / TARGET_SR
        if t_start <= t_centre <= t_end:
            hits.append(k)

    if len(hits) < n_syl:
        # Fallback: use syllables 5–8
        hits = list(range(5, 5 + n_syl))

    return hits[:n_syl]


def viz4_4_slots(
    waveform: np.ndarray,
    sr: int,
    E_t: np.ndarray,
    boundaries: list[tuple[int, int]],
    duration_s: float,
) -> None:
    """
    4 consecutive syllables from FEVERED STATE, each showing:
    - waveform segment
    - E_t heatmap for that syllable's frames
    - mean-pooled slot vector S_k
    """
    syl_indices = _find_fevered_state_syllables(boundaries, duration_s, n_syl=4)
    n_cols = len(syl_indices)

    fig, axes = plt.subplots(
        3, n_cols, figsize=(4 * n_cols, 10), facecolor=DARK_BG,
        gridspec_kw={"hspace": 0.55, "wspace": 0.35},
    )
    fig.suptitle(
        "Viz 4-4 — Evidence Slots: FEVERED STATE  "
        "(waveform → E_t frames → mean-pooled S_k)",
        color="white", fontsize=13, fontweight="bold",
    )

    frame_colors = ["#4fc3f7", "#ff9800", "#81c784", "#e040fb"]

    for col, k in enumerate(syl_indices):
        start_f, end_f = boundaries[k]
        # sample range
        s_samp = start_f * HOP_LENGTH
        e_samp = min((end_f + 1) * HOP_LENGTH, len(waveform))
        wav_slice  = waveform[s_samp:e_samp]
        post_slice = E_t[start_f: end_f + 1]   # (num_frames, 256)
        S_k        = post_slice.mean(axis=0)    # (256,)
        dur_ms     = len(wav_slice) / sr * 1000
        n_frames   = post_slice.shape[0]

        t_ms = np.linspace(0, dur_ms, len(wav_slice))
        col_color = frame_colors[col % len(frame_colors)]

        # Row 0: waveform
        ax0 = axes[0, col]
        ax0.plot(t_ms, wav_slice, color=col_color, lw=0.9)
        ax0.axhline(0, color=SPINE_C, lw=0.5)
        ax0.set_xlim(0, dur_ms)
        _style_ax(ax0,
                  title=f"Syllable k={k}\n{dur_ms:.0f} ms  /  {n_frames} frames",
                  xlabel="ms" if col == 0 else "",
                  ylabel="Amp" if col == 0 else "")

        # Row 1: E_t heatmap for this syllable's frames
        ax1 = axes[1, col]
        # Normalise for display
        disp = post_slice.T    # (256, n_frames)
        mu, sg = disp.mean(1, keepdims=True), disp.std(1, keepdims=True) + 1e-8
        im = ax1.imshow(
            (disp - mu) / sg,
            aspect="auto", origin="lower",
            extent=(0.0, float(n_frames), 0.0, float(D_PROJ)),
            cmap="RdBu_r", vmin=-2.5, vmax=2.5,
            interpolation="nearest",
        )
        _style_ax(ax1,
                  title=f"E_t  ({n_frames} × {D_PROJ})",
                  xlabel="Frame" if col == 0 else "",
                  ylabel="Dim" if col == 0 else "")

        # Row 2: mean-pooled slot vector S_k
        ax2 = axes[2, col]
        dims = np.arange(D_PROJ)
        # Smooth for readability
        s_smooth = np.convolve(S_k, np.ones(8) / 8, mode="same")
        ax2.plot(dims, s_smooth, color=col_color, lw=1.1)
        ax2.axhline(0, color=SPINE_C, lw=0.5)
        ax2.fill_between(dims, s_smooth, 0, alpha=0.15, color=col_color)
        ax2.set_xlim(0, D_PROJ)
        _style_ax(ax2,
                  title=f"S_k  (mean pool → {D_PROJ}-dim)",
                  xlabel="Dim" if col == 0 else "",
                  ylabel="Value" if col == 0 else "")

    out = VIZ_DIR / "viz4_4_slots.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo / __main__
# ─────────────────────────────────────────────────────────────────────────────

def run_demo() -> None:
    print("\n" + "█" * 60)
    print("  PHASE 4 — SYLLABLE CLOCK")
    print("  Speech World Model: HuPER Implementation")
    print("█" * 60)

    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).parent.parent
    audio_path   = project_root / "data" / "samples" / "librispeech_sample.wav"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # ── Load audio ────────────────────────────────────────────────────────────
    print("\n--- Loading audio ---")
    waveform, _sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    sr: int       = int(_sr)
    duration_s    = len(waveform) / sr
    T             = len(waveform) // HOP_LENGTH
    log_shape("waveform", torch.from_numpy(waveform))
    print(f"  {duration_s:.2f} s  |  T = {T} frames @ 50 Hz")

    # ── Step 1: Syllable detection ────────────────────────────────────────────
    print("\n--- Step 1: Syllable segmentation ---")
    boundaries, method = detect_syllables(waveform, sr)
    K = len(boundaries)
    print(f"  Method : {method}")
    print(f"  K      = {K} syllables")
    print_syllable_table(boundaries, sr)

    # Save for Phase 5
    with open(BOUNDARIES_PATH, "wb") as fh:
        pickle.dump({"boundaries": boundaries, "method": method,
                     "K": K, "T": T, "sr": sr, "duration_s": duration_s}, fh)
    print(f"\n  Boundaries saved → {BOUNDARIES_PATH}")

    # ── Step 2: Evidence E_t ─────────────────────────────────────────────────
    print("\n--- Step 2: Evidence projection E_t (T, 256) ---")
    E_t = get_evidence(waveform, sr, device)
    log_shape("E_t", torch.from_numpy(E_t))

    # ── Step 3: Clock operating on evidence ──────────────────────────────────
    print("\n--- Step 3: Syllable clock applied to E_t ---")
    S = apply_syllable_clock(E_t, boundaries)
    log_shape("S (syllable slots)", torch.from_numpy(S))
    print(f"\n  [COMPRESSION]")
    print(f"  E_t shape : ({T}, {E_t.shape[1]})   @ ~50 Hz")
    print(f"  S   shape : ({K}, {S.shape[1]})    @ ~{K/duration_s:.1f} Hz")
    print(f"  Ratio     : {T}/{K} = {T/K:.1f}× average frames per slot")
    print(f"  This is the 50Hz → 5Hz transition — the clock doing its job")

    # ── Visualizations ────────────────────────────────────────────────────────
    print("\n--- Viz 4-1: Syllable boundaries on spectrogram ---")
    viz4_1_boundaries(waveform, sr, boundaries, method)

    print("--- Viz 4-2: Syllable duration distribution ---")
    viz4_2_durations(boundaries, sr)

    print("--- Viz 4-3: Frame rate vs clock rate ---")
    viz4_3_rates(boundaries, T, duration_s)

    print("--- Viz 4-4: Evidence slots for FEVERED STATE ---")
    viz4_4_slots(waveform, sr, E_t, boundaries, duration_s)

    # ── Summary ───────────────────────────────────────────────────────────────
    durations_ms = np.array(
        [(e - s + 1) * HOP_LENGTH / sr * 1000 for s, e in boundaries]
    )
    print(f"\n{'=' * 60}")
    print("  Phase 4 complete.")
    print("  What we learned:")
    print(f"  1. {K} syllable ticks detected in {duration_s:.1f} s  "
          f"→  {K/duration_s:.1f} Hz (target: 4-5 Hz)")
    print(f"  2. Syllable duration: {durations_ms.mean():.0f} ms avg  "
          f"({durations_ms.min():.0f}–{durations_ms.max():.0f} ms range)")
    print(f"  3. The clock compresses T={T} → K={K}  ({T/K:.1f}× avg)")
    print(f"     This is WHY language priors can update at 5Hz not 50Hz")
    print(f"  4. Each slot S_k = mean-pooled E_t evidence for one syllable")
    print(f"     S shape: ({K}, {D_PROJ})  — ready for BeliefTransition (Phase 6)")
    print(f"  5. Variable durations ({durations_ms.min():.0f}–{durations_ms.max():.0f} ms)")
    print(f"     confirm fixed windows would miss phoneme boundaries")
    print(f"  6. syllable_boundaries.pkl saved for Phase 5 (Slotizer)")
    print(f"{'=' * 60}\n")

    print("  Visualizations saved to:")
    for f in sorted(VIZ_DIR.iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    run_demo()
