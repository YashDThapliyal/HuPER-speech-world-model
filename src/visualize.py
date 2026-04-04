"""
Module: visualize
Phase: 2 (extended)
Goal: Four focused visualizations that make phones and CTC tangible.

Generates:
  viz1_phone_heatmap.png        — Full 44×T posterior heatmap with IPA y-axis
  viz2_blank_timeline.png       — Frame-by-frame blank/phone timeline + transcript
  viz3_fevered_deepdive.png     — Zoom into "FEVERED": waveform + spectrogram + posteriors
  viz4_ctc_collapse_diagram.png — First 10 frames: top-3 probs table + CTC collapse walk-through
"""

# --- Imports ---
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import librosa.display
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

matplotlib.use("Agg")

# --- Constants ---
MODEL_ID    = "vitouphy/wav2vec2-xls-r-300m-timit-phoneme"
TARGET_SR   = 16_000
BLANK_TOKEN = "[PAD]"
WORD_SEP    = "|"
EXCLUDE_TOK = {BLANK_TOKEN, WORD_SEP, "[UNK]", "<s>", "</s>", " "}

TRANSCRIPT  = "HE WAS IN A FEVERED STATE OF MIND OWING TO THE BLIGHT HIS WIFE S ACTION THREATENED TO CAST UPON HIS ENTIRE FUTURE"

# IPA labels for FEVERED's phones (as this model decodes them)
FEVERED_IPA = ["f", "ɪ", "v", "ɝ", "d"]   # f-EH-V-ER-D

VIZ_DIR = Path(__file__).parent.parent / "data" / "visualizations"

DARK_BG  = "#0e1117"
PANEL_BG = "#1a1d23"
SPINE_C  = "#333333"
TICK_C   = "#666666"
TEXT_C   = "#cccccc"


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING / INFERENCE (shared across all four plots)
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> tuple[
    np.ndarray, int, np.ndarray, list[str], str, torch.device
]:
    """
    Load audio + run phone recognizer.  Returns:
        waveform, sr, posteriors (T,44), phone_labels, decoded_str, device
    """
    project_root = Path(__file__).parent.parent
    audio_path   = project_root / "data" / "samples" / "librispeech_sample.wav"

    print("  Loading audio …")
    waveform, _sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    sr: int = int(_sr)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Loading model on {device} …")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model     = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device)  # type: ignore[arg-type]
    model.eval()

    inputs       = processor(waveform, sampling_rate=sr,
                             return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits          # (1, T, 44)

    posteriors  = F.softmax(logits[0], dim=-1).cpu().numpy()  # type: ignore[attr-defined]
    vocab       = processor.tokenizer.get_vocab()              # type: ignore[attr-defined]
    id_to_phone = {v: k for k, v in vocab.items()}
    phone_labels = [id_to_phone[i] for i in range(logits.shape[-1])]

    pred_ids    = np.argmax(posteriors, axis=-1)
    decoded_str = processor.decode(pred_ids)

    print(f"  Posteriors: {posteriors.shape}  device: {device}")
    return waveform, sr, posteriors, phone_labels, decoded_str, device


def _style_ax(ax: plt.Axes, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TICK_C)
    ax.spines[:].set_color(SPINE_C)
    if title:  ax.set_title(title, color="white", fontsize=10, pad=6)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT_C, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT_C, fontsize=9)


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 1 — Full phone posterior heatmap  (44 × T)
# ─────────────────────────────────────────────────────────────────────────────

def viz1_phone_heatmap(
    waveform: np.ndarray,
    sr: int,
    posteriors: np.ndarray,
    phone_labels: list[str],
) -> None:
    """
    Full heatmap: ALL 44 phone rows × T columns.
    X = time (s),  Y = IPA phone label,  Color = probability.
    Blank/boundary rows are kept but visually dimmed at the bottom.
    """
    duration_s = len(waveform) / sr
    n_frames   = posteriors.shape[0]

    # Separate real phones from special tokens — keep blank visible for context
    real_idx  = [i for i, p in enumerate(phone_labels) if p not in EXCLUDE_TOK]
    blank_idx = [i for i, p in enumerate(phone_labels) if p == BLANK_TOKEN]

    # Stack: real phones (sorted by mean act, descending) + blank at bottom
    real_mean = posteriors[:, real_idx].mean(axis=0)
    sort_ord  = np.argsort(real_mean)[::-1]   # most-active phones at top
    sorted_real_idx   = [real_idx[i] for i in sort_ord]
    sorted_real_names = [phone_labels[i] for i in sorted_real_idx]

    # Build display matrix: (n_real+1, T)  — append blank as last row
    mat = np.vstack([
        posteriors[:, sorted_real_idx].T,           # (n_real, T)
        posteriors[:, blank_idx].mean(axis=1)[None], # (1, T)
    ])
    y_labels = sorted_real_names + [BLANK_TOKEN]
    n_rows   = len(y_labels)

    fig, axes = plt.subplots(
        2, 1, figsize=(16, 9),
        gridspec_kw={"height_ratios": [1, 5], "hspace": 0.25},
        facecolor=DARK_BG,
    )
    fig.suptitle("Viz 1 — Phone Probability Heatmap", color="white",
                 fontsize=13, fontweight="bold", y=0.99)

    # ── Waveform ──────────────────────────────────────────────────────────────
    t_wav = np.linspace(0, duration_s, len(waveform))
    axes[0].plot(t_wav, waveform, color="#4fc3f7", lw=0.4, alpha=0.85)
    axes[0].set_xlim(0, duration_s)
    axes[0].axhline(0, color=SPINE_C, lw=0.5)
    _style_ax(axes[0],
              title=f"Waveform  ({duration_s:.2f} s  |  {n_frames} frames @ ~50 Hz)",
              ylabel="Amp")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    ax = axes[1]
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="upper",
        extent=(0.0, float(duration_s), float(n_rows) - 0.5, -0.5),
        cmap="inferno",
        vmin=0.0,
        vmax=0.65,
        interpolation="nearest",
    )
    # Horizontal separator before blank row
    ax.axhline(n_rows - 1.5, color="#555555", lw=0.8, linestyle="--")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(y_labels, fontsize=8, color=TEXT_C)
    ax.set_xlim(0, duration_s)
    _style_ax(ax,
              title=f"Phone Posteriors  ({n_rows - 1} IPA phones + blank  ×  {n_frames} frames)",
              xlabel="Time (s)",
              ylabel="Phone (IPA)")

    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.015)
    cbar.set_label("Probability", color=TEXT_C, fontsize=8)
    cbar.ax.tick_params(colors=TICK_C)

    out = VIZ_DIR / "viz1_phone_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 2 — Blank vs non-blank timeline
# ─────────────────────────────────────────────────────────────────────────────

def viz2_blank_timeline(
    waveform: np.ndarray,
    sr: int,
    posteriors: np.ndarray,
    phone_labels: list[str],
) -> None:
    """
    Horizontal bar: each frame colored green (phone) or red (blank).
    Below it: waveform + word markers from the transcript.
    """
    duration_s  = len(waveform) / sr
    n_frames    = posteriors.shape[0]
    blank_id    = phone_labels.index(BLANK_TOKEN)
    pred_ids    = np.argmax(posteriors, axis=-1)    # (T,)
    is_blank    = pred_ids == blank_id

    frame_times = np.linspace(0, duration_s, n_frames)

    # Approximate word boundaries: divide duration evenly across words
    words        = TRANSCRIPT.split()
    n_words      = len(words)
    word_starts  = [i / n_words * duration_s for i in range(n_words)]

    fig = plt.figure(figsize=(18, 5), facecolor=DARK_BG)
    fig.suptitle("Viz 2 — Blank vs Non-Blank Timeline", color="white",
                 fontsize=13, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.05,
                           height_ratios=[2, 0.6, 1.2])

    # ── Waveform ──────────────────────────────────────────────────────────────
    ax_wav = fig.add_subplot(gs[0])
    t_wav  = np.linspace(0, duration_s, len(waveform))
    ax_wav.plot(t_wav, waveform, color="#4fc3f7", lw=0.4, alpha=0.8)
    ax_wav.set_xlim(0, duration_s)
    ax_wav.axhline(0, color=SPINE_C, lw=0.5)
    # Shade blanks on waveform too
    for i, blank in enumerate(is_blank):
        if blank:
            t0 = frame_times[i] - 0.01
            t1 = frame_times[i] + 0.01
            ax_wav.axvspan(t0, t1, color="#ff7043", alpha=0.15, linewidth=0)
    _style_ax(ax_wav,
              title="Waveform  (red shading = blank frames)",
              ylabel="Amp")
    ax_wav.tick_params(labelbottom=False)

    # ── Frame timeline bar ────────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1])
    colors = np.where(is_blank, 0.0, 1.0)   # 0=blank→red palette, 1=phone→green
    # Draw as a scatter of thin rectangles
    frame_w = duration_s / n_frames * 0.95
    for i in range(n_frames):
        c = "#ff7043" if is_blank[i] else "#66bb6a"
        ax_bar.barh(0, frame_w, left=frame_times[i] - frame_w / 2,
                    height=0.8, color=c, linewidth=0)
    ax_bar.set_xlim(0, duration_s)
    ax_bar.set_ylim(-0.5, 0.5)
    ax_bar.set_yticks([])
    ax_bar.tick_params(labelbottom=False)
    _style_ax(ax_bar)

    # Legend inside bar
    blank_pct = 100 * is_blank.sum() / n_frames
    ax_bar.text(0.01, 0.55, f"■ phone ({100-blank_pct:.0f}%)",
                transform=ax_bar.transAxes, color="#66bb6a", fontsize=8, va="top")
    ax_bar.text(0.12, 0.55, f"■ blank ({blank_pct:.0f}%)",
                transform=ax_bar.transAxes, color="#ff7043", fontsize=8, va="top")

    # ── Transcript word markers ───────────────────────────────────────────────
    ax_txt = fig.add_subplot(gs[2])
    ax_txt.set_facecolor(DARK_BG)
    ax_txt.set_xlim(0, duration_s)
    ax_txt.set_ylim(0, 1)
    ax_txt.set_yticks([])
    ax_txt.tick_params(colors=TICK_C)
    ax_txt.spines[:].set_color(SPINE_C)
    ax_txt.set_xlabel("Time (s)", color=TEXT_C, fontsize=9)

    # Draw word ticks + labels
    for w, t in zip(words, word_starts):
        ax_txt.axvline(t, color="#555555", lw=0.6, ymin=0.6, ymax=1.0)
        ax_txt.text(t + (duration_s / n_words) * 0.1, 0.45, w,
                    color=TEXT_C, fontsize=7, va="center",
                    rotation=35, rotation_mode="anchor")

    # Highlight FEVERED word bar
    fevered_idx = words.index("FEVERED")
    f_start = word_starts[fevered_idx]
    f_end   = word_starts[fevered_idx + 1] if fevered_idx + 1 < n_words else duration_s
    ax_bar.axvspan(f_start, f_end, color="#ffd740", alpha=0.25, zorder=5)
    ax_wav.axvspan(f_start, f_end, color="#ffd740", alpha=0.10, zorder=5)
    ax_bar.text(
        (f_start + f_end) / 2, -0.35, "FEVERED",
        ha="center", va="center", color="#ffd740", fontsize=7, fontweight="bold",
    )

    out = VIZ_DIR / "viz2_blank_timeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 3 — "FEVERED" deep dive
# ─────────────────────────────────────────────────────────────────────────────

def _find_fevered_frames(
    posteriors: np.ndarray, phone_labels: list[str], duration_s: float
) -> tuple[int, int]:
    """
    Locate the frame range for FEVERED by scanning the argmax sequence
    for the IPA pattern  f → ɪ/i/ɛ → v → ɝ → d
    after the initial silence region.

    Returns (start_frame, end_frame) — inclusive.
    Fallbacks to a position-ratio estimate if pattern not found.
    """
    n_frames  = posteriors.shape[0]
    pred_ids  = np.argmax(posteriors, axis=-1)
    labels    = [phone_labels[i] for i in pred_ids]

    # Find 'f' occurrences after frame 40 (past "HE WAS IN A")
    f_id    = phone_labels.index("f") if "f" in phone_labels else -1
    v_id    = phone_labels.index("v") if "v" in phone_labels else -1
    er_id   = phone_labels.index("ɝ") if "ɝ" in phone_labels else -1

    candidates: list[tuple[int, int]] = []
    if f_id >= 0 and v_id >= 0:
        for i in range(40, n_frames - 20):
            if pred_ids[i] == f_id:
                # Look for v within next 20 frames, then ɝ within 15 more
                v_pos = next((j for j in range(i, min(i + 20, n_frames))
                              if pred_ids[j] == v_id), None)
                if v_pos is None:
                    continue
                er_pos = next((j for j in range(v_pos, min(v_pos + 15, n_frames))
                               if er_id >= 0 and pred_ids[j] == er_id), None)
                end = er_pos + 10 if er_pos else v_pos + 15
                candidates.append((i, min(end, n_frames - 1)))

    if candidates:
        start, end = candidates[0]
        # Pad slightly
        return max(0, start - 3), min(n_frames - 1, end + 5)

    # Fallback: FEVERED is ~5th word of 20 → ~25% into clip
    words    = TRANSCRIPT.split()
    n_words  = len(words)
    f_word   = words.index("FEVERED")
    t_start  = f_word / n_words * duration_s
    t_end    = (f_word + 1) / n_words * duration_s
    fps      = n_frames / duration_s
    return int(t_start * fps), int(t_end * fps)


def viz3_fevered_deepdive(
    waveform: np.ndarray,
    sr: int,
    posteriors: np.ndarray,
    phone_labels: list[str],
) -> None:
    """
    Three-panel deep dive into the word FEVERED:
      Panel 1: waveform slice + phone labels
      Panel 2: log-mel spectrogram slice
      Panel 3: phone posterior heatmap (real phones only) for those frames
    """
    duration_s = len(waveform) / sr
    n_frames   = posteriors.shape[0]
    hop        = int(TARGET_SR * 20 / 1000)     # 320 samples per frame

    start_f, end_f = _find_fevered_frames(posteriors, phone_labels, duration_s)
    start_s = start_f * hop
    end_s   = min(end_f * hop + hop, len(waveform))

    wav_slice  = waveform[start_s:end_s]
    post_slice = posteriors[start_f:end_f + 1]   # (K, 44)
    t_start    = start_f / n_frames * duration_s
    t_end      = (end_f + 1) / n_frames * duration_s
    duration_slice = t_end - t_start

    # Log-mel for this slice
    log_mel = librosa.feature.melspectrogram(
        y=wav_slice, sr=sr,
        n_fft=hop, hop_length=hop, win_length=hop,
        n_mels=40, fmin=80, fmax=7600,
    )
    log_mel = librosa.power_to_db(log_mel, ref=np.max)

    # Real phone indices for heatmap
    real_idx   = [i for i, p in enumerate(phone_labels) if p not in EXCLUDE_TOK]
    real_names = [phone_labels[i] for i in real_idx]
    post_real  = post_slice[:, real_idx]     # (K, n_real)

    # Sort by mean activation in this slice
    sort_ord = np.argsort(post_real.mean(axis=0))[::-1]
    post_disp = post_real[:, sort_ord].T     # (n_real, K)
    names_disp = [real_names[i] for i in sort_ord]

    # Phone annotation: label FEVERED's expected IPA phones at approximate positions
    n_slice = post_slice.shape[0]
    phone_annotations = [
        (0.0 * n_slice, "f"),
        (0.2 * n_slice, "ɪ"),
        (0.4 * n_slice, "v"),
        (0.6 * n_slice, "ɝ"),
        (0.82 * n_slice, "d"),
    ]

    fig = plt.figure(figsize=(14, 10), facecolor=DARK_BG)
    fig.suptitle(
        f'Viz 3 — "FEVERED" Deep Dive  '
        f'(frames {start_f}–{end_f}  /  {t_start:.2f}–{t_end:.2f} s)',
        color="white", fontsize=13, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

    # ── Waveform slice ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    t_wav = np.linspace(0, duration_slice, len(wav_slice)) * 1000   # ms
    ax1.plot(t_wav, wav_slice, color="#4fc3f7", lw=1.0)
    ax1.set_xlim(0, duration_slice * 1000)
    ax1.axhline(0, color=SPINE_C, lw=0.5)
    # Phone label markers
    for frac_pos, label in phone_annotations:
        x_ms = frac_pos / n_slice * duration_slice * 1000
        ax1.axvline(x_ms, color="#ffd740", lw=1.0, linestyle="--", alpha=0.8)
        ax1.text(x_ms + 1, ax1.get_ylim()[1] * 0.75 if ax1.get_ylim()[1] != 0 else 0.15,
                 f"/{label}/", color="#ffd740", fontsize=10, fontweight="bold")
    _style_ax(ax1,
              title="Waveform  (gold markers = expected phone boundaries for F-EH-V-ER-D)",
              xlabel="Time (ms)",
              ylabel="Amp")

    # ── Log-mel spectrogram ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(
        log_mel,
        aspect="auto",
        origin="lower",
        extent=(0.0, float(duration_slice), 0.0, float(log_mel.shape[0])),
        cmap="inferno",
        interpolation="nearest",
    )
    ax2.set_ylabel("Mel bin", color=TEXT_C, fontsize=9)
    for frac_pos, label in phone_annotations:
        x_s = frac_pos / n_slice * duration_slice
        ax2.axvline(x_s, color="#ffd740", lw=1.0, linestyle="--", alpha=0.8)
        ax2.text(x_s + 0.001, ax2.get_ylim()[1] * 0.88,
                 f"/{label}/", color="#ffd740", fontsize=9, fontweight="bold")
    _style_ax(ax2,
              title="Log-Mel Spectrogram",
              xlabel="Time (s)",
              ylabel="Mel freq")

    # ── Phone posterior heatmap for slice ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    im = ax3.imshow(
        post_disp,
        aspect="auto",
        origin="upper",
        extent=(0.0, float(duration_slice), float(len(names_disp)) - 0.5, -0.5),
        cmap="inferno",
        vmin=0.0,
        vmax=0.8,
        interpolation="nearest",
    )
    for frac_pos, label in phone_annotations:
        x_s = frac_pos / n_slice * duration_slice
        ax3.axvline(x_s, color="#ffd740", lw=1.2, linestyle="--", alpha=0.9)
        ax3.text(x_s + duration_slice * 0.01, -0.3,
                 f"/{label}/", color="#ffd740", fontsize=9, fontweight="bold")
    ax3.set_yticks(range(len(names_disp)))
    ax3.set_yticklabels(names_disp, fontsize=7, color=TEXT_C)
    fig.colorbar(im, ax=ax3, pad=0.01, fraction=0.015).ax.tick_params(colors=TICK_C)
    _style_ax(ax3,
              title=f"Phone Posteriors  ({n_slice} frames, sorted by activation)",
              xlabel="Time (s)",
              ylabel="Phone (IPA)")

    out = VIZ_DIR / "viz3_fevered_deepdive.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# VIZ 4 — CTC collapse diagram (first 10 frames)
# ─────────────────────────────────────────────────────────────────────────────

def viz4_ctc_collapse(
    posteriors: np.ndarray,
    phone_labels: list[str],
) -> None:
    """
    Table diagram for first 10 frames showing CTC in slow motion:
      Row 0: Frame index
      Row 1: Top-3 phones + probabilities (bar-style sparkline)
      Row 2: Argmax winner (blank or phone)
      Row 3: Accumulated CTC sequence after each frame
    """
    N = 10
    blank_id = phone_labels.index(BLANK_TOKEN)
    post10   = posteriors[:N]           # (10, 44)
    pred_ids = np.argmax(post10, axis=-1)

    # CTC accumulation state
    ctc_seq: list[str] = []
    ctc_snapshots: list[str] = []
    for pid in pred_ids:
        token = phone_labels[pid]
        if token == BLANK_TOKEN:
            pass   # blank resets repeat-suppression but produces no output
        elif not ctc_seq or ctc_seq[-1] != token:
            ctc_seq.append(token)
        ctc_snapshots.append(" ".join(ctc_seq) if ctc_seq else "—")

    # Figure — use matplotlib table + bar annotations
    fig, ax = plt.subplots(figsize=(18, 7), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.axis("off")
    fig.suptitle("Viz 4 — CTC Collapse Walk-Through  (first 10 frames)",
                 color="white", fontsize=13, fontweight="bold", y=0.98)

    cell_w = 1.0 / (N + 1)     # +1 for label column
    row_h  = 0.18
    rows   = 4
    row_tops = [0.88, 0.65, 0.38, 0.18]   # y positions for each row
    row_labels = [
        "Frame #",
        "Top-3 phones\n(prob bars)",
        "Argmax\n(winner)",
        "CTC sequence\nafter frame",
    ]

    label_col_x = 0.01

    # Row header labels
    for r, (label, y) in enumerate(zip(row_labels, row_tops)):
        ax.text(label_col_x, y, label,
                color="#aaaaaa", fontsize=9, va="top", fontweight="bold",
                transform=ax.transAxes)

    # Separator line after header column
    ax.axvline(cell_w, color=SPINE_C, lw=1.0, ymin=0.10, ymax=0.95)

    # Column headers: frame numbers
    for col in range(N):
        cx = cell_w + (col + 0.5) * cell_w   # centre x of cell
        is_blank_frame = pred_ids[col] == blank_id
        cell_bg = "#2d1a1a" if is_blank_frame else "#1a2d1a"

        # Background rect for this column
        rect = mpatches.FancyBboxPatch(
            (cell_w + col * cell_w + 0.005, 0.09),
            cell_w - 0.01, 0.86,
            boxstyle="round,pad=0.01",
            facecolor=cell_bg, edgecolor=SPINE_C, lw=0.8,
            transform=ax.transAxes, clip_on=False,
        )
        ax.add_patch(rect)

        # Row 0: frame number
        ax.text(cx, row_tops[0], str(col),
                color="white", fontsize=11, ha="center", va="top",
                fontweight="bold", transform=ax.transAxes)

        # Row 1: top-3 phones
        top3_idx  = np.argsort(post10[col])[::-1][:3]
        top3_info = [(phone_labels[i], post10[col, i]) for i in top3_idx]
        bar_y_start = row_tops[1] - 0.02
        bar_max_w   = cell_w * 0.75
        for rank, (phone, prob) in enumerate(top3_info):
            bar_y = bar_y_start - rank * 0.065
            bar_w = prob * bar_max_w
            bar_color = "#ff7043" if phone == BLANK_TOKEN else "#4fc3f7"
            bar_rect = mpatches.FancyBboxPatch(
                (cx - bar_max_w / 2, bar_y - 0.022),
                bar_w, 0.030,
                boxstyle="square,pad=0",
                facecolor=bar_color, alpha=0.7, edgecolor="none",
                transform=ax.transAxes, clip_on=False,
            )
            ax.add_patch(bar_rect)
            ax.text(cx - bar_max_w / 2 + bar_w + 0.003, bar_y - 0.006,
                    f"{phone}  {prob:.2f}",
                    color=TEXT_C, fontsize=7, va="center",
                    transform=ax.transAxes)

        # Row 2: argmax winner
        winner     = phone_labels[pred_ids[col]]
        win_color  = "#ff7043" if winner == BLANK_TOKEN else "#66bb6a"
        win_label  = "[ blank ]" if winner == BLANK_TOKEN else f"/{winner}/"
        ax.text(cx, row_tops[2],
                win_label,
                color=win_color, fontsize=10, ha="center", va="top",
                fontweight="bold", transform=ax.transAxes)

        # Row 3: CTC accumulated sequence
        ax.text(cx, row_tops[3],
                ctc_snapshots[col],
                color="#ffd740", fontsize=8, ha="center", va="top",
                transform=ax.transAxes)

    # Arrow between row 2 and row 3 label column
    ax.annotate("", xy=(label_col_x + 0.005, row_tops[3] + 0.01),
                xytext=(label_col_x + 0.005, row_tops[2] - 0.07),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))

    # Bottom caption
    ax.text(0.5, 0.03,
            "Red = blank frame  ·  Green = phone frame  ·  "
            "Yellow bottom row = accumulated CTC output after each frame",
            color="#888888", fontsize=8, ha="center", va="bottom",
            transform=ax.transAxes)

    out = VIZ_DIR / "viz4_ctc_collapse.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved → {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo / __main__
# ─────────────────────────────────────────────────────────────────────────────

def run_all() -> None:
    print("\n" + "█" * 60)
    print("  PHASE 2 — EXTENDED VISUALIZATIONS")
    print("█" * 60 + "\n")

    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    waveform, sr, posteriors, phone_labels, decoded_str, device = load_data()

    print("\n--- Viz 1: Phone posterior heatmap ---")
    viz1_phone_heatmap(waveform, sr, posteriors, phone_labels)

    print("--- Viz 2: Blank vs non-blank timeline ---")
    viz2_blank_timeline(waveform, sr, posteriors, phone_labels)

    print("--- Viz 3: FEVERED deep dive ---")
    viz3_fevered_deepdive(waveform, sr, posteriors, phone_labels)

    print("--- Viz 4: CTC collapse walk-through ---")
    viz4_ctc_collapse(posteriors, phone_labels)

    print("\n" + "=" * 60)
    print(f"  All visualizations saved to:")
    print(f"  {VIZ_DIR}")
    for f in sorted(VIZ_DIR.iterdir()):
        print(f"    {f.name}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all()
