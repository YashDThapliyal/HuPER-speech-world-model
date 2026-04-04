"""
Module: audio_explorer
Phase: 1
Goal: Understand audio as raw numerical data — before any model touches it.

What this teaches:
- Audio is just a sequence of amplitude values sampled N times per second
- A "frame" is a short sliding window (20ms) — the unit HuPER processes
- 1 second at 16kHz = 16,000 samples ≈ 50 frames at HuPER's 20ms stride
- The log-mel spectrogram is the bridge from raw samples to perceptually-meaningful features
"""

# --- Imports ---
import math
import numpy as np
import torch
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import soundfile as sf
from pathlib import Path
from typing import Optional

matplotlib.use("Agg")  # headless backend — saves to file instead of opening GUI

# --- Constants ---
TARGET_SR: int = 16_000          # HuPER / WavLM expected sample rate
FRAME_SIZE_MS: float = 20.0      # HuPER frame length (milliseconds)
FRAME_STRIDE_MS: float = 20.0    # non-overlapping frames (stride == size)
N_MELS: int = 80                 # mel filter banks (standard for ASR)
HOP_LENGTH: int = int(TARGET_SR * FRAME_STRIDE_MS / 1000)   # 320 samples @ 16kHz
WIN_LENGTH: int = int(TARGET_SR * FRAME_SIZE_MS  / 1000)    # 320 samples @ 16kHz
HUPER_FRAMES_PER_SEC: float = TARGET_SR / HOP_LENGTH        # ≈ 50 Hz


# --- Utilities ---
def log_shape(name: str, arr: np.ndarray) -> None:
    """Print array name, shape, dtype, and value range."""
    print(f"  {name:20s}: shape={arr.shape}  dtype={arr.dtype}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:  # type: ignore[return]
    """
    Load audio file and resample to target_sr if needed.

    Returns
    -------
    waveform : np.ndarray  shape (num_samples,)  float32, range [-1, 1]
    sr       : int         actual sample rate (after resampling)
    """
    path = Path(path)
    waveform, sr = librosa.load(str(path), sr=target_sr, mono=True)
    return waveform, int(sr)


# --- Analysis Functions ---
def print_audio_info(waveform: np.ndarray, sr: int, path: str | Path) -> None:
    """Print key numerical properties of the audio."""
    duration_s = len(waveform) / sr
    num_frames  = math.ceil(len(waveform) / HOP_LENGTH)

    print("\n" + "=" * 60)
    print("AUDIO PROPERTIES")
    print("=" * 60)
    print(f"  File            : {Path(path).name}")
    print(f"  Sample rate     : {sr:,} Hz  (one sample every {1/sr*1000:.3f} ms)")
    print(f"  Duration        : {duration_s:.3f} s")
    print(f"  Num samples     : {len(waveform):,}")
    print(f"  Amplitude range : [{waveform.min():.4f}, {waveform.max():.4f}]")
    print(f"  RMS amplitude   : {np.sqrt(np.mean(waveform**2)):.4f}")
    print()
    print(f"  --- Frame decomposition ---")
    print(f"  Frame size      : {FRAME_SIZE_MS} ms  = {WIN_LENGTH} samples")
    print(f"  Frame stride    : {FRAME_STRIDE_MS} ms  = {HOP_LENGTH} samples")
    print(f"  Num frames (T)  : {num_frames}  (~{HUPER_FRAMES_PER_SEC:.1f} Hz)")
    print(f"  Frames / second : {num_frames / duration_s:.1f}")
    print("=" * 60 + "\n")


def compute_log_mel_spectrogram(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute log-mel spectrogram.

    Returns
    -------
    log_mel : np.ndarray  shape (N_MELS, T)  — frequency bins × time frames
    """
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        fmin=80.0,
        fmax=7600.0,
    )
    log_mel = librosa.power_to_db(mel_spec, ref=np.max)  # shape: (N_MELS, T)
    return log_mel


def plot_all(
    waveform: np.ndarray,
    sr: int,
    log_mel: np.ndarray,
    save_path: str | Path,
    title: str = "Phase 1 — Audio as Data",
) -> None:
    """
    Three-panel figure:
      1. Waveform (amplitude vs time)
      2. Log-mel spectrogram (frequency vs time)
      3. Zoom: first 3 frames — what ONE frame looks like as raw samples
    """
    duration_s = len(waveform) / sr
    time_axis  = np.linspace(0, duration_s, len(waveform))

    fig = plt.figure(figsize=(14, 10), facecolor="#0e1117")
    fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

    # ── Panel 1: Waveform ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time_axis, waveform, color="#4fc3f7", linewidth=0.5, alpha=0.85)
    ax1.set_facecolor("#1a1d23")
    ax1.set_xlabel("Time (seconds)", color="#aaaaaa", fontsize=9)
    ax1.set_ylabel("Amplitude", color="#aaaaaa", fontsize=9)
    ax1.set_title(
        f"Waveform  —  {len(waveform):,} samples @ {sr:,} Hz  ({duration_s:.2f} s)",
        color="white", fontsize=10,
    )
    ax1.tick_params(colors="#666666")
    ax1.spines[:].set_color("#333333")
    ax1.set_xlim(0, duration_s)
    ax1.axhline(0, color="#444444", linewidth=0.5)

    # ── Panel 2: Log-mel spectrogram ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    img = librosa.display.specshow(
        log_mel,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        fmin=80,
        fmax=7600,
        ax=ax2,
        cmap="inferno",
    )
    fig.colorbar(img, ax=ax2, format="%+2.0f dB", pad=0.01)
    ax2.set_title(
        f"Log-Mel Spectrogram  —  {N_MELS} mel bins × {log_mel.shape[1]} frames",
        color="white", fontsize=10,
    )
    ax2.set_facecolor("#1a1d23")
    ax2.tick_params(colors="#666666")
    ax2.spines[:].set_color("#333333")
    ax2.yaxis.label.set_color("#aaaaaa")
    ax2.xaxis.label.set_color("#aaaaaa")

    # ── Panel 3: First 3 frames zoomed in ───────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    show_frames = 3
    show_samples = show_frames * HOP_LENGTH
    frame_time   = np.arange(show_samples) / sr * 1000  # milliseconds

    ax3.plot(frame_time, waveform[:show_samples], color="#81c784", linewidth=1.2)
    ax3.set_facecolor("#1a1d23")

    colors = ["#ff7043", "#ffd740", "#40c4ff"]
    for i in range(show_frames):
        start_ms = i * FRAME_STRIDE_MS
        end_ms   = start_ms + FRAME_SIZE_MS
        ax3.axvspan(start_ms, end_ms, alpha=0.15, color=colors[i],
                    label=f"Frame {i}  [{start_ms:.0f}–{end_ms:.0f} ms]")
        ax3.axvline(start_ms, color=colors[i], linewidth=1.0, linestyle="--", alpha=0.7)

    ax3.legend(loc="upper right", fontsize=8, framealpha=0.3,
               labelcolor="white", facecolor="#1a1d23")
    ax3.set_xlabel("Time (ms)", color="#aaaaaa", fontsize=9)
    ax3.set_ylabel("Amplitude", color="#aaaaaa", fontsize=9)
    ax3.set_title(
        f"Frame Decomposition  —  first {show_frames} frames  "
        f"({FRAME_SIZE_MS:.0f} ms each = {WIN_LENGTH} samples each)",
        color="white", fontsize=10,
    )
    ax3.tick_params(colors="#666666")
    ax3.spines[:].set_color("#333333")
    ax3.set_xlim(0, show_frames * FRAME_STRIDE_MS)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Plot saved → {save_path}")


def verify_mps() -> torch.device:
    """Check Apple Silicon MPS availability and print status."""
    print("\n--- Device Check ---")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        # Quick smoke test
        x = torch.randn(4, 256, device=device)
        _ = x @ x.T
        print(f"  MPS (Apple Silicon GPU) : AVAILABLE ✓")
    else:
        device = torch.device("cpu")
        print(f"  MPS unavailable — falling back to CPU")
    print(f"  Active device : {device}\n")
    return device


def download_librispeech_sample(save_dir: str | Path) -> Path:
    """
    Download a single LibriSpeech clip via the HuggingFace datasets library.
    Uses the 'clean' validation split (small, no login required).

    Returns path to saved .wav file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / "librispeech_sample.wav"

    if out_path.exists():
        print(f"  Sample already exists: {out_path}")
        return out_path

    print("  Downloading LibriSpeech sample (this only happens once)...")
    from datasets import load_dataset  # noqa: PLC0415

    # streaming=True avoids downloading all shards — grabs first clip immediately
    ds = load_dataset(
        "librispeech_asr",
        "clean",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )
    sample = next(iter(ds))
    audio_array = np.array(sample["audio"]["array"], dtype=np.float32)
    native_sr   = sample["audio"]["sampling_rate"]

    # Resample to 16kHz if needed
    if native_sr != TARGET_SR:
        audio_array = librosa.resample(audio_array, orig_sr=native_sr, target_sr=TARGET_SR)

    sf.write(str(out_path), audio_array, TARGET_SR)
    print(f"  Saved to: {out_path}")
    if "text" in sample:
        print(f"  Transcript: \"{sample['text']}\"")
    return out_path


# --- Demo / __main__ ---
def run_demo(audio_path: Optional[str | Path] = None) -> None:
    """
    Full Phase 1 demo. Pass an audio_path or leave None to auto-download
    a LibriSpeech sample.
    """
    print("\n" + "█" * 60)
    print("  PHASE 1 — AUDIO AS DATA")
    print("  Speech World Model: HuPER Implementation")
    print("█" * 60)

    # ── 0. Device check ──────────────────────────────────────────────────────
    device = verify_mps()

    # ── 1. Get audio ─────────────────────────────────────────────────────────
    project_root = Path(__file__).parent.parent
    data_dir     = project_root / "data" / "samples"

    if audio_path is None:
        audio_path = download_librispeech_sample(data_dir)
    audio_path = Path(audio_path)

    # ── 2. Load & inspect ────────────────────────────────────────────────────
    print("\n--- Loading audio ---")
    waveform, sr = load_audio(audio_path, target_sr=TARGET_SR)
    log_shape("waveform", waveform)
    print_audio_info(waveform, sr, audio_path)

    # ── 3. Frame count verification ──────────────────────────────────────────
    print("--- Frame math verification ---")
    duration_s  = len(waveform) / sr
    num_frames  = math.ceil(len(waveform) / HOP_LENGTH)
    print(f"  {duration_s:.3f} s × {HUPER_FRAMES_PER_SEC:.1f} frames/s "
          f"= {num_frames} frames  (T ≈ {num_frames})")
    print(f"  [KEY INSIGHT] 1 second of audio @ {sr:,} Hz "
          f"= {sr:,} samples = ~{HUPER_FRAMES_PER_SEC:.0f} HuPER frames\n")

    # ── 4. Log-mel spectrogram ───────────────────────────────────────────────
    print("--- Computing log-mel spectrogram ---")
    log_mel = compute_log_mel_spectrogram(waveform, sr)
    log_shape("log_mel", log_mel)
    print(f"  Shape: ({N_MELS} mel_bins, {log_mel.shape[1]} time_frames)\n")

    # ── 5. Plot ───────────────────────────────────────────────────────────────
    print("--- Plotting ---")
    plot_path = project_root / "data" / "phase1_audio_explorer.png"
    plot_all(waveform, sr, log_mel, save_path=plot_path)

    # ── 6. Torch tensor demo ──────────────────────────────────────────────────
    print("\n--- Converting to torch tensor (device demo) ---")
    waveform_t = torch.from_numpy(waveform).unsqueeze(0).to(device)  # shape: (1, T)
    print(f"  waveform tensor : {waveform_t.shape} | dtype: {waveform_t.dtype} | device: {waveform_t.device}")

    # Compute energy per HOP_LENGTH block on device — shows GPU can touch audio
    frames = waveform_t.squeeze(0).unfold(0, HOP_LENGTH, HOP_LENGTH)  # shape: (num_frames, hop)
    frame_energy = frames.pow(2).mean(dim=-1)                          # shape: (num_frames,)
    print(f"  frame_energy    : {frame_energy.shape} | device: {frame_energy.device}")
    print(f"  mean frame RMS  : {frame_energy.sqrt().mean().item():.4f}")

    print("\n" + "=" * 60)
    print("  Phase 1 complete.")
    print("  What we learned:")
    print("  1. Audio is just a NumPy array of float32 values in [-1, 1]")
    print(f"  2. At {sr:,} Hz: 1 sample = {1/sr*1000:.3f} ms of time")
    print(f"  3. One HuPER frame = {FRAME_SIZE_MS:.0f} ms = {WIN_LENGTH} samples")
    print(f"  4. 1 second → ~{HUPER_FRAMES_PER_SEC:.0f} frames → this becomes T in (T, 1024)")
    print("  5. Log-mel turns raw samples into frequency × time patterns")
    print("  6. All of HuPER's complexity is finding patterns in these numbers")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_demo()
