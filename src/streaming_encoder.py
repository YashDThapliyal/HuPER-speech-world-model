"""
Module: streaming_encoder
Phase: 8 — streaming-huper-encoder branch
Goal: Pseudo-streaming HuPER-style front end that ingests audio in short
      chunks and emits acoustic-phonetic features (E_t) with low latency.

Architecture
------------
audio chunks  →  internal buffer  →  windowed WavLM-Large
                                          ↓
                              layer-24 hidden states (local window)
                                          ↓
                              EvidenceProjector (Linear 1024→256 + GELU)
                                          ↓
                              emitted frames E_t: (new_frames, 256)

The encoder is "pseudo-streaming" in the sense that it uses the real
WavLM-Large model but runs it on short overlapping windows rather than
the full utterance. This limits each frame's attention context to the
local window, degrading quality relative to the offline full-context
oracle — exactly what this phase measures.

Key configurable parameters
---------------------------
chunk_size_ms     new audio delivered per push_audio() call
hop_size_ms       how many frames to emit per processing step
left_context_ms   history included in each window (improves quality)
right_lookahead_ms  future audio included before emitting (adds latency,
                    improves quality)

Supported modes
---------------
offline_oracle       full utterance, full context (reference)
streaming_causal     no right lookahead
streaming_lookahead  bounded right lookahead (40 / 80 / 160 ms)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMModel

import sys
sys.path.insert(0, str(Path(__file__).parent))
from huper_features import EvidenceProjector

# ─── constants ───────────────────────────────────────────────────────────────
WAVLM_ID     = "microsoft/wavlm-large"
TARGET_SR    = 16_000
D_RAW        = 1024        # WavLM layer-24 width
D_PROJ       = 256         # EvidenceProjector output dim
LAYER_IDX    = 24          # which WavLM transformer layer
FRAME_STRIDE = 320         # samples per output frame at 16kHz (= 20ms, 50Hz)
WAVLM_KERNEL = 400         # approximate CNN receptive field offset for frame count
# Frame count formula: T = (N - WAVLM_KERNEL) // FRAME_STRIDE + 1  for N >= WAVLM_KERNEL


# ─────────────────────────────────────────────────────────────────────────────
# Config and result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StreamingConfig:
    """All tunable parameters for the streaming encoder."""

    # How much new audio is delivered per push_audio() call
    chunk_size_ms:      int   = 320

    # Step size between successive processing windows.
    # Must be <= chunk_size_ms.  Equal to chunk_size_ms means each chunk
    # triggers exactly one window evaluation.
    hop_size_ms:        int   = 320

    # Past audio included in every window for left context.
    # Larger = better frame quality, no latency cost.
    left_context_ms:    int   = 640

    # Future audio required before a frame can be emitted.
    # right_lookahead_ms = 0  →  strictly causal (minimum latency)
    # right_lookahead_ms > 0  →  bounded lookahead (trades latency for quality)
    right_lookahead_ms: int   = 0

    sample_rate:        int   = TARGET_SR

    mode: Literal[
        "offline_oracle",
        "streaming_causal",
        "streaming_lookahead",
    ] = "streaming_causal"

    @property
    def chunk_samples(self) -> int:
        return int(self.chunk_size_ms * self.sample_rate / 1000)

    @property
    def hop_samples(self) -> int:
        return int(self.hop_size_ms * self.sample_rate / 1000)

    @property
    def left_ctx_samples(self) -> int:
        return int(self.left_context_ms * self.sample_rate / 1000)

    @property
    def lookahead_samples(self) -> int:
        return int(self.right_lookahead_ms * self.sample_rate / 1000)

    @property
    def algorithmic_latency_ms(self) -> float:
        """Worst-case delay from audio capture to feature emission (ms)."""
        return float(self.hop_size_ms + self.right_lookahead_ms)

    @property
    def window_size_ms(self) -> int:
        return self.left_context_ms + self.hop_size_ms + self.right_lookahead_ms


@dataclass
class EmissionResult:
    """Output of a single push_audio() or flush() call."""

    # Newly emitted feature frames: (new_frames, 256).  May be empty.
    features:        np.ndarray

    # Number of newly emitted frames this call
    new_frame_count: int

    # Cumulative frames emitted so far (including this call)
    total_frames:    int

    # Audio timestamps for the NEWLY emitted frames
    start_time_s:   float
    end_time_s:     float

    # Wall-clock time spent in WavLM + projection for this call (seconds)
    wall_time_s:    float

    # Effective lookahead used (ms)
    lookahead_ms:   int

    # Algorithmic latency for this call (ms)
    latency_ms:     float


# ─────────────────────────────────────────────────────────────────────────────
# StreamingHuPEREncoder
# ─────────────────────────────────────────────────────────────────────────────

class StreamingHuPEREncoder:
    """
    Pseudo-streaming HuPER-style acoustic-phonetic feature extractor.

    The encoder ingests raw audio in chunks via push_audio() and emits
    projected WavLM layer-24 features at ~50 Hz.

    Parameters
    ----------
    config     : StreamingConfig — all tunable knobs
    device     : torch.device
    projector  : pre-loaded EvidenceProjector (optional; random init if None)

    The WavLM model is loaded once on first push_audio() call and shared
    across all subsequent calls.
    """

    def __init__(
        self,
        config:         StreamingConfig,
        device:         torch.device,
        projector:      EvidenceProjector | None = None,
        wavlm_model:    WavLMModel | None = None,
        feat_extractor: AutoFeatureExtractor | None = None,
    ) -> None:
        self.config = config
        self.device = device

        # Model components — share WavLM across encoder instances to avoid
        # repeated loading.  Pass wavlm_model + feat_extractor to reuse.
        self._feat_extractor: AutoFeatureExtractor | None = feat_extractor
        self._wavlm: WavLMModel | None = wavlm_model
        self._projector: EvidenceProjector = (
            projector if projector is not None
            else EvidenceProjector(D_RAW, D_PROJ)
        )
        self._projector = self._projector.to(device)
        self._projector.eval()

        # Streaming state — reset() initialises these
        self._buffer:         np.ndarray = np.empty(0, dtype=np.float32)
        self._emit_cursor:    int        = 0    # samples processed so far
        self._frame_cursor:   int        = 0    # frames emitted so far
        self._emitted_chunks: list[np.ndarray] = []
        self._total_wall_s:   float      = 0.0

        # Stats
        self._n_windows_processed: int = 0

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all internal state.  Call before processing a new utterance."""
        self._buffer          = np.empty(0, dtype=np.float32)
        self._emit_cursor     = 0
        self._frame_cursor    = 0
        self._emitted_chunks  = []
        self._total_wall_s    = 0.0
        self._n_windows_processed = 0

    def push_audio(
        self,
        audio_chunk: np.ndarray,    # (N_samples,) float32 at config.sample_rate
        sample_rate: int = TARGET_SR,   # noqa: ARG002
    ) -> EmissionResult:
        """
        Ingest a new audio chunk and emit any newly available features.

        Parameters
        ----------
        audio_chunk : (N_samples,) float32

        Returns
        -------
        EmissionResult with newly emitted frames (may be empty if
        right_lookahead_ms > 0 and not enough future audio yet).
        """
        self._buffer = np.concatenate([self._buffer, audio_chunk])
        return self._try_emit()

    def flush(self) -> EmissionResult:
        """
        Signal end of utterance.  Pad the buffer with silence equal to the
        lookahead window and emit remaining frames.

        Always call flush() once after the last push_audio() to drain
        any frames held back by a positive right_lookahead_ms.
        """
        if self.config.lookahead_samples > 0:
            padding = np.zeros(self.config.lookahead_samples, dtype=np.float32)
            self._buffer = np.concatenate([self._buffer, padding])

        # Also pad enough for at least one full hop to avoid missing tail frames
        tail_pad = np.zeros(self.config.hop_samples, dtype=np.float32)
        self._buffer = np.concatenate([self._buffer, tail_pad])

        return self._try_emit()

    def get_emitted_features(self) -> np.ndarray:
        """
        Return all features emitted so far as a single array.

        Returns
        -------
        (T_total, 256)  float32,  or empty (0, 256) if nothing emitted yet.
        """
        if not self._emitted_chunks:
            return np.empty((0, D_PROJ), dtype=np.float32)
        return np.concatenate(self._emitted_chunks, axis=0)   # (T_total, 256)

    @property
    def total_wall_time_s(self) -> float:
        return self._total_wall_s

    @property
    def n_windows_processed(self) -> int:
        return self._n_windows_processed

    # ── core emission loop ────────────────────────────────────────────────────

    def _try_emit(self) -> EmissionResult:
        """
        Check whether any new frames can be emitted given the current buffer,
        and run WavLM + projection for each eligible window.

        A frame at position p (in samples from the start) is eligible to be
        emitted when we have buffered at least:
            p + hop_samples + lookahead_samples

        We process in steps of hop_samples.
        """
        cfg       = self.config
        new_parts: list[np.ndarray] = []
        start_wall = time.perf_counter()

        start_frame_before = self._frame_cursor

        while True:
            # Minimum buffer needed to emit the next hop
            emit_end = self._emit_cursor + cfg.hop_samples + cfg.lookahead_samples
            if emit_end > len(self._buffer):
                break   # not enough audio yet

            # Window bounds (in sample indices into buffer)
            win_start = max(0, self._emit_cursor - cfg.left_ctx_samples)
            win_end   = self._emit_cursor + cfg.hop_samples + cfg.lookahead_samples
            window    = self._buffer[win_start:win_end]   # (W_samples,)

            if len(window) < WAVLM_KERNEL:
                break   # window too short for WavLM

            # Run WavLM on this window
            layer24_win = self._run_wavlm_window(window)   # (T_win, 1024)
            self._n_windows_processed += 1

            if layer24_win.shape[0] == 0:
                self._emit_cursor += cfg.hop_samples
                continue

            # Frame indices within the window corresponding to the emit region
            # The emit region starts at sample (emit_cursor - win_start) into the window
            left_offset_samples = self._emit_cursor - win_start
            left_frame          = left_offset_samples // FRAME_STRIDE
            hop_frames          = cfg.hop_samples // FRAME_STRIDE

            end_frame = min(left_frame + hop_frames, layer24_win.shape[0])

            if left_frame >= layer24_win.shape[0]:
                self._emit_cursor += cfg.hop_samples
                continue

            emit_layer24 = layer24_win[left_frame:end_frame]   # (F, 1024)

            # Project → (F, 256)
            E_new = self._project(emit_layer24)   # (F, 256)

            new_parts.append(E_new)
            self._emitted_chunks.append(E_new)
            self._frame_cursor += E_new.shape[0]
            self._emit_cursor  += cfg.hop_samples

        wall_s = time.perf_counter() - start_wall
        self._total_wall_s += wall_s

        # Assemble result
        if new_parts:
            E_emitted = np.concatenate(new_parts, axis=0)   # (new_frames, 256)
        else:
            E_emitted = np.empty((0, D_PROJ), dtype=np.float32)

        new_count = self._frame_cursor - start_frame_before

        start_s = start_frame_before * FRAME_STRIDE / cfg.sample_rate
        end_s   = self._frame_cursor  * FRAME_STRIDE / cfg.sample_rate

        return EmissionResult(
            features        = E_emitted,
            new_frame_count = new_count,
            total_frames    = self._frame_cursor,
            start_time_s    = start_s,
            end_time_s      = end_s,
            wall_time_s     = wall_s,
            lookahead_ms    = cfg.right_lookahead_ms,
            latency_ms      = cfg.algorithmic_latency_ms,
        )

    # ── model helpers ─────────────────────────────────────────────────────────

    def _ensure_wavlm_loaded(self) -> None:
        if self._wavlm is not None:
            return
        print(f"  [StreamingEncoder] Loading {WAVLM_ID} … (one-time)")
        self._feat_extractor = AutoFeatureExtractor.from_pretrained(WAVLM_ID)
        mdl = WavLMModel.from_pretrained(WAVLM_ID)   # type: ignore[arg-type]
        mdl = mdl.to(self.device)
        mdl.eval()
        self._wavlm = mdl
        n = sum(p.numel() for p in mdl.parameters())
        print(f"  [StreamingEncoder] WavLM-Large loaded  ({n/1e6:.0f}M params)  device={self.device}")

    def _run_wavlm_window(self, window: np.ndarray) -> np.ndarray:
        """
        Run WavLM on a short audio window.

        Parameters
        ----------
        window : (W_samples,) float32

        Returns
        -------
        layer24 : (T_win, 1024) float32 numpy
        """
        self._ensure_wavlm_loaded()
        fe  = self._feat_extractor
        mdl = self._wavlm

        inp = fe(window, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)   # type: ignore[operator]
        with torch.no_grad():
            out = mdl(inp.input_values.to(self.device), output_hidden_states=True)   # type: ignore[union-attr,operator]

        layer24 = out.hidden_states[LAYER_IDX].squeeze(0).cpu().numpy()   # type: ignore[index]
        return layer24   # (T_win, 1024)

    def _project(self, layer24: np.ndarray) -> np.ndarray:
        """
        Apply EvidenceProjector: (F, 1024) → (F, 256).
        """
        x = torch.from_numpy(layer24).to(self.device)
        with torch.no_grad():
            E = self._projector(x)   # (F, 256)
        return E.cpu().numpy()

    # ── debug helpers ─────────────────────────────────────────────────────────

    def print_state(self, tag: str = "") -> None:
        cfg = self.config
        print(
            f"  [{tag or 'encoder'}] "
            f"buffer={len(self._buffer):6d}smp  "
            f"emit_cursor={self._emit_cursor:6d}smp  "
            f"emitted={self._frame_cursor:4d}fr  "
            f"windows_run={self._n_windows_processed:4d}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Offline oracle helper — full-context WavLM on the complete utterance
# ─────────────────────────────────────────────────────────────────────────────

def extract_offline_oracle(
    waveform:    np.ndarray,   # (N_samples,) float32
    device:      torch.device,
    projector:   EvidenceProjector,
    extractor:   "StreamingHuPEREncoder",   # for WavLM model reuse
) -> np.ndarray:
    """
    Run WavLM on the full utterance (full context) and project.

    Returns
    -------
    E_offline : (T, 256)  full-context acoustic evidence
    """
    layer24 = extractor._run_wavlm_window(waveform)   # (T, 1024)
    projector.eval()
    x = torch.from_numpy(layer24).to(device)
    with torch.no_grad():
        E = projector(x)
    return E.cpu().numpy()   # (T, 256)


# ─────────────────────────────────────────────────────────────────────────────
# Factory — build configs for the 4 streaming modes used in the evaluation
# ─────────────────────────────────────────────────────────────────────────────

def make_eval_configs(
    chunk_size_ms:    int = 320,
    left_context_ms:  int = 640,
) -> list[StreamingConfig]:
    """
    Return the four streaming configs used in the Phase 8 evaluation:
      causal (0ms lookahead) and three lookahead settings.
    """
    configs = []
    for la_ms in [0, 40, 80, 160]:
        mode = "streaming_causal" if la_ms == 0 else "streaming_lookahead"
        configs.append(StreamingConfig(
            chunk_size_ms      = chunk_size_ms,
            hop_size_ms        = chunk_size_ms,   # non-overlapping chunks
            left_context_ms    = left_context_ms,
            right_lookahead_ms = la_ms,
            mode               = mode,
        ))
    return configs


# ─────────────────────────────────────────────────────────────────────────────
# Demo — __main__
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import librosa
    ROOT = Path(__file__).parent.parent
    SAMPLE = ROOT / "data" / "samples" / "librispeech_sample.wav"

    print("█" * 56)
    print("  PHASE 8 — streaming_encoder.py DEMO")
    print("  Streaming HuPER front end on one utterance")
    print("█" * 56)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n  Device: {device}")

    waveform, sr = librosa.load(str(SAMPLE), sr=16000, mono=True)
    waveform = waveform.astype(np.float32)
    duration_s = len(waveform) / sr
    print(f"  Audio: {duration_s:.2f}s  ({len(waveform)} samples @ {sr}Hz)")

    projector = EvidenceProjector(D_RAW, D_PROJ).to(device)
    projector.eval()

    for la_ms in [0, 80, 160]:
        cfg = StreamingConfig(
            chunk_size_ms      = 320,
            hop_size_ms        = 320,
            left_context_ms    = 640,
            right_lookahead_ms = la_ms,
            mode               = "streaming_causal" if la_ms == 0 else "streaming_lookahead",
        )
        enc = StreamingHuPEREncoder(cfg, device, projector=projector)

        chunk_samples = cfg.chunk_samples
        n_chunks = int(np.ceil(len(waveform) / chunk_samples))

        t0 = time.perf_counter()
        for i in range(n_chunks):
            chunk = waveform[i * chunk_samples : (i + 1) * chunk_samples]
            enc.push_audio(chunk)
        enc.flush()
        elapsed = time.perf_counter() - t0

        E = enc.get_emitted_features()
        rtf = elapsed / duration_s
        print(f"\n  lookahead={la_ms:3d}ms  |  "
              f"T_emitted={E.shape[0]:4d}  shape={E.shape}  |  "
              f"latency={cfg.algorithmic_latency_ms:.0f}ms  |  "
              f"RTF={rtf:.3f}  |  windows={enc.n_windows_processed}")
        enc.print_state(f"la={la_ms}ms")

    print("\n  Demo complete.")
