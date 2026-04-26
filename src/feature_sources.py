"""
Module: feature_sources
Phase: 8 — streaming-huper-encoder branch
Goal: Unified interface for offline and streaming acoustic evidence extraction.

Both paths produce a FeatureBundle containing E_t: (T, 256) float32 and
shared metadata so downstream code (slotizer, belief GRU) can consume either
source without branching.

Offline path
------------
  full waveform  →  WavLM-Large (full context)  →  EvidenceProjector  →  E_t

Streaming path
--------------
  waveform in chunks  →  windowed WavLM  →  EvidenceProjector  →  E_t
  (quality degrades without right lookahead; 40 ms is the recommended operating point)

The same EvidenceProjector *instance* is passed to both functions so the
comparison isolates the windowed-attention effect, not projector weight drift.

Usage
-----
  from feature_sources import extract_offline_features, extract_streaming_features

  offline = extract_offline_features(waveform, sr, projector, wavlm, fe, device)
  stream  = extract_streaming_features(waveform, sr, projector, wavlm, fe, device,
                                       right_lookahead_ms=40)
  # offline.E_t and stream.E_t are both (T, 256) float32 numpy arrays
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from transformers import AutoFeatureExtractor, WavLMModel

import sys
sys.path.insert(0, str(Path(__file__).parent))
from huper_features import EvidenceProjector
from streaming_encoder import (
    StreamingHuPEREncoder,
    StreamingConfig,
    extract_offline_oracle,
)


# ─────────────────────────────────────────────────────────────────────────────
# FeatureBundle — shared output type for both paths
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FeatureBundle:
    """
    Acoustic evidence features produced by one extraction path.

    Fields
    ------
    E_t         : (T, 256)  float32 numpy — projected WavLM layer-24 features
    T           : int — number of frames
    source      : "offline" or "streaming"
    latency_ms  : float — algorithmic latency (0.0 for offline)
    wall_s      : float — wall-clock extraction time in seconds
    config      : dict — mode-specific parameters (lookahead_ms, chunk_ms, etc.)
    """
    E_t:        np.ndarray                         # (T, 256) float32
    T:          int
    source:     Literal["offline", "streaming"]
    latency_ms: float
    wall_s:     float
    config:     dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# extract_offline_features
# ─────────────────────────────────────────────────────────────────────────────

def extract_offline_features(
    waveform:      np.ndarray,           # (N_samples,) float32 at 16 kHz
    sr:            int,
    projector:     EvidenceProjector,
    wavlm_model:   WavLMModel,
    feat_extractor: AutoFeatureExtractor,
    device:        torch.device,
) -> FeatureBundle:
    """
    Full-context offline feature extraction.

    Runs WavLM-Large on the entire waveform at once (all frames attend to all
    other frames) then applies EvidenceProjector.

    Parameters
    ----------
    waveform       : (N_samples,) float32
    sr             : sample rate — should be 16 000
    projector      : EvidenceProjector instance (shared with streaming path)
    wavlm_model    : pre-loaded WavLMModel (avoids reloading the 315M model)
    feat_extractor : matching AutoFeatureExtractor
    device         : torch.device

    Returns
    -------
    FeatureBundle with source="offline", latency_ms=0.0
    """
    # Build a throw-away encoder purely to reuse _run_wavlm_window.
    # The encoder accepts an injected wavlm_model so WavLM is not re-loaded.
    _cfg = StreamingConfig()   # default config — only the WavLM runner is used
    _enc = StreamingHuPEREncoder(
        config         = _cfg,
        device         = device,
        projector      = projector,
        wavlm_model    = wavlm_model,
        feat_extractor = feat_extractor,
    )

    t0 = time.perf_counter()
    E_t = extract_offline_oracle(waveform, device, projector, _enc)   # (T, 256) numpy
    wall_s = time.perf_counter() - t0

    return FeatureBundle(
        E_t        = E_t,
        T          = E_t.shape[0],
        source     = "offline",
        latency_ms = 0.0,
        wall_s     = wall_s,
        config     = {
            "mode":    "offline_oracle",
            "sr":      sr,
            "n_samples": len(waveform),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# extract_streaming_features
# ─────────────────────────────────────────────────────────────────────────────

def extract_streaming_features(
    waveform:            np.ndarray,     # (N_samples,) float32 at 16 kHz
    sr:                  int,
    projector:           EvidenceProjector,
    wavlm_model:         WavLMModel,
    feat_extractor:      AutoFeatureExtractor,
    device:              torch.device,
    chunk_ms:            int = 320,
    left_context_ms:     int = 640,
    right_lookahead_ms:  int = 40,
) -> FeatureBundle:
    """
    Pseudo-streaming feature extraction using a sliding window over WavLM.

    Audio is pushed in chunk_ms increments; each window includes left_context_ms
    of history and right_lookahead_ms of future audio before frames are emitted.

    Recommended operating point: right_lookahead_ms=40 (cos ≈ 0.855 vs oracle,
    algorithmic latency = 360 ms).  Strict causal (right_lookahead_ms=0) gives
    cos ≈ 0.187 — WavLM is not trained for causal inference.

    Parameters
    ----------
    waveform             : (N_samples,) float32
    sr                   : sample rate — should be 16 000
    projector            : EvidenceProjector instance (shared with offline path)
    wavlm_model          : pre-loaded WavLMModel
    feat_extractor       : matching AutoFeatureExtractor
    device               : torch.device
    chunk_ms             : new audio per push_audio() call (default 320)
    left_context_ms      : history included in each window (default 640)
    right_lookahead_ms   : future audio required before emitting (default 40)

    Returns
    -------
    FeatureBundle with source="streaming", latency_ms=chunk_ms+right_lookahead_ms
    """
    mode = "streaming_causal" if right_lookahead_ms == 0 else "streaming_lookahead"
    cfg = StreamingConfig(
        chunk_size_ms      = chunk_ms,
        hop_size_ms        = chunk_ms,          # non-overlapping chunks
        left_context_ms    = left_context_ms,
        right_lookahead_ms = right_lookahead_ms,
        sample_rate        = sr,
        mode               = mode,
    )

    enc = StreamingHuPEREncoder(
        config         = cfg,
        device         = device,
        projector      = projector,
        wavlm_model    = wavlm_model,
        feat_extractor = feat_extractor,
    )

    chunk_samples = cfg.chunk_samples
    n_chunks = int(np.ceil(len(waveform) / chunk_samples))

    t0 = time.perf_counter()
    for i in range(n_chunks):
        enc.push_audio(waveform[i * chunk_samples : (i + 1) * chunk_samples])
    enc.flush()
    wall_s = time.perf_counter() - t0

    E_t = enc.get_emitted_features()   # (T, 256) float32 numpy

    return FeatureBundle(
        E_t        = E_t,
        T          = E_t.shape[0],
        source     = "streaming",
        latency_ms = cfg.algorithmic_latency_ms,
        wall_s     = wall_s,
        config     = {
            "mode":             mode,
            "chunk_ms":         chunk_ms,
            "left_context_ms":  left_context_ms,
            "right_lookahead_ms": right_lookahead_ms,
            "n_chunks":         n_chunks,
            "sr":               sr,
            "n_samples":        len(waveform),
        },
    )
