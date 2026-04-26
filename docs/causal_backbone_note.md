# Why a Native Causal Backbone Is Needed to Eliminate Lookahead

**Date:** 2026-04-10
**Branch:** `streaming-huper-encoder`
**Applies to:** Phase 8 windowed pseudo-streaming encoder (`src/streaming_encoder.py`)

---

## What "pseudo-streaming" means

The current streaming encoder runs the **unchanged WavLM-Large** model on short
overlapping windows of audio rather than on the full utterance.  The window is:

```
[left_context_640ms | new_chunk_320ms | right_lookahead_Xms]
```

Within each window, WavLM's self-attention is **fully bidirectional** — every
frame attends to every other frame inside the window.  What is absent is
*cross-window attention*: frames in the current window cannot attend to frames
in the previous or future windows.

This is not true causal inference — it is a batch model being re-run on
successive short excerpts.

---

## Why 40 ms of right lookahead recovers most of the quality

WavLM-Large was trained with full bidirectional context.  Each transformer layer
has learned to rely on a local neighbourhood of frames on **both sides** of the
current frame.  Empirically, ~2 frames (40 ms) to the right of a chunk boundary
is sufficient to restore the attention patterns for the boundary frames to
approximately match what they receive in the full-context setting.

Phase 8 results confirm this sharply:

| Mode | Latency | Frame cos vs oracle |
|---|---|---|
| Causal (0 ms) | 320 ms | **0.187** |
| +40 ms lookahead | 360 ms | **0.855** |
| +160 ms lookahead | 480 ms | 0.905 |

The jump from 0 ms to 40 ms (+0.668) dwarfs the gain from 40 ms to 160 ms
(+0.050).  The dominant effect is the right-context starvation at boundary
frames.  A small right buffer cures it.

---

## Why strict causal inference (0 ms) is catastrophic today

At 0 ms lookahead, WavLM processes a window that ends exactly at the chunk
boundary.  The rightmost frames of that window see only zero-padding where real
audio would be.  This is an **out-of-distribution input** for a model that was
never trained with right-masked attention.

Every attention head in every layer re-weights around the zero-boundary
artefact.  The result is not just noisier features — the feature geometry is
qualitatively wrong (cos ≈ 0.187).  This is a training-distribution mismatch,
not a buffering or implementation bug.

---

## What eliminating the lookahead requirement would actually take

### Option A — Natively causal transformer backbone

Train a bidirectional-equivalent model with a **triangular causal attention
mask** so each frame only attends to past frames.  Examples:

- Streaming Conformer (Google, 2021)
- Emformer (Meta, 2021)
- Mamba-based SSM (state-space) speech encoders

These models achieve near-bidirectional quality on many ASR benchmarks with
0 ms algorithmic lookahead, but require training from scratch or fine-tuning
on hundreds to thousands of hours of speech.

### Option B — Chunk-trained bidirectional model

Keep the bidirectional architecture but train with **block-causal attention
masks** so the model learns not to rely on cross-chunk attention.  During
inference, run each chunk without lookahead.

Examples: chunk-wise wav2vec2, HuBERT fine-tuned with block attention masking.
This is cheaper than a full causal retrain but still requires a dedicated
training run.

### Option C — Small learned right-buffer predictor

Keep WavLM frozen.  Train a small model to **predict the next 40 ms of features**
conditioned on the current belief state and use the prediction as a synthetic
right context.  This is speculative and the quality of synthetic context is
bounded by the predictor quality.  The BeliefTransitionGRU already produces
next-slot predictions (Ŝ_{k+1}) that could be the seed for this approach.

---

## Recommended operating point for Phase 8

Ship `right_lookahead_ms=40` (360 ms total algorithmic latency) as the operating
point.  This gives:

- Frame cosine vs oracle: **0.877**
- Slot cosine vs oracle: **0.910**
- Belief cosine vs oracle: **0.988**
- Prediction cosine vs oracle: **0.958**
- RTF on M3 Pro MPS: **~0.20** (real-time capable)

The downstream belief states are almost perfectly aligned with the offline
oracle (0.988) despite the 12% frame-level degradation — the `BeliefTransitionGRU`
is robust to the upstream noise.  This makes the 40 ms lookahead operating
point practically viable.

Treat the causal-backbone swap as a separate multi-week research track.

---

## References

- Park et al., *Streaming End-to-End Speech Recognition for Mobile Devices* (2019)
- Shi et al., *Emformer: Efficient Memory Transformer Based Acoustic Model for Low Latency Streaming Speech Recognition* (2021)
- Chen et al., *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing* (2022)
- HuPER: *A Human-Inspired Framework for Phonetic Perception* (arXiv:2602.01634)
