# Phase 8 — Streaming Encoder Update

**TL;DR:** Implemented a causal, chunk-by-chunk streaming front-end for the WavLM-Large
evidence projector and measured its fidelity against a full-context offline oracle.

---

## What's new

- `src/streaming_encoder.py` — `StreamingHuPEREncoder` processes audio in 320 ms hops
  with a 640 ms sliding left context.  Right lookahead is a tunable knob.
- `streaming_eval.py` — evaluation harness: streams 8 LibriSpeech val utterances,
  compares streaming features vs. oracle frame-by-frame.

## Numbers (8 utterances, LibriSpeech val)

| Mode | Latency | Cosine to oracle |
|------|---------|-----------------|
| Causal (0 ms) | 320 ms | **0.187** ± 0.066 |
| +40 ms lookahead | 360 ms | **0.855** ± 0.016 |

- Strict causal inference achieves **0.187** cosine to the offline oracle at 320 ms algorithmic latency.
- Adding a 40 ms right lookahead improved cosine by **+0.668** at only 40 ms extra latency.
- RTF (causal, M3 Pro MPS) = **0.158** — below 1.0, real-time capable.
- Chunk boundary artefacts are small (mean boundary drop < 0.05 across all modes).

## What this enables

The belief GRU downstream is already causal and slot-by-slot.  Now the upstream evidence
stream is also causal.  The full inference path can run incrementally with bounded latency.

Next: wire Phone CTC and ASR CTC training losses into the streaming pipeline.

---

![Latency quality tradeoff](phase8_fig1_latency_quality.png)

![Bar quality summary](phase8_fig2_bar_quality.png)
