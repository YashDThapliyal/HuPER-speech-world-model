# Speech World Model — Streaming-Capable HuPER-Inspired Prototype

A learn-by-building implementation of a syllable-clocked belief propagation model
for speech, inspired by the HuPER framework (arXiv:2602.01634), with integrated
phone-mode training/inference/evaluation.

Raw audio is passed through a frozen WavLM-Large encoder, projected into a compact acoustic
evidence stream, segmented at the syllable level, and fed into a GRU-based belief model that
learns to predict upcoming syllable slots — a concrete instantiation of the world-model
objective applied to speech perception.

> **Current state:** offline + pseudo-streaming runtime paths are supported, and
> the branch includes an integrated trainable phone mode with reproducible eval tooling.

---

## Motivation

Standard ASR maps audio directly to text. Human speech perception doesn't work that way.
Listeners maintain a running internal state that is updated ~4–5 times per second at the
syllable boundary, balancing bottom-up acoustic evidence with top-down lexical predictions.

This project asks: *can we model that dynamics layer?*

The architecture separates the two timescales explicitly:

- **Fast (50 Hz):** WavLM-Large produces a frame-level acoustic evidence stream
- **Slow (~5 Hz):** A GRU belief model updates at each syllable tick, predicting what comes next

The prediction objective — *given the current belief, predict the next syllable slot* — is
what turns this from a multi-task encoder into a world model.

---

## What the System Does

Given a speech waveform:

1. **Feature extraction** — WavLM-Large layer 24 produces `(T, 1024)` hidden states at ~50 Hz
2. **Evidence projection** — A trainable `Linear(1024→256) + GELU` compresses to `E_t: (T, 256)`
3. **Syllable segmentation** — librosa onset detection partitions frames into K syllable slots (~5 Hz)
4. **Slotization** — Mean-pooling or attention-pooling collapses `E_t` into `S_k: (K, 256)`
5. **Belief transition** — A GRU reads `S_k` sequentially and maintains belief states `B_k: (K, 256)`
6. **Next-slot prediction** — An MLP predicts `Ŝ_{k+1}` from `B_k`; the cosine alignment between `Ŝ_{k+1}` and `S_{k+1}` is the world-model signal

---

## Architecture

```
Audio (16 kHz)
      │
      ▼
WavLM-Large [FROZEN, 315M params]
Layer-24 hidden states: (T, 1024) @ ~50 Hz
      │
      ▼
EvidenceProjector [TRAINABLE, 262K params]
Linear(1024→256) + GELU → E_t: (T, 256)
      │
      ├──────────────────────────┐
      ▼                          ▼
Onset-based Syllable Clock   MeanPoolSlotizer
K boundaries @ ~5 Hz         or AttentionSlotizer
                             S_k: (K, 256)
      │                          │
      └──────────────────────────┘
                     │
                     ▼
        BeliefTransitionGRU [TRAINABLE, 658K params]
        GRU(256→256) + LayerNorm
        B_k: (K, 256)  — belief state after tick k
              │
     ┌────────┴────────┐
     ▼                 ▼
NextSlotPredictor  LanguagePriorHead
MLP → Ŝ_{k+1}     MLP → L_k (auxiliary)

Loss: MSE(Ŝ_{k+1}, S_{k+1})
      + monitor cosine(Ŝ_{k+1}, S_{k+1})
```

**Total trainable parameters: 920,832 (world-model core only)**
**Phone mode adds:** `PhoneCTCHead` (~11,308 params with current vocab)
Frozen backbone: WavLM-Large (315M)

---

## What Has Been Implemented

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `src/audio_explorer.py` | Audio loading, waveform/spectrogram visualisation, frame decomposition |
| 2 | `src/phone_demo.py` | Legacy standalone phone demo (teacher posterior inspection and CTC collapse visuals) |
| 2+ | `src/visualize.py` | Focused diagnostics and phone visualisations |
| 3 | `src/huper_features.py` | WavLM-Large all-layer extraction, `EvidenceProjector`, phone-geometry PCA |
| 4 | `src/syllable_clock.py` | Syllable boundary detection (onset-based, ~5 Hz), boundary visualisation |
| 5 | `src/slotizer.py` | `MeanPoolSlotizer`, `AttentionSlotizer` with learnable query, compression plots |
| 6 | `src/belief_model.py` | `BeliefTransitionGRU`, single-utterance overfit proof, 5 diagnostic figures |
| 7 | `src/pipeline.py` | Main runtime pipeline (offline + pseudo-streaming) with optional integrated phone decode |
| 8 | `src/phone_ctc.py` | Phone teacher wrapper, pseudo-label cache, CTC loss/decode helpers, PER/edit-distance metrics |
| 8 | `train.py` | World-model training with optional additive phone CTC objective (`--enable_phone_mode`) |
| 8 | `phone_infer.py` | Phone inference CLI from checkpoint (offline/streaming/both, decode tuning presets) |
| 8 | `phone_mode_eval.py` | Reproducible phone eval CLI with JSON artifacts and aggregate metrics |
| 8 | `run_phone_mode_sweep.sh` | Checkpoint/lookahead/loss-weight sweep harness for operating-point search |
| 8 | `verify.py` | Scientific verification for world-model behavior (non-phone diagnostics) |

---

## Results

### Phase 6 — Single-utterance overfit (proof of architecture)

Trained on one 6.6-second LibriSpeech clip (K=34 syllable slots):

| Metric | Value |
|--------|-------|
| Final train loss (MSE) | ~0.000000 |
| Final prediction cosine | 1.0000 |
| Belief smoothness cos(B_k, B_{k+1}) | 0.941 |

The GRU can memorise the transition dynamics of a single utterance to within floating-point
precision. This confirms the architecture's capacity and that the training signal is well-formed.
Belief smoothness of 0.941 falls in the healthy range (0.6–0.95), confirming the GRU is
accumulating context rather than copying its input.

### Phase 7 — Multi-utterance training (32 train / 8 val, LibriSpeech)

`EvidenceProjector + BeliefTransitionGRU` trained jointly for 57 epochs (~2.7 min, M3 Pro):

| Metric | Value |
|--------|-------|
| Best val loss (MSE) | 0.00270 |
| Val prediction cosine (at best checkpoint) | 0.785 |
| Train prediction cosine (final) | 0.851 |
| Train / val generalisation gap | 0.066 |
| Belief smoothness (val, mean) | 0.972 |
| Mean slot count per utterance | 59.3 |

The train/val gap of 0.066 indicates mild overfitting, expected at this dataset scale (32
utterances). Belief smoothness stays well within the healthy range across all utterances.
The model generalises to unseen utterances — prediction cosine of 0.785 on val vs. 0.851 on
train confirms the representations are not utterance-specific.

> **Default loss:** MSE next-slot prediction.
> **Optional phone mode:** frame-level phone CTC head on `E_t` with pseudo-label
> supervision from a frozen teacher model.

### Phase 8 — Integrated phone mode (best validated operating point)

Using checkpoint `data/checkpoints/phone_w20.pt` and tuned decode preset
(`lookahead_ms=80`, `blank_bias=0.2`, `min_phone_conf=0.35`) on 20 validation utterances:

| Metric | Value |
|--------|-------|
| Offline frame accuracy vs teacher | 0.6447 |
| Streaming frame accuracy vs teacher | 0.3121 |
| Offline PER vs teacher | 0.1563 |
| Streaming PER vs teacher | 0.1785 |
| Offline vs streaming sequence agreement | 0.8873 |
| Streaming/offline sequence length ratio | 1.0132 |

Interpretation:
- Offline and streaming decoded phone sequences are closely matched at sequence level.
- Streaming sequence lengths are stable (near 1.0 ratio), meaning decode behavior is not drifting.
- Streaming frame-level token agreement is lower than offline, so near-term improvements should
  focus on frame alignment under streaming constraints rather than sequence collapse behavior.

---

## Repository Structure

```text
speech_world_model/
├── src/
│   ├── audio_explorer.py     # Phase 1 — audio as data
│   ├── phone_demo.py         # Legacy standalone phone analysis demo
│   ├── phone_ctc.py          # Phone teacher/pseudo-label/CTC/decode utilities
│   ├── visualize.py          # Focused phone/world-model visual diagnostics
│   ├── huper_features.py     # Phase 3 — WavLM hidden states + EvidenceProjector
│   ├── feature_sources.py    # Shared feature source abstractions
│   ├── streaming_encoder.py  # Pseudo-streaming WavLM feature extraction
│   ├── syllable_clock.py     # Phase 4 — syllable boundary detection
│   ├── slotizer.py           # Phase 5 — mean-pool and attention-pool slotizer
│   ├── belief_model.py       # Phase 6 — BeliefTransitionGRU (core world model)
│   └── pipeline.py           # Main pipeline API (offline + pseudo-streaming + phone mode)
├── train.py                  # Main training entrypoint (world model + optional phone mode)
├── phone_infer.py            # Phone inference CLI from checkpoint
├── phone_mode_eval.py        # Phone eval CLI with JSON outputs
├── run_phone_mode_sweep.sh   # End-to-end sweep (loss weight x lookahead)
├── verify.py                 # World-model verification + visualisations
├── streaming_eval.py         # Streaming-focused eval helpers
├── streaming_pipeline_eval.py
├── production_wiring_eval.py
├── requirements.txt
├── data/
    ├── samples/              # reference clips
    ├── cache/
    │   ├── features/         # WavLM tensors per utterance (.pt, gitignored)
    │   ├── boundaries/       # Syllable boundary lists (.pkl)
    │   └── phone_labels/     # Cached teacher pseudo labels for phone mode
    ├── checkpoints/          # best and sweep checkpoints (gitignored)
    ├── figures_phase7/       # world-model verification figures
    ├── figures_phase8_final/ # phone-mode eval JSON + reports
    └── training_history.json # per-epoch logs
└── docs/
    ├── BRANCH_WORKFLOW.md
    └── PHONE_MODE_SESSION_REPORT_2026-04-25.md/.pdf
```

---

## Setup

**Requirements:** Python 3.11+, Apple Silicon (MPS) recommended. CPU fallback works.
CUDA has not been tested but should work with standard PyTorch device handling.

```bash
git clone <this-repo>
cd speech_world_model
pip install -r requirements.txt
```

WavLM-Large (~1.2 GB) downloads automatically from HuggingFace on first run and is cached
by the `transformers` library under `~/.cache/huggingface/`.

---

## How to Run

### Quick world-model demo (one utterance, no training needed)

```bash
python3 src/pipeline.py
```

Runs the full world-model pipeline on the included LibriSpeech sample using random-init weights.
Produces `data/figures_phase7/viz7_1_pipeline_overview.png`.
WavLM layer-24 features are cached after the first run.

### Train (world model only, default path)

```bash
python3 train.py                           # default world-model objective only
python3 train.py --train_n 64 --val_n 16   # larger split
python3 train.py --epochs 500 --lr 5e-4    # custom schedule
```

WavLM features are extracted and cached to `data/cache/features/` on the first run.
Subsequent runs load from cache — only the fast projector and GRU forward passes repeat
each epoch. Best checkpoint saved to `data/checkpoints/belief_model_best.pt`.

### Train with integrated phone mode (recommended settings)

```bash
# Current recommended phone-mode training setup
python3 train.py --enable_phone_mode --phone_eval --phone_loss_weight 2.0 \
  --train_n 32 --val_n 8 --epochs 40 --patience 10 --log_every 2
```

Notes:
- On Apple Silicon, CTC may use CPU fallback while the rest stays on MPS.
- Teacher pseudo-labels are cached under `data/cache/phone_labels/` for reuse.

### Verify a trained model

```bash
python3 verify.py                  # loads best checkpoint automatically
python3 verify.py --demo           # sanity-check pipeline without training
```

Answers 5 diagnostic questions and generates all 6 verification figures under
`data/figures_phase7/`.

### Phone-mode inference and eval CLIs

```bash
# Inference (offline + streaming) using tuned preset by default
python3 phone_infer.py --mode both

# Explicit tuning controls
python3 phone_infer.py --mode both --preset none --lookahead_ms 80 \
  --blank_bias 0.2 --min_phone_conf 0.35

# Eval (JSON report, tuned preset by default)
python3 phone_mode_eval.py --ckpt data/checkpoints/phone_w20.pt --n_utt 20
```

### Sweep script (checkpoint + lookahead search)

```bash
./run_phone_mode_sweep.sh
```

---

## Current Limitations

- **Pseudo-streaming encoder.** Streaming inference is available, but uses windowed
  WavLM rather than a natively causal backbone.
- **Phone supervision is pseudo-labeled.** Phone mode currently learns from a frozen
  teacher model, not human phone transcripts.
- **No top-down feedback.** The `LanguagePriorHead` output `L_k` is computed but not yet
  fed back into the GRU as a top-down signal.
- **Approximate syllable detection.** librosa onset detection is a reasonable proxy but
  not linguistically grounded. A dedicated syllable segmentation model (e.g. Sylber) would
  give more principled boundaries.
- **Small training set.** Phase 7 used 32 train utterances. The model has not been evaluated
  at scale.
- **ASR text decoding not integrated.** The current recognizer outputs phone sequences; a
  grapheme/word end-to-end path is a separate next milestone.

---

## Next Direction: Toward Fully Causal Real-Time

Development continues on the `streaming-huper-encoder` branch. The goal is to replace the
batch WavLM pipeline with a streaming acoustic-phonetic front end suitable for low-latency
and eventually real-time inference:

- **Chunked feature extraction** — process audio in fixed-size causal windows without future
  context
- **Online syllable detection** — boundary detection from a causal onset model
- **ASR CTC head** — connect belief states to character-level CTC using LibriSpeech transcripts
  (transcripts are already available in the HuggingFace dataset stream)
- **Incremental belief updates** — the GRU step is already causal; the bottleneck is the
  upstream encoder

The belief transition model itself (`BeliefTransitionGRU`) requires no changes — it processes
one slot at a time and is already structurally causal. Phase 8 is about making the evidence
stream that feeds it causal as well.

---

## References

- HuPER: *A Human-Inspired Framework for Phonetic Perception* — [arXiv:2602.01634](https://arxiv.org/abs/2602.01634)
- WavLM: *Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing* — [arXiv:2110.13900](https://arxiv.org/abs/2110.13900)
- Ha & Schmidhuber, *World Models* — [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)
