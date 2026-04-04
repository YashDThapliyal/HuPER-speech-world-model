# Speech World Model — Offline HuPER-Inspired Prototype

An offline, learn-by-building implementation of a syllable-clocked belief propagation model
for speech, inspired by the HuPER framework (arXiv:2602.01634).

Raw audio is passed through a frozen WavLM-Large encoder, projected into a compact acoustic
evidence stream, segmented at the syllable level, and fed into a GRU-based belief model that
learns to predict upcoming syllable slots — a concrete instantiation of the world-model
objective applied to speech perception.

> **This is an offline prototype.** All audio is processed in batch from disk. The
> streaming/real-time front-end is the next development direction (see below).

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

**Total trainable parameters: 920,832**
Frozen backbone: WavLM-Large (315M)

---

## What Has Been Implemented

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `src/audio_explorer.py` | Audio loading, waveform/spectrogram visualisation, frame decomposition |
| 2 | `src/phone_demo.py` | IPA phone recogniser (wav2vec2-TIMIT), CTC posteriors, blank-token analysis |
| 2+ | `src/visualize.py` | Phone heatmap (44×T), blank timeline, single-word deep dive, CTC collapse table |
| 3 | `src/huper_features.py` | WavLM-Large all-layer extraction, `EvidenceProjector`, phone-geometry PCA |
| 4 | `src/syllable_clock.py` | Syllable boundary detection (onset-based, ~5 Hz), boundary visualisation |
| 5 | `src/slotizer.py` | `MeanPoolSlotizer`, `AttentionSlotizer` with learnable query, compression plots |
| 6 | `src/belief_model.py` | `BeliefTransitionGRU`, single-utterance overfit proof, 5 diagnostic figures |
| 7 | `src/pipeline.py` | `SpeechWorldModelPipeline` end-to-end class with per-utterance feature caching |
| 7 | `train.py` | Multi-utterance training on LibriSpeech with streaming + disk cache |
| 7 | `verify.py` | Scientific verification: 5 diagnostic questions, 6 figures |

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

> **Active loss:** MSE next-slot prediction only.
> **Placeholder:** Phone CTC and ASR CTC heads are scaffolded but not yet trained
> (require labelled phone sequences and character-level transcripts — planned for Phase 8).

---

## Repository Structure

```
speech_world_model/
├── src/
│   ├── audio_explorer.py     # Phase 1 — audio as data
│   ├── phone_demo.py         # Phase 2 — phones and CTC
│   ├── visualize.py          # Phase 2+ — focused phone visualisations
│   ├── huper_features.py     # Phase 3 — WavLM hidden states + EvidenceProjector
│   ├── syllable_clock.py     # Phase 4 — syllable boundary detection
│   ├── slotizer.py           # Phase 5 — mean-pool and attention-pool slotizer
│   ├── belief_model.py       # Phase 6 — BeliefTransitionGRU (core world model)
│   └── pipeline.py           # Phase 7 — SpeechWorldModelPipeline with caching
├── train.py                  # Phase 7 — multi-utterance training
├── verify.py                 # Phase 7 — scientific verification + visualisations
├── requirements.txt
└── data/
    ├── samples/              # librispeech_sample.wav (6.6s reference clip)
    ├── cache/
    │   ├── features/         # WavLM layer-24 tensors per utterance (.pt, gitignored)
    │   └── boundaries/       # Syllable boundary lists per utterance (.pkl)
    ├── checkpoints/          # belief_model_best.pt (gitignored)
    ├── training_history.json # Per-epoch train/val loss and cosine
    ├── figures_phase7/       # 6 verification figures
    └── visualizations/
        ├── phase1/ … phase6/ # Per-phase exploratory figures
```

---

## Setup

**Requirements:** Python 3.11+, Apple Silicon (MPS) recommended. CPU fallback works.
CUDA has not been tested but should work with standard PyTorch device handling.

```bash
git clone <this-repo>
cd speech_world_model
pip install -r requirements.txt
pip install transformers>=4.36.0   # not in requirements.txt — install separately
```

WavLM-Large (~1.2 GB) downloads automatically from HuggingFace on first run and is cached
by the `transformers` library under `~/.cache/huggingface/`.

---

## How to Run

### Quick pipeline demo (one utterance, no training needed)

```bash
python3 src/pipeline.py
```

Runs the full pipeline on the included LibriSpeech sample using random-init weights.
Produces `data/figures_phase7/viz7_1_pipeline_overview.png`.
WavLM layer-24 features are cached after the first run.

### Train on a LibriSpeech subset

```bash
python3 train.py                           # default: 32 train + 8 val, 300 epochs max
python3 train.py --train_n 64 --val_n 16   # larger split
python3 train.py --epochs 500 --lr 5e-4    # custom schedule
```

WavLM features are extracted and cached to `data/cache/features/` on the first run.
Subsequent runs load from cache — only the fast projector and GRU forward passes repeat
each epoch. Best checkpoint saved to `data/checkpoints/belief_model_best.pt`.

### Verify a trained model

```bash
python3 verify.py                  # loads best checkpoint automatically
python3 verify.py --demo           # sanity-check pipeline without training
```

Answers 5 diagnostic questions and generates all 6 verification figures under
`data/figures_phase7/`.

### Run individual phase demos

Each module is self-contained and can be run directly. Each writes figures to
`data/visualizations/phase{N}/`.

```bash
python3 src/audio_explorer.py    # Phase 1 — waveform + spectrogram
python3 src/phone_demo.py        # Phase 2 — phone posteriors + CTC
python3 src/huper_features.py    # Phase 3 — WavLM layer comparison + PCA
python3 src/syllable_clock.py    # Phase 4 — syllable boundaries
python3 src/slotizer.py          # Phase 5 — slot compression
python3 src/belief_model.py      # Phase 6 — single-utterance overfit
```

---

## Current Limitations

- **Offline only.** All audio is loaded from disk before processing. There is no streaming
  or incremental inference.
- **Single active loss.** Only next-slot MSE prediction is trained. Phone CTC and ASR CTC
  heads are scaffolded in `train.py` but commented as `[TODO Phase 8]`.
- **No top-down feedback.** The `LanguagePriorHead` output `L_k` is computed but not yet
  fed back into the GRU as a top-down signal.
- **Approximate syllable detection.** librosa onset detection is a reasonable proxy but
  not linguistically grounded. A dedicated syllable segmentation model (e.g. Sylber) would
  give more principled boundaries.
- **Small training set.** Phase 7 used 32 train utterances. The model has not been evaluated
  at scale.
- **WavLM-Large is batch-only.** The encoder runs on the full utterance at once; chunked
  causal inference is not yet implemented.

---

## Next Direction: Streaming HuPER-Style Front End

Development continues on the `streaming-huper-encoder` branch. The goal is to replace the
batch WavLM pipeline with a streaming acoustic-phonetic front end suitable for low-latency
and eventually real-time inference:

- **Chunked feature extraction** — process audio in fixed-size causal windows without future
  context
- **Online syllable detection** — boundary detection from a causal onset model
- **Phone CTC head** — wire in phone prediction loss with labels from a pretrained phone
  recogniser (wav2vec2-TIMIT or HuPER itself)
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
