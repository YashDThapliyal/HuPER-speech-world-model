# Branch Workflow (streaming-huper-encoder)

This branch now supports two tracks:
- world-model training/inference (default)
- optional phone mode (`--enable_phone_mode`)

## Recommended commit boundaries

1. Hygiene only
- `.gitignore`
- workflow/docs only

2. Phone pseudo-label cache layer
- teacher label extraction/cache logic
- no model architecture changes

3. Model/loss wiring
- phone head
- optional CTC loss in training/checkpoints

4. Inference wiring
- pipeline phone-mode outputs for offline/streaming
- dedicated inference CLI

5. Eval CLI
- frame agreement, PER-style metric, offline-vs-streaming agreement

6. README refresh
- usage docs + status update

## Track vs ignore

Track source/runtime files:
- `src/*.py`
- `train.py`, `phone_infer.py`, `phone_mode_eval.py`
- docs updates

Do not track generated artifacts:
- `data/cache/**`
- `data/checkpoints/**`
- `data/figures_phase*/**`
- temporary eval wavs/jsons produced during local runs

## Repro commands

Train (default world-model only):
```bash
python3 train.py --train_n 8 --val_n 2 --epochs 2
```

Train with phone mode:
```bash
python3 train.py --enable_phone_mode --phone_eval --train_n 8 --val_n 2 --epochs 2
```

Infer phones:
```bash
python3 phone_infer.py --mode both --lookahead_ms 40
```

Eval phone mode:
```bash
python3 phone_mode_eval.py --n_utt 2 --lookahead_ms 40
```
