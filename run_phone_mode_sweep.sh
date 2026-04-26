#!/usr/bin/env bash
set -euo pipefail

# Sweep settings
TRAIN_N="${TRAIN_N:-32}"
VAL_N="${VAL_N:-8}"
EPOCHS="${EPOCHS:-25}"
PATIENCE="${PATIENCE:-8}"
LOG_EVERY="${LOG_EVERY:-2}"
N_UTT_EVAL="${N_UTT_EVAL:-8}"
LOOKAHEADS="${LOOKAHEADS:-0 40 80}"
LOSS_WEIGHTS="${LOSS_WEIGHTS:-0.5 1.0 2.0}"

# Optional: enable MPS fallback for unsupported ops like CTC on Apple GPU.
# Set ENABLE_MPS_FALLBACK=0 to disable.
ENABLE_MPS_FALLBACK="${ENABLE_MPS_FALLBACK:-1}"
if [[ "$ENABLE_MPS_FALLBACK" == "1" ]]; then
  export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

CKPT_DIR="data/checkpoints"
SWEEP_DIR="data/figures_phase8_final/sweeps"
mkdir -p "$CKPT_DIR" "$SWEEP_DIR"

echo "=============================================================="
echo "Phone Mode Sweep"
echo "train_n=$TRAIN_N val_n=$VAL_N epochs=$EPOCHS patience=$PATIENCE"
echo "loss_weights=[$LOSS_WEIGHTS] lookaheads=[$LOOKAHEADS] eval_n_utt=$N_UTT_EVAL"
echo "=============================================================="

echo
echo "[1/3] Training checkpoints for each phone_loss_weight..."
for W in $LOSS_WEIGHTS; do
  TAG="w$(echo "$W" | tr -d '.')"
  OUT_CKPT="$CKPT_DIR/phone_${TAG}.pt"

  echo
  echo "--- Training weight=$W -> $OUT_CKPT ---"
  python3 train.py \
    --enable_phone_mode \
    --phone_eval \
    --train_n "$TRAIN_N" \
    --val_n "$VAL_N" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --log_every "$LOG_EVERY" \
    --phone_loss_weight "$W"

  cp "$CKPT_DIR/belief_model_best.pt" "$OUT_CKPT"
done

echo
echo "[2/3] Running eval grid (checkpoint x lookahead)..."
for CKPT in "$CKPT_DIR"/phone_w*.pt; do
  CKPT_BASE="$(basename "$CKPT" .pt)"
  for LA in $LOOKAHEADS; do
    OUT_JSON="$SWEEP_DIR/${CKPT_BASE}_la${LA}.json"
    echo "--- Eval $CKPT_BASE at lookahead=${LA}ms -> $OUT_JSON ---"
    python3 phone_mode_eval.py \
      --ckpt "$CKPT" \
      --n_utt "$N_UTT_EVAL" \
      --lookahead_ms "$LA" \
      --out_json "$OUT_JSON"
  done
done

echo
echo "[3/3] Summary table"
python3 - <<'PY'
import glob, json, os

rows = []
for p in sorted(glob.glob("data/figures_phase8_final/sweeps/*.json")):
    with open(p, "r") as fh:
        d = json.load(fh)
    a = d["aggregate"]
    name = os.path.basename(p).replace(".json", "")
    rows.append((
        name,
        a.get("offline_per_vs_teacher", float("nan")),
        a.get("streaming_per_vs_teacher", float("nan")),
        a.get("offline_vs_streaming_seq_agreement", float("nan")),
        a.get("offline_vs_streaming_seq_len_ratio", float("nan")),
    ))

print(f"{'run':28s}  {'off_PER':>8}  {'str_PER':>8}  {'seq_agree':>10}  {'len_ratio':>10}")
for r in rows:
    print(f"{r[0]:28s}  {r[1]:8.4f}  {r[2]:8.4f}  {r[3]:10.4f}  {r[4]:10.4f}")
PY

echo
echo "Done. JSON outputs are in: $SWEEP_DIR"
