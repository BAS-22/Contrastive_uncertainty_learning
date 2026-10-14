#!/usr/bin/env bash
# Evaluate a trained model on the test split (Dice / HD95 / SD).
#
# Usage:
#   DATA_ROOT=/path/to/data bash scripts/test_cul.sh {baseline|cul|class_based|classic_boundary}
set -euo pipefail

MODE=${1:-cul}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT}
PRETRAIN=${PRETRAIN:-pretrained/profound_mae_checkpoint-799.pth}
OUT=${OUT:-outputs}
SEED=${SEED:-42}

case "$MODE" in
  cul)         EXTRA="--use_cucl" ;;
  class_based) EXTRA="--use_cucl --cucl_class_based" ;;
  *)           EXTRA="" ;;
esac

python -u main_segmentation.py \
  --state test --model vit_upernet_uncer --train fintune --case test \
  --dataset anatomy --root "$DATA_ROOT" --pretrain "$PRETRAIN" \
  --output_dir "${OUT}/${MODE}" --log_dir "${OUT}/${MODE}" \
  --batch_size 1 --seed "$SEED" \
  --sliding_window --use_mc_dropout --mc_samples 5 --uncertainty \
  $EXTRA
