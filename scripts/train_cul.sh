#!/usr/bin/env bash
# Train CUL / baseline / ablations from the paper (scheduler-agnostic).
#
# Usage:
#   DATA_ROOT=/path/to/data PRETRAIN=pretrained/profound_mae_checkpoint-799.pth \
#     bash scripts/train_cul.sh {baseline|cul|class_based|classic_boundary}
set -euo pipefail

MODE=${1:-cul}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT}
PRETRAIN=${PRETRAIN:-pretrained/profound_mae_checkpoint-799.pth}
OUT=${OUT:-outputs}
SEED=${SEED:-42}

case "$MODE" in
  baseline)
    EXTRA="" ;;
  cul)
    EXTRA="--use_mc_dropout --mc_samples 5 --use_cucl --cucl_contrastive_weight 0.05 --cucl_calibration_weight 0.0 --boundary_loss_weight 0.0 --calibration_loss_weight 0.0" ;;
  class_based)
    EXTRA="--use_mc_dropout --mc_samples 5 --use_cucl --cucl_class_based --cucl_contrastive_weight 0.05 --cucl_calibration_weight 0.0 --boundary_loss_weight 0.0 --calibration_loss_weight 0.0" ;;
  classic_boundary)
    EXTRA="--use_classic_boundary_loss --classic_boundary_loss_weight 0.01" ;;
  *)
    echo "Unknown mode: $MODE (baseline|cul|class_based|classic_boundary)"; exit 1 ;;
esac

RESUME=""
CKPT="${OUT}/${MODE}/anatomy/vit_upernet_uncer/fintune/${SEED}/last.pth.tar"
[[ -f "$CKPT" ]] && RESUME="--resume $CKPT"

python -u main_segmentation.py \
  --state train --model vit_upernet_uncer --train fintune --case prenc \
  --dataset anatomy --root "$DATA_ROOT" --pretrain "$PRETRAIN" \
  --output_dir "${OUT}/${MODE}" --log_dir "${OUT}/${MODE}" \
  --batch_size 1 --epochs 100 --warmup_epochs 10 --lr 1e-4 --seed "$SEED" \
  --aux_loss_warmup_epochs 20 --sliding_window \
  $EXTRA $RESUME
