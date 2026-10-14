#$ -wd /home/bisidiqi/profound/uncer_module_logs/seg_job
#$ -l gpu=true,gpu_type=a6000|a40|a100|a100_80|a100_dgx
#$ -l tmem=32G
#$ -R y
#$ -l h_rt=24:0:0
#$ -N promis_vit_ablation

source /share/apps/source_files/cuda/cuda-11.8.source
export PATH=/share/apps/miniconda3/bin:$PATH
cd /home/bisidiqi/profound/ProFound-downstream-yp/CUL_project
source activate proFM
nvidia-smi
hostname

EPOCH=799
ABLATION_MODE=${1:-baseline}

BASE_OUTPUT=/SAN/medic/PosPretrain/project1/outputs/seg
BASE_NAME=promis_profound_vit_zone

EXTRA_ARGS=""
RUN_TAG="baseline"

case "$ABLATION_MODE" in
  baseline)
    RUN_TAG="baseline"
    ;;
  modulation)
    RUN_TAG="modulation_only"
    EXTRA_ARGS="--use_mc_dropout --mc_samples 5 --use_uncertainty_modulation --boundary_loss_weight 0.0 --calibration_loss_weight 0.0 --cucl_contrastive_weight 0.0 --cucl_calibration_weight 0.0"
    ;;
  boundary)
    RUN_TAG="boundary_only"
    EXTRA_ARGS="--use_mc_dropout --mc_samples 5 --boundary_loss_weight 0.1 --calibration_loss_weight 0.0 --cucl_contrastive_weight 0.0 --cucl_calibration_weight 0.0 --use_uncertainty_modulation"
    ;;
  calibration)
    RUN_TAG="calibration_only"
    EXTRA_ARGS="--use_mc_dropout --mc_samples 5 --boundary_loss_weight 0.0 --calibration_loss_weight 0.05 --cucl_contrastive_weight 0.0 --cucl_calibration_weight 0.0 --use_uncertainty_modulation"
    ;;
  cucl)
    RUN_TAG="cucl_only"
    EXTRA_ARGS="--use_mc_dropout --mc_samples 5 --boundary_loss_weight 0.0 --calibration_loss_weight 0.0 --use_cucl --cucl_contrastive_weight 0.05 --cucl_calibration_weight 0.1"
    ;;
  cucl_no_calibration)
    RUN_TAG="cucl_no_calibration"
    EXTRA_ARGS="--use_mc_dropout --mc_samples 5 --boundary_loss_weight 0.0 --calibration_loss_weight 0.0 --use_cucl --cucl_contrastive_weight 0.05 --cucl_calibration_weight 0.0"
    ;;
  full)
    RUN_TAG="full_um_cucl"
    EXTRA_ARGS="--use_mc_dropout --mc_samples 5 --use_uncertainty_modulation --use_cucl --boundary_loss_weight 0.1 --calibration_loss_weight 0.05 --cucl_contrastive_weight 0.05 --cucl_calibration_weight 0.1"
    ;;
  class_based)
    # Priority 1a: supervised contrastive WITHOUT uncertainty guidance.
    # Same backbone/projector/temperature as cucl_no_calibration, but anchor/
    # positive/negative selection is purely ground-truth class identity
    # (Khosla-style supervised contrastive) instead of uncertainty/correctness
    # masks. Isolates the contribution of contrastive learning itself from
    # uncertainty-guided sample selection (reviewer Weakness 2).
    RUN_TAG="cucl_class_based"
    EXTRA_ARGS="--use_mc_dropout --mc_samples 5 --boundary_loss_weight 0.0 --calibration_loss_weight 0.0 --use_cucl --cucl_contrastive_weight 0.05 --cucl_calibration_weight 0.0 --cucl_class_based"
    ;;
  classic_boundary)
    # Priority 1b: classic Kervadec-style distance-transform boundary loss +
    # plain DiceCE. No MC dropout, no uncertainty modulation, no contrastive
    # component at all (reviewer's literal "boundary loss / HD-loss" ask).
    RUN_TAG="classic_boundary_loss"
    EXTRA_ARGS="--use_classic_boundary_loss --classic_boundary_loss_weight 0.01"
    ;;
  *)
    echo "Unknown ablation mode: $ABLATION_MODE"
    echo "Valid modes: baseline | modulation | boundary | calibration | cucl | cucl_no_calibration | full | class_based | classic_boundary"
    exit 1
    ;;
esac

OUTPUT_DIR=${BASE_OUTPUT}/${BASE_NAME}_${RUN_TAG}_${EPOCH}epoch

RESUME_ARGS=""
if [ -f "${OUTPUT_DIR}/anatomy/vit_upernet_uncer/fintune/42/last.pth.tar" ]; then
  RESUME_ARGS="--resume ${OUTPUT_DIR}/anatomy/vit_upernet_uncer/fintune/42/last.pth.tar"
fi

python -u main_segmentation.py \
    --batch_size 1 \
    --model vit_upernet_uncer \
    --output_dir "${OUTPUT_DIR}" \
    --log_dir "${OUTPUT_DIR}" \
    --train fintune \
    --case prenc \
    --state train \
    --pretrain /SAN/medic/profound/output_mae/output_mae_wo_norm_r8000/checkpoint-${EPOCH}.pth \
    --sliding_window \
    --epochs 100 \
    --aux_loss_warmup_epochs 20 \
    --dataset anatomy \
    --warmup_epochs 10 \
    --lr 1e-4 \
    --seed 42 \
    ${EXTRA_ARGS} \
    ${RESUME_ARGS}
