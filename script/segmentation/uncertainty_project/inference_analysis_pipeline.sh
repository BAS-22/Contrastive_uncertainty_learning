#$ -wd /home/bisidiqi/profound/uncer_module_logs/seg_job
#$ -l gpu=true,gpu_type=a6000|a40|a100|a100_80|a100_dgx
#$ -l tmem=32G
#$ -R y
#$ -l h_rt=08:0:0
#$ -N seg_uncer_pipeline

set -euo pipefail

source /share/apps/source_files/cuda/cuda-11.8.source
export PATH=/share/apps/miniconda3/bin:$PATH
cd /home/bisidiqi/profound/ProFound-downstream-yp/CUL_project
source activate proFM
nvidia-smi
hostname

EPOCH=${EPOCH:-799}
SEED=${SEED:-42}
DATASET=${DATASET:-anatomy}
BASE_OUTPUT=${BASE_OUTPUT:-/SAN/medic/PosPretrain/project1/outputs/seg}
BASE_NAME=${BASE_NAME:-promis_profound_vit_zone}
BEST_TAG=${BEST_TAG:-cucl_no_calibration}
SKIP_TESTS=${SKIP_TESTS:-0}
SKIP_BASELINE_TEST=${SKIP_BASELINE_TEST:-0}
SKIP_BEST_TEST=${SKIP_BEST_TEST:-0}
SKIP_COMPARISON=${SKIP_COMPARISON:-0}
SKIP_REPORT=${SKIP_REPORT:-0}
SKIP_STRUCTURE_REPORT=${SKIP_STRUCTURE_REPORT:-0}
HARD_SPLIT_PERCENTILE=${HARD_SPLIT_PERCENTILE:-50}

# Allow BEST_TAG override from positional argument for robust qsub usage:
# qsub script/segmentation/uncertainty_project/inference_analysis_pipeline.sh cucl_only
if [[ -n "${1:-}" ]]; then
  BEST_TAG="${1}"
fi

if [[ "${SKIP_TESTS}" == "1" ]]; then
  SKIP_BASELINE_TEST=1
  SKIP_BEST_TEST=1
fi

echo "Resolved config: EPOCH=${EPOCH}, SEED=${SEED}, DATASET=${DATASET}, BEST_TAG=${BEST_TAG}"

BEST_EXTRA_ARGS=""
case "${BEST_TAG}" in
  full_um_cucl)
    BEST_EXTRA_ARGS="--use_uncertainty_modulation --use_cucl"
    ;;
  cucl_only)
    BEST_EXTRA_ARGS="--use_cucl"
    ;;
  cucl_no_calibration)
    BEST_EXTRA_ARGS="--use_cucl"
    ;;
  cucl_class_based)
    BEST_EXTRA_ARGS="--use_cucl --cucl_class_based"
    ;;
  modulation_only|boundary_only|calibration_only)
    BEST_EXTRA_ARGS="--use_uncertainty_modulation"
    ;;
  baseline)
    BEST_EXTRA_ARGS=""
    ;;
  *)
    echo "Warning: unknown BEST_TAG=${BEST_TAG}; running best-model test without extra uncertainty flags"
    BEST_EXTRA_ARGS=""
    ;;
esac

BASELINE_DIR=${BASE_OUTPUT}/${BASE_NAME}_baseline_${EPOCH}epoch/${DATASET}/vit_upernet_uncer/fintune/${SEED}
BEST_DIR=${BASE_OUTPUT}/${BASE_NAME}_${BEST_TAG}_${EPOCH}epoch/${DATASET}/vit_upernet_uncer/fintune/${SEED}

STAMP=$(date +%Y%m%d_%H%M%S)
DEFAULT_REPORT_DIR=${BASE_OUTPUT}/uncertainty_reports/${BASE_NAME}_${BEST_TAG}_vs_baseline_${EPOCH}epoch_${STAMP}
REPORT_DIR=${REPORT_DIR:-${DEFAULT_REPORT_DIR}}
BASELINE_TEST_OUT=${REPORT_DIR}/baseline_test
BEST_TEST_OUT=${REPORT_DIR}/best_test
COMPARISON_OUT=${REPORT_DIR}/comparison
BASELINE_TEST_RUN_DIR=${BASELINE_TEST_OUT}/${DATASET}/vit_upernet_uncer/fintune/${SEED}
BEST_TEST_RUN_DIR=${BEST_TEST_OUT}/${DATASET}/vit_upernet_uncer/fintune/${SEED}

mkdir -p "${REPORT_DIR}" "${BASELINE_TEST_OUT}" "${BEST_TEST_OUT}" "${COMPARISON_OUT}"
mkdir -p "${BASELINE_TEST_RUN_DIR}" "${BEST_TEST_RUN_DIR}"

if [[ ! -f "${BASELINE_DIR}/best.pth.tar" ]]; then
  echo "Missing baseline checkpoint: ${BASELINE_DIR}/best.pth.tar"
  exit 1
fi

if [[ ! -f "${BEST_DIR}/best.pth.tar" ]]; then
  echo "Missing best checkpoint: ${BEST_DIR}/best.pth.tar"
  exit 1
fi

# main_segmentation.py test() loads from <output_dir>/<dataset>/<model>/<train>/<seed>/best.pth.tar,
# so place symlinks at the expected nested locations for evaluation outputs.
ln -sfn "${BASELINE_DIR}/best.pth.tar" "${BASELINE_TEST_RUN_DIR}/best.pth.tar"
ln -sfn "${BEST_DIR}/best.pth.tar" "${BEST_TEST_RUN_DIR}/best.pth.tar"

echo "Stage toggles: SKIP_BASELINE_TEST=${SKIP_BASELINE_TEST}, SKIP_BEST_TEST=${SKIP_BEST_TEST}, SKIP_COMPARISON=${SKIP_COMPARISON}, SKIP_REPORT=${SKIP_REPORT}"
echo "Using REPORT_DIR=${REPORT_DIR}"

if [[ "${SKIP_BASELINE_TEST}" == "1" ]]; then
  echo "[1/4] Skipped baseline inference"
else
  echo "[1/4] Baseline inference with uncertainty metrics"
  python -u main_segmentation.py \
    --batch_size 1 \
    --model vit_upernet_uncer \
    --output_dir "${BASELINE_TEST_OUT}" \
    --log_dir "${BASELINE_TEST_OUT}" \
    --train fintune \
    --case test \
    --state test \
    --resume "${BASELINE_DIR}/best.pth.tar" \
    --pretrain /SAN/medic/profound/output_mae/output_mae_wo_norm_r8000/checkpoint-${EPOCH}.pth \
    --use_mc_dropout \
    --mc_samples 5 \
    --uncertainty \
    --sliding_window \
    --epochs 100 \
    --dataset "${DATASET}" \
    --warmup_epochs 10 \
    --lr 1e-4 \
    --seed "${SEED}"
fi

if [[ "${SKIP_BEST_TEST}" == "1" ]]; then
  echo "[2/4] Skipped best-model inference"
else
  echo "[2/4] Best-model inference with uncertainty metrics"
  python -u main_segmentation.py \
    --batch_size 1 \
    --model vit_upernet_uncer \
    --output_dir "${BEST_TEST_OUT}" \
    --log_dir "${BEST_TEST_OUT}" \
    --train fintune \
    --case test \
    --state test \
    --resume "${BEST_DIR}/best.pth.tar" \
    --pretrain /SAN/medic/profound/output_mae/output_mae_wo_norm_r8000/checkpoint-${EPOCH}.pth \
    --use_mc_dropout \
    --mc_samples 5 \
    --uncertainty \
    --sliding_window \
    --epochs 100 \
    --dataset "${DATASET}" \
    --warmup_epochs 10 \
    --lr 1e-4 \
    --seed "${SEED}" \
    ${BEST_EXTRA_ARGS}
fi

if [[ "${SKIP_COMPARISON}" == "1" ]]; then
  echo "[3/4] Skipped boundary/uncertainty comparison"
else
  echo "[3/4] Boundary/uncertainty comparison"
  python -u compare_boundaries.py \
    --baseline_dir "${BASELINE_DIR}" \
    --uncertainty_dir "${BEST_DIR}" \
    --uncertainty_tag "${BEST_TAG}" \
    --output_dir "${COMPARISON_OUT}" \
    --pretrain /SAN/medic/profound/output_mae/output_mae_wo_norm_r8000/checkpoint-${EPOCH}.pth \
    --root /SAN/medic/foundation/downstream_data \
    --dataset "${DATASET}" \
    --sliding_window \
    --lr 1e-4 \
    --mc_samples 5 \
    --threshold_hd95 0.5 \
    --threshold_sd 0.1 \
    --top_k 10
fi

if [[ "${SKIP_REPORT}" == "1" ]]; then
  echo "[4/4] Skipped markdown report generation"
else
  echo "[4/4] Generate markdown report"
  python -u script/segmentation/uncertainty_project/generate_uncertainty_report.py \
    --csv_path "${COMPARISON_OUT}/sample_metrics.csv" \
    --output_md "${REPORT_DIR}/report.md" \
    --threshold_hd95 0.5 \
    --threshold_sd 0.1 \
    --threshold_dice 0.0 \
    --hard_split_percentile "${HARD_SPLIT_PERCENTILE}" \
    --baseline_name "baseline" \
    --best_name "${BEST_TAG}" \
    --top_k 10
fi

if [[ "${SKIP_STRUCTURE_REPORT}" == "1" ]]; then
  echo "[5/5] Skipped structure-wise (organ-wise) report"
else
  echo "[5/5] Generate structure-wise (organ-wise) report"
  python -u script/segmentation/uncertainty_project/structure_wise_report.py \
    --csv_path "${COMPARISON_OUT}/sample_metrics.csv" \
    --output_csv "${REPORT_DIR}/structure_wise_stats.csv" \
    --output_md "${REPORT_DIR}/structure_wise_report.md" \
    --baseline_name "baseline" \
    --method_name "${BEST_TAG}" \
    --alpha 0.05
fi

echo "Pipeline complete. Artifacts in: ${REPORT_DIR}"
