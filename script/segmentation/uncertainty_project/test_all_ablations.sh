#$ -wd /home/bisidiqi/profound/uncer_module_logs/seg_job
#$ -l gpu=true,gpu_type=a6000|a40|a100|a100_80|a100_dgx
#$ -l tmem=32G
#$ -R y
#$ -l h_rt=10:0:0
#$ -N seg_ablation_test

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
MC_SAMPLES=${MC_SAMPLES:-5}

TAGS=${TAGS:-"baseline,modulation_only,boundary_only,calibration_only,cucl_only,cucl_no_calibration,full_um_cucl,cucl_class_based,classic_boundary_loss"}
EVAL_ROOT=${BASE_OUTPUT}/ablation_test_eval/${BASE_NAME}_${EPOCH}epoch/${DATASET}/seed${SEED}
SUMMARY_DIR=${EVAL_ROOT}/summary

mkdir -p "${SUMMARY_DIR}"

IFS=',' read -r -a TAG_ARRAY <<< "${TAGS}"

for tag in "${TAG_ARRAY[@]}"; do
	tag=$(echo "${tag}" | xargs)
	if [[ -z "${tag}" ]]; then
		continue
	fi

	TRAIN_DIR=${BASE_OUTPUT}/${BASE_NAME}_${tag}_${EPOCH}epoch/${DATASET}/vit_upernet_uncer/fintune/${SEED}
	EVAL_DIR=${EVAL_ROOT}/${tag}
	RUN_DIR=${EVAL_DIR}/${DATASET}/vit_upernet_uncer/fintune/${SEED}
	mkdir -p "${RUN_DIR}"

	if [[ ! -f "${TRAIN_DIR}/best.pth.tar" ]]; then
		echo "[SKIP] Missing checkpoint for ${tag}: ${TRAIN_DIR}/best.pth.tar"
		continue
	fi

	ln -sfn "${TRAIN_DIR}/best.pth.tar" "${RUN_DIR}/best.pth.tar"

	EXTRA_ARGS=""
	if [[ "${tag}" == "cucl_only" || "${tag}" == "cucl_no_calibration" || "${tag}" == "full_um_cucl" || "${tag}" == "cucl_class_based" ]]; then
		EXTRA_ARGS="--use_cucl"
	fi

	echo "[TEST] Running test for ${tag}"
	python -u main_segmentation.py \
		--batch_size 1 \
		--model vit_upernet_uncer \
		--output_dir "${EVAL_DIR}" \
		--log_dir "${EVAL_DIR}" \
		--train fintune \
		--case test \
		--state test \
		--pretrain /SAN/medic/profound/output_mae/output_mae_wo_norm_r8000/checkpoint-${EPOCH}.pth \
		--use_mc_dropout \
		--mc_samples "${MC_SAMPLES}" \
		--uncertainty \
		--sliding_window \
		--epochs 100 \
		--dataset "${DATASET}" \
		--warmup_epochs 10 \
		--lr 1e-4 \
		--seed "${SEED}" \
		${EXTRA_ARGS}
done

python -u script/segmentation/uncertainty_project/summarize_ablation_tests.py \
	--eval_root "${EVAL_ROOT}" \
	--tags "${TAGS}" \
	--output_csv "${SUMMARY_DIR}/ablation_test_summary.csv" \
	--output_md "${SUMMARY_DIR}/ablation_test_summary.md"

echo "Done. Summary in: ${SUMMARY_DIR}"
