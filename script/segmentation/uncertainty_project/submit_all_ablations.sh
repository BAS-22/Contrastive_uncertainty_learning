#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

SCRIPT_PATH="script/segmentation/uncertainty_project/profound_vit_upernet_ablation.sh"
MODES=(baseline modulation boundary calibration cucl cucl_no_calibration full class_based classic_boundary)
SEQUENTIAL=0

usage() {
  cat <<'EOF'
Usage:
  bash script/segmentation/uncertainty_project/submit_all_ablations.sh [--sequential]

Options:
  --sequential   Submit each ablation job with qsub -hold_jid dependency
                 so jobs run one after another in this order:
                 baseline -> modulation -> boundary -> calibration -> cucl -> cucl_no_calibration -> full -> class_based -> classic_boundary

Default behavior:
  Submits all jobs immediately (independent scheduling).
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--sequential" ]]; then
  SEQUENTIAL=1
elif [[ -n "${1:-}" ]]; then
  echo "Error: unknown option '${1}'"
  usage
  exit 1
fi

if ! command -v qsub >/dev/null 2>&1; then
  echo "Error: qsub not found in PATH"
  exit 1
fi

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Error: missing ablation script at $SCRIPT_PATH"
  exit 1
fi

extract_job_id() {
  local qsub_output="$1"
  # Typical SGE output: "Your job 123456 (\"name\") has been submitted"
  echo "$qsub_output" | sed -n 's/.*Your job \([0-9][0-9]*\).*/\1/p' | head -n1
}

last_job_id=""
echo "Submitting ablation jobs using: $SCRIPT_PATH"

for mode in "${MODES[@]}"; do
  if [[ "$SEQUENTIAL" -eq 1 && -n "$last_job_id" ]]; then
    submit_out=$(qsub -hold_jid "$last_job_id" "$SCRIPT_PATH" "$mode")
  else
    submit_out=$(qsub "$SCRIPT_PATH" "$mode")
  fi

  job_id=$(extract_job_id "$submit_out")

  if [[ -z "$job_id" ]]; then
    echo "Warning: could not parse job ID for mode=$mode"
    echo "qsub output: $submit_out"
  else
    echo "Submitted mode=$mode job_id=$job_id"
    last_job_id="$job_id"
  fi
done

if [[ "$SEQUENTIAL" -eq 1 ]]; then
  echo "All ablation jobs submitted with sequential dependencies."
else
  echo "All ablation jobs submitted (independent scheduling)."
fi
