#!/bin/bash

set -euo pipefail

ROOT=${PROJECT_ROOT}
PYTHON=${CONDA_ENVS_ROOT}/isaac/bin/python
POINTCEPT_EXP=${1:-${TEACHER_EXP_ROOT}}
RUN_NAME=${2:-teacher_exp_extra_unseen_online}
MAX_ASSETS=${3:-20}
NUM_ENVS=${4:-64}
CANDIDATE_FPS_COUNT=${5:-256}
OUT_DIR=${ROOT}/experiments/offline_teacher_eval/runs/${RUN_NAME}

mkdir -p "${OUT_DIR}"
LOG_PATH="${OUT_DIR}/console.log"

exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[INFO] Started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] ROOT=${ROOT}"
echo "[INFO] PYTHON=${PYTHON}"
echo "[INFO] POINTCEPT_EXP=${POINTCEPT_EXP}"
echo "[INFO] RUN_NAME=${RUN_NAME}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] MAX_ASSETS=${MAX_ASSETS}"
echo "[INFO] NUM_ENVS=${NUM_ENVS}"
echo "[INFO] CANDIDATE_FPS_COUNT=${CANDIDATE_FPS_COUNT}"

unset DISPLAY
"${PYTHON}" "${ROOT}/tools/pair_transfer/evaluate_extra_unseen_pairs_online.py" \
  --headless \
  --config configs/offline_pair_conditioned.yaml \
  --num-envs "${NUM_ENVS}" \
  --exp-dir "${POINTCEPT_EXP}" \
  --pointcept-root ${POINTCEPT_ROOT} \
  --data-root ${PROJECT_ROOT}/data/clothes \
  --out-dir "${OUT_DIR}" \
  --asset-order shuffle \
  --candidate-fps-count "${CANDIDATE_FPS_COUNT}" \
  --distance-bins 4 \
  --predicted-high-per-distance-bin 4 \
  --score-quantile-bins 4 \
  --quantile-per-cell 3 \
  --top-k 16 \
  --max-assets "${MAX_ASSETS}"

status=$?
echo "[INFO] Finished at $(date '+%Y-%m-%d %H:%M:%S') with status=${status}"
exit ${status}
