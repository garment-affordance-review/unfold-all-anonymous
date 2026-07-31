#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-python}"
CFG=${1:-${ROOT}/experiments/pair_policy/configs/train/monai_unet.yaml}

OUT_DIR=$("${PYTHON}" - <<PY
from pathlib import Path
import yaml
cfg = yaml.full_load(Path("${CFG}").read_text(encoding="utf-8"))
print(Path(cfg["train"]["output_dir"]).resolve())
PY
)

NPROC=$("${PYTHON}" - <<PY
from pathlib import Path
import yaml
cfg = yaml.full_load(Path("${CFG}").read_text(encoding="utf-8"))
print(int(((cfg.get("train", {}) or {}).get("distributed", {}) or {}).get("nproc_per_node", 1)))
PY
)

mkdir -p "${OUT_DIR}"
LOG_PATH="${OUT_DIR}/console.log"

exec > >(tee "${LOG_PATH}") 2>&1

echo "[INFO] Started at $(date '+%Y-%m-%d %H:%M:%S')"
echo "[INFO] ROOT=${ROOT}"
echo "[INFO] PYTHON=${PYTHON}"
echo "[INFO] CFG=${CFG}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] NPROC=${NPROC}"

cd "${ROOT}"
TORCH_LIB_DIR="$("${PYTHON}" - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec("torch")
if spec is None or spec.origin is None:
    raise SystemExit(1)
print((Path(spec.origin).resolve().parent / "lib").resolve())
PY
)"

env \
  PYTHONPATH="${ROOT}" \
  HDF5_USE_FILE_LOCKING=FALSE \
  LD_LIBRARY_PATH="${TORCH_LIB_DIR}:${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}" \
  "${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" \
  "${ROOT}/apps/train_pair_policy.py" --config "${CFG}"

status=$?
echo "[INFO] Finished at $(date '+%Y-%m-%d %H:%M:%S') with status=${status}"
exit ${status}
