#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

CFG="${1:-experiments/pair_policy/configs/render/supervision.yaml}"

if [[ -n "${PYTHON:-}" ]]; then
  PY="${PYTHON}"
else
  PY=""
  for candidate in \
    "python" \
    "${CONDA_ENVS_ROOT}/pointcept/bin/python" \
    "${HOME}/miniconda3/envs/pointcept/bin/python"
  do
    if command -v "${candidate}" >/dev/null 2>&1 || [[ -x "${candidate}" ]]; then
      if "${candidate}" -c "import h5py" >/dev/null 2>&1; then
        PY="${candidate}"
        break
      fi
    fi
  done
  if [[ -z "${PY}" ]]; then
    echo "failed to find a Python interpreter with h5py; set PYTHON=/path/to/python" >&2
    exit 1
  fi
fi

OUT_DIR=$("${PY}" - <<PY
from pathlib import Path
import yaml
cfg = yaml.full_load(Path("${CFG}").read_text(encoding="utf-8"))
print(Path(cfg["output_dir"]).resolve())
PY
)
LOG_PATH="${OUT_DIR}/console.log"
PID_PATH="${OUT_DIR}/pid.txt"

RESUME="${RENDER_SUPERVISION_RESUME:-${RESUME:-0}}"

if [[ -f "$PID_PATH" ]]; then
  OLD_PID="$(cat "$PID_PATH" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && ps -p "$OLD_PID" > /dev/null 2>&1; then
    echo "render supervision already running: pid=$OLD_PID"
    echo "log=$LOG_PATH"
    exit 1
  fi
fi

if [[ "$RESUME" != "1" ]]; then
  rm -rf "$OUT_DIR"
fi
mkdir -p "$OUT_DIR"

echo "[INFO] Using Python: $PY"

ARGS=(--config "$CFG")
if [[ "$RESUME" == "1" ]]; then
  ARGS+=(--resume)
fi

nohup env PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1 "$PY" -u apps/build_render_supervision.py \
  "${ARGS[@]}" \
  > "$LOG_PATH" 2>&1 < /dev/null &

echo $! > "$PID_PATH"
echo "started supervision pid=$(cat "$PID_PATH")"
echo "log=$LOG_PATH"
