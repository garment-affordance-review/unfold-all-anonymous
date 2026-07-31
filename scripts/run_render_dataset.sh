#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_OUTPUT="data/datasets/render_pair_policy"
DEFAULT_CONFIG="configs/render_replicator.yaml"
DEFAULT_PYTHON_CANDIDATES=(
  "${CONDA_ENVS_ROOT}/isaac/bin/python"
  "${CONDA_ENVS_ROOT}/env_isaaclab/bin/python"
)

ARGS=()
OUTPUT_PATH=""
HAS_OUTPUT=0
HAS_CONFIG=0
HAS_HEADLESS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output|--out)
      HAS_OUTPUT=1
      OUTPUT_PATH="$2"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    --output=*|--out=*)
      HAS_OUTPUT=1
      OUTPUT_PATH="${1#*=}"
      ARGS+=("$1")
      shift
      ;;
    --config)
      HAS_CONFIG=1
      ARGS+=("$1" "$2")
      shift 2
      ;;
    --config=*)
      HAS_CONFIG=1
      ARGS+=("$1")
      shift
      ;;
    --headless)
      HAS_HEADLESS=1
      ARGS+=("$1")
      shift
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ $HAS_OUTPUT -eq 0 ]]; then
  OUTPUT_PATH="$DEFAULT_OUTPUT"
  ARGS+=("--output" "$OUTPUT_PATH")
fi

if [[ $HAS_CONFIG -eq 0 ]]; then
  ARGS+=("--config" "$DEFAULT_CONFIG")
fi

if [[ $HAS_HEADLESS -eq 0 ]]; then
  ARGS+=("--headless")
fi

if [[ -z "${PYTHON:-}" ]]; then
  for candidate in "${DEFAULT_PYTHON_CANDIDATES[@]}"; do
    if [[ -x "$candidate" ]]; then
      export PYTHON="$candidate"
      break
    fi
  done
fi

if [[ -z "${PYTHON:-}" || ! -x "$PYTHON" ]]; then
  echo "Failed to resolve PYTHON interpreter. Export PYTHON first." >&2
  exit 1
fi

unset DISPLAY

mkdir -p "$ROOT_DIR/$(dirname "$OUTPUT_PATH")"
LOG_PATH="${OUTPUT_PATH%/}_run.log"

echo "[render] repo=$ROOT_DIR"
echo "[render] python=$PYTHON"
echo "[render] output=$OUTPUT_PATH"
echo "[render] log=$LOG_PATH"

cd "$ROOT_DIR"
"$PYTHON" apps/render_dataset.py "${ARGS[@]}" 2>&1 | tee "$LOG_PATH"
