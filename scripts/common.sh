#!/usr/bin/env bash
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
INVOCATION_DIR="$PWD"

absolute_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$INVOCATION_DIR" "$1" ;;
    esac
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

: "${MODEL_PATH:?Set MODEL_PATH to a local model directory or Hugging Face model ID.}"
MODEL_PATH="${MODEL_PATH%/}"
if [[ -d "$MODEL_PATH" ]]; then
    MODEL_PATH="$(absolute_path "$MODEL_PATH")"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ "$PYTHON_BIN" == */* ]]; then
    PYTHON_BIN="$(absolute_path "$PYTHON_BIN")"
fi
command -v "$PYTHON_BIN" >/dev/null || die "Python executable not found: $PYTHON_BIN"

BLOCK_SIZE="${BLOCK_SIZE:-2}"
TOTAL_SAVE_DIM="${TOTAL_SAVE_DIM:-0}"
CONFIG_NAME="block${BLOCK_SIZE}_save${TOTAL_SAVE_DIM}"
cd -- "$REPO_ROOT"
