#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

case "${1:-}" in
    echo) entrypoint=train.py; default_root="$REPO_ROOT/training_results"; extra_args=() ;;
    hybrid) entrypoint=train_think.py; default_root="$REPO_ROOT/training_results_hybrid"
        extra_args=(--compress_ratio "${COMPRESS_RATIO:-0.5}") ;;
    *) die "Expected mode: echo or hybrid" ;;
esac

DATASET_PATH="$(absolute_path "${DATASET_PATH:-$REPO_ROOT/data/LongAlpaca-1k.json}")"
TRAINING_ROOT="$(absolute_path "${TRAINING_ROOT:-$default_root}")"
[[ -f "$DATASET_PATH" ]] || die "Training data not found: $DATASET_PATH"
stage1_dir="$TRAINING_ROOT/${CONFIG_NAME}_kvmse"
stage2_dir="$TRAINING_ROOT/${CONFIG_NAME}_kvmse_600_omse"
stage1_checkpoint="$stage1_dir/checkpoint-600.pt"

common_args=(
    --model_path "$MODEL_PATH"
    --block_size "$BLOCK_SIZE"
    --total_save_dim "$TOTAL_SAVE_DIM"
    --echo_input_mode joint
    --dataset_path "$DATASET_PATH"
    --lr 5e-4
    --batch_size 1
    --save_per_step 100
)

mkdir -p -- "$stage1_dir"
"$PYTHON_BIN" -u "$entrypoint" "${common_args[@]}" "${extra_args[@]}" \
    --num_steps 600 --scheduler_steps 1000 --loss_type kv_mse --output_dir "$stage1_dir" \
    2>&1 | tee "$stage1_dir/log.txt"

[[ -f "$stage1_checkpoint" ]] || die "Stage 1 did not produce $stage1_checkpoint"
mkdir -p -- "$stage2_dir"
"$PYTHON_BIN" -u "$entrypoint" "${common_args[@]}" "${extra_args[@]}" \
    --num_steps 1000 --scheduler_steps 2000 --loss_type o_mse --mlp_path "$stage1_checkpoint" \
    --output_dir "$stage2_dir" 2>&1 | tee "$stage2_dir/log.txt"
