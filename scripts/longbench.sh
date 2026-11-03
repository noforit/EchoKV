#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh"

case "${1:-}" in
    echo) entrypoint=inference_longbench.py
        default_training_root="$REPO_ROOT/training_results"
        default_save_root="$REPO_ROOT/longbench_results"; extra_args=() ;;
    hybrid) entrypoint=inference_longbench_think.py
        default_training_root="$REPO_ROOT/training_results_hybrid"
        default_save_root="$REPO_ROOT/longbench_results_hybrid"
        extra_args=(--compress_ratio "${COMPRESS_RATIO:-0.5}") ;;
    *) die "Expected mode: echo or hybrid" ;;
esac

METHOD="${METHOD:-echo}"
[[ "$METHOD" == echo || "$METHOD" == base ]] || die "METHOD must be echo or base"
TRAINING_ROOT="$(absolute_path "${TRAINING_ROOT:-$default_training_root}")"
MLP_PATH="$(absolute_path "${MLP_PATH:-$TRAINING_ROOT/${CONFIG_NAME}_kvmse_600_omse/checkpoint-1000.pt}")"
SAVE_DIR="$(absolute_path "${SAVE_DIR:-$default_save_root/${CONFIG_NAME}_kvmse_600_omse}")"
DATA_DIR="$(absolute_path "${DATA_DIR:-$REPO_ROOT/data/LongBench}")"
task_names="${DATASETS:-all}"
if [[ "$task_names" == all ]]; then
    task_names="narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique gov_report qmsum multi_news trec triviaqa samsum passage_count passage_retrieval_en lcc repobench-p"
fi
read -r -a datasets <<< "$task_names"
[[ ${#datasets[@]} -gt 0 ]] || die "DATASETS must contain at least one task"
for dataset in "${datasets[@]}"; do
    [[ -f "$DATA_DIR/$dataset.jsonl" ]] || die "LongBench data not found: $DATA_DIR/$dataset.jsonl"
done
checkpoint_args=()
if [[ "$METHOD" == echo ]]; then
    [[ -f "$MLP_PATH" ]] || die "Predictor checkpoint not found: $MLP_PATH"
    checkpoint_args=(--mlp_path "$MLP_PATH")
fi

"$PYTHON_BIN" -u "$entrypoint" \
    --method "$METHOD" --model_path "$MODEL_PATH" \
    --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
    --sink_size 4 --local_window_size 128 \
    --block_size "$BLOCK_SIZE" --total_save_dim "$TOTAL_SAVE_DIM" \
    --echo_input_mode joint "${extra_args[@]}" "${checkpoint_args[@]}" \
    --data_dir "$DATA_DIR" --datasets "${datasets[@]}" --save_dir "$SAVE_DIR"

model_name="${MODEL_PATH##*/}"
"$PYTHON_BIN" eval.py --results_dir "$SAVE_DIR/${model_name,,}"
