#!/bin/bash

module use --append /share/public/apps/modulefiles
module load cuda/12.6.0

. /share/home/syji/miniconda3/etc/profile.d/conda.sh
conda activate echo

export CUDA_VISIBLE_DEVICES=0


MODEL_PATH='/your/Meta-Llama-3.1-8B-Instruct'


attn_implementation=flash_attention_2 


MODEL_NAME=$(basename $MODEL_PATH)
MLP_NAME=$(basename $(dirname $MLP_PATH))


METHOD='echo' # base
echo_input_mode='joint' #default
compress_ratio=0.5 # for think to compress key
MLP_PATH='./training_results_hybrid/block2_save0_kvmse_600_omse/checkpoint-1000.pt'

SAVE_DIR=./longbench_results_hybrid/block2_save0_kvmse_600_omse/

datasets="narrativeqa"

echo "Starting Longbench Inference..."
echo "Dataset: ${datasets}"



python -u inference_longbench.py \
    --method "${METHOD}" \
    --model_path "${MODEL_PATH}" \
    --attn_implementation "${attn_implementation}" \
    --sink_size 4 \
    --local_window_size 128 \
    --block_size 2 \
    --total_save_dim 0 \
    --echo_input_mode "${echo_input_mode}" \
    --compress_ratio "${compress_ratio}" \
    --mlp_path "${MLP_PATH}" \
    --datasets ${datasets} \
    --save_dir "${SAVE_DIR}"


echo "Starting Evaluation..."

conda activate kv

MODEL_NAME=${MODEL_NAME,,}

eval_path="${SAVE_DIR}${MODEL_NAME}"
python3 eval.py --results_dir ${eval_path}
echo "eval_path: $eval_path"

echo "All Done."