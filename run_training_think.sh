#!/bin/bash

module use --append /share/public/apps/modulefiles
module load cuda/12.6.0

. /share/home/syji/miniconda3/etc/profile.d/conda.sh
conda activate echo



export CUDA_VISIBLE_DEVICES=0

MODEL_PATH='/your/Meta-Llama-3.1-8B-Instruct'



DATASET_PATH='./data/alpaca-1k.json' # training with alpaca only need 1 min!
echo_input_mode='joint'  # default
compress_ratio=0.5 # for think to compress key

# Stage 1
SAVE_DIR='./training_results_hybrid/block2_save0_kvmse'
loss_type='kv_mse' 



mkdir -p $SAVE_DIR

python -u train_think.py \
    --model_path "${MODEL_PATH}" \
    --block_size 2 \
    --total_save_dim 0 \
    --echo_input_mode "${echo_input_mode}" \
    --compress_ratio "${compress_ratio}" \
    --dataset_path "${DATASET_PATH}" \
    --lr 5e-4 \
    --num_steps 600 \
    --batch_size 1 \
    --output_dir "${SAVE_DIR}" \
    --save_per_step 100 \
    --mlp_path "${mlp_path}" \
    --loss_type "${loss_type}" \
    2>&1 | tee "${SAVE_DIR}/log.txt"


# Stage 2
SAVE_DIR='./training_results_hybrid/block2_save0_kvmse_600_omse'
loss_type='kv_mse' 
mlp_path='./training_results_hybrid/block2_save0_kvmse/checkpoint-100.pt' # stage2 need


mkdir -p $SAVE_DIR

python -u train_think.py \
    --model_path "${MODEL_PATH}" \
    --block_size 2 \
    --total_save_dim 0 \
    --echo_input_mode "${echo_input_mode}" \
    --compress_ratio "${compress_ratio}" \
    --dataset_path "${DATASET_PATH}" \
    --lr 5e-4 \
    --num_steps 1000 \
    --batch_size 1 \
    --output_dir "${SAVE_DIR}" \
    --save_per_step 100 \
    --mlp_path "${mlp_path}" \
    --loss_type "${loss_type}" \
    2>&1 | tee "${SAVE_DIR}/log.txt"
