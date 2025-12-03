#!/bin/bash

# Optional: pick which GPU
# export CUDA_VISIBLE_DEVICES=0

# NCCL envs are mostly harmless even for 1 GPU
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ASYNC_ERROR_HANDLING=1

# Activate your environment
# source activate vila-u
# or
# source ~/miniconda3/bin/activate vila-u

# Global batch size and gradient accumulation
global_bs=${BATCH_SIZE:-1}        # choose something that fits in 40GB
acc_step=${ACC_STEP:-1}
bs=$((global_bs / acc_step))

echo "Using per_device_train_batch_size = $bs"
echo "Gradient accumulation steps       = $acc_step"

python vila_u/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path vila-u-7b-256 \
    --version v1 \
    --data_mixture rt_1 \
    --chunk_sampler True \
    --mm_projector mlp2x_gelu \
    --tune_mm_projector False \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --mm_use_vi_start_end True \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir ./checkpoints/vila-u-sft \
    --num_train_epochs 1 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $acc_step \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to none
