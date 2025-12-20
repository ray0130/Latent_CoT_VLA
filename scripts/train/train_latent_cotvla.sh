#!/bin/bash


export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ASYNC_ERROR_HANDLING=1


# Global batch size and gradient accumulation
global_bs=${BATCH_SIZE:-128}
acc_step=${ACC_STEP:-16}
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
    --tune_mm_projector True \
    --tune_language_model False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --mm_use_vi_start_end True \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --output_dir ./checkpoints/latent_cotvla_model_example \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size $bs \
    --per_device_eval_batch_size $bs \
    --gradient_accumulation_steps $acc_step \
    --evaluation_strategy "steps" \
    --eval_steps 140 \
    --save_strategy "steps" \
    --save_steps 140 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.07 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --vflan_no_system_prompt True \
    --report_to wandb
