#!/bin/bash
# Standalone torchrun launcher for SenseNova-U1 30B-A3B (MoE + MoT image-gen).
#
# Single-node (8 GPUs):
#   bash shell/train_u1/A3B.sh
#
# Multi-node — run on each node, replacing NODE_RANK / MASTER_ADDR:
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash shell/train_u1/A3B.sh
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash shell/train_u1/A3B.sh

set -e
cd "$(dirname "$0")/../.."  # repo root

# ============================ Distributed (torchrun) ============================ #
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}

# ============================ Model & data (placeholders — fill in!) ============================ #
export CONFIG_NAME="configs/sensenovavl_qwen3_gen/qwen3_base_30a3b_48layer_isp_mot_pt.py"
export MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"/path/to/SenseNova-U1-A3B-SFT"}
export VOCAB_FILE=${VOCAB_FILE:-"/path/to/qwen3/tokenizer"}
export TOKENIZER_PATH=${TOKENIZER_PATH:-"/path/to/qwen3/tokenizer"}
export mm_data_path=${mm_data_path:-"data/sample/sample_data_meta.json"}
export load_optimizer=${load_optimizer:-"model"}

# resume (uncomment to enable)
# export load_optimizer="all"
# export auto_resume=true
# export resume_ds=true

# ============================ Parallelism ============================ #
export zero1_size=-1
export wp_size=8
export tp_size=2
export pp_size=1

# ============================ Optimization ============================ #
export SEED=42
export lr=2e-4
export lr_scheduler_type="constant"
export min_lr_ratio=0.5
export mlp_lr_scale=1.0
export weight_decay=0
export grad_accm=1
export total_steps=200000
export init_steps=2000

# ============================ MoE ============================ #
export balance_moe_loss_coef=0.005
export moe_balance_image_ratio=1.0
export moe_balance_text_ratio=1.0

# ============================ Data / sequence ============================ #
export num_imgs=144
export seq_len=28672
export max_sample_tokens=28672
export dataset_replacement=true
export min_num_frame=1
export max_num_frame=128
export dynamic_image_version="native_resolution"
export CONV_STYLE="sensenovalm2-chat-v3"
export down_sample_ratio=0.5
export max_pixels=$((512 * 512))
export min_pixels=$((256 * 256))
export max_pixels_gen=$((512 * 512))
export min_pixels_gen=$((256 * 256))
export LLM_DATA_WEIGHTS=0
export MM_CC_DATA_WEIGHTS=0

# ============================ Freeze / trainable modules ============================ #
export freeze_llm=false
export freeze_backbone=false
export unfreeze_mot_gen=true
export mot_random_init=false  # SenseNova-U1-A3B-SFT already has the MoT generation branch

# ============================ Generation / diffusion ============================ #
export time_schedule="standard"
export time_shift_type="exponential"
export time_base_dist="logit_normal"
export base_shift=0.5
export max_shift=1.15
export base_image_seq_len=64
export max_image_seq_len=4096
export noise_scale_mode="resolution"
export noise_scale_base_image_seq_len=64
export add_noise_scale_embedding=true
export noise_scale_max_value=8
export P_mean=-0.8
export P_std=0.8
export cfg_txt_uncond_drop_prob=0.1
export cfg_img_uncond_drop_prob=0
export cfg_txtimg_uncond_drop_prob=0.1
export cfg_is_uncond_drop_independent='false'
export ema_decay=0.9999
export thinking_method="tag"

# ============================ Understanding ============================ #
export pad_dummy_image_gen='true'
export ce_loss_weight=0.1
export enable_und_loss='true'

# ============================ Job / logging ============================ #
export JOB_NAME=${JOB_NAME:-"sensenovau1_a3b_smoke_test"}
# export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
# export WANDB_PROJECT="neo_unify"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# ============================ Launch ============================ #
torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train_sensenovau1.py \
        --config "${CONFIG_NAME}" \
        --launcher torch \
        --seed "${SEED}" \
        --backend nccl
