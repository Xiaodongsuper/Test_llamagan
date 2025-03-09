#!/bin/bash
set -x

MODEL_DIR="/home/dongxiao/LlamaGen/saved_model_cub200_single_gpu/20241212140105-GPT-B"

# 遍历MODEL_DIR下的所有子目录
for checkpoint_dir in "$MODEL_DIR"/*/ ; do
    if [ -f "${checkpoint_dir}/consolidated.pth" ]; then
        echo "Processing checkpoint: ${checkpoint_dir}consolidated.pth"
        
        CUDA_VISIBLE_DEVICES=0 torchrun \
        --nnodes=1 --nproc_per_node=1 --node_rank=0 \
        --master_port=12345 \
        autoregressive/sample/sample_c2i.py \
        --vq_ckpt ./pretrained_models/vae/vq_ds16_c2i.pt \
        --gpt_ckpt "${checkpoint_dir}consolidated.pth" \
        --from_fsdp \
        --gpt_model GPT-B \
        --image_size 384 \
        --cfg_scale 1.2 \
        --from_fsdp
    fi
done

"$@"