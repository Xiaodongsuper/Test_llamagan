# !/bin/bash
set -x

CUDA_VISIBLE_DEVICES=0 torchrun \
--nnodes=1 --nproc_per_node=1 --node_rank=0 \
--master_port=12345 \
autoregressive/sample/sample_c2i_ddp.py \
--vq_ckpt ./pretrained_models/vae/vq_ds16_c2i.pt \
--gpt_ckpt "/home/dongxiao/LlamaGen/saved_model_cub200_single_gpu/20241212140105-GPT-B/0018000/consolidated.pth" \
--from_fsdp
"$@"
