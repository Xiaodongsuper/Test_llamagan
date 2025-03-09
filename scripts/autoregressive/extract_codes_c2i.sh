# !/bin/bash
set -x

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
--nnodes=1 --nproc_per_node=4 --node_rank=0 \
--master_port=12335 \
autoregressive/train/extract_codes_c2i.py "$@"
