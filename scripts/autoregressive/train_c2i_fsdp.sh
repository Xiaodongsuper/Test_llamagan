# !/bin/bash
set -x

# torchrun \
# --nnodes=$nnodes --nproc_per_node=$nproc_per_node --node_rank=$node_rank \
# --master_addr=$master_addr --master_port=$master_port \
# autoregressive/train/train_c2i_fsdp.py "$@"

CUDA_VISIBLE_DEVICES=0 torchrun \
--nnodes=1 --nproc_per_node=1 --node_rank=0 \
--master_addr=localhost --master_port=12345 \
autoregressive/train/train_c2i_fsdp.py "$@"