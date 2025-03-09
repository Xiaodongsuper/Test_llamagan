# !/bin/bash
set -x
export PYTHONPATH=$PYTHONPATH:$(pwd)
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
--nnodes=1 --nproc_per_node=4 --node_rank=0 \
--master_addr=localhost --master_port=12345 \
tokenizer/tokenizer_image/vq_train.py "$@"