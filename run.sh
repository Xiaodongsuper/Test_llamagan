# 切换到项目根目录
cd "$(dirname "$0")/../.."  
PROJECT_ROOT=/home/dongxiao/LlamaGen # Modify the project root path
cd $PROJECT_ROOT
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/dongxiao/LlamaGen/env # Modify the conda environment path

export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

echo "Current directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"
# python /home/dongxiao/LlamaGen/tokenizer/tokenizer_image/vq_demo.py


# Here, we do not retrain the VQ-VAE model, we use the pre-trained model vq_ds16_c2i
# Modify the model path and dataset path in your own path
bash ./scripts/autoregressive/extract_codes_c2i.sh --vq-ckpt ./pretrained_models/vae/vq_ds16_c2i.pt --data-path ./Datasets/CUB_200_2011_dataset/train --code-path ${PROJECT_ROOT}/cub200_code_c2i_flip_ten_crop1 --ten-crop --crop-range 1.1 --image-size 384

# 300 epeoch needs 4 hours in single L40s gpu
bash scripts/autoregressive/train_c2i_fsdp.sh \
--cloud_save_path "${PROJECT_ROOT}/saved_model_cub200_single_gpu/" \
--code_path "${PROJECT_ROOT}/cub200_code_c2i_flip_ten_crop/" \
--image_size 384 \
--gpt_model GPT-B \
--num_classes 200 \
--epochs 300

# bash ./scripts/autoregressive/extract_codes_c2i.sh --vq-ckpt ./pretrained_models/vae/vq_ds16_c2i.pt --data-path /home/dongxiao/Datasets/cub200/train --code-path /home/dongxiao/LlamaGen/cub200_code_c2i_flip_ten_crop --ten-crop --crop-range 1.1 --image-size 384

# Modify the model path in sample_c2i_class_each_model.sh according to cloud_save_path
bash scripts/autoregressive/sample_c2i_class_each_model.sh 
# bash scripts/tokenizer/train_vq.sh --cloud-save-path /home/dongxiao/LlamaGen/saved_model --data-path /home/dongxiao/Datasets/cifar10/images --image-size 256 --vq-model VQ-16

# bash ./scripts/autoregressive/extract_codes_c2i.sh --vq-ckpt ./pretrained_models/vae/vq_ds16_c2i.pt --data-path /home/dongxiao/Datasets/cifar10/images --code-path /home/dongxiao/LlamaGen/cifar10_code_c2i_flip_ten_crop --ten-crop --crop-range 1.1 --image-size 384