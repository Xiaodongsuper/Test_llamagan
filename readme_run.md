Following Next Steps to run the code
### 1.obtain the running environment using the command: docker pull xiaodongsysu/llamagen:latest

### 2. Prepare dataset https://github.com/FoundationVision/LlamaGen?tab=readme-ov-file and put it in the folder /llamagen/Datasets

### 3. Prepare pretrained model （https://drive.google.com/file/d/1W3XTKUzKpFCYKBmiHh0VTRlfTR5utDT8/view?usp=sharing） and put it in the folder /llamagen/pretrained_models/vae/

### 4. sudo docker run --gpus all -it --shm-size=8g  -v /home/dongxiao/LlamaGen:/llamagen xiaodongsysu/llamagen:latest

### 5. cd /llamagen

### 6. bash run_2.sh

### 5. 
