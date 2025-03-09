FROM nvidia/cuda:12.1.0-base-ubuntu20.04

# 避免交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 安装基础包
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    curl \
    vim \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    cmake \
    pkg-config \
    libssl-dev \
    unzip \
    software-properties-common \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# 将 conda 添加到环境变量
ENV PATH=/opt/conda/bin:$PATH

# 创建指定 Python 3.9 版本的环境
RUN conda create -n env python=3.9 -y

# 设置工作目录
WORKDIR /llamagen

# 复制现有环境
COPY env /opt/conda/envs/env

# 修复环境路径
RUN cd /opt/conda/envs/env/bin && \
    find . -type f -exec sed -i -e 's|/home/dongxiao/LlamaGen/env|/opt/conda/envs/env|g' {} \;

# 初始化 conda
RUN conda init bash

# 重新安装 PyTorch
RUN /opt/conda/envs/env/bin/pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 复制并安装 requirements.txt
COPY requirements.txt /llamagen/
RUN /opt/conda/envs/env/bin/pip install -r /llamagen/requirements.txt

# 设置默认环境
ENV CONDA_DEFAULT_ENV=env
ENV PATH=/opt/conda/envs/env/bin:$PATH


docker commit 6431d316f0a2 llamagen:latest
CMD ["/bin/bash"]