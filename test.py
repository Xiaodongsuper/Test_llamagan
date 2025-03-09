import numpy as np

# 读取 .npy 文件
file_path = "/home/dongxiao/LlamaGen/cub200_code_c2i_flip_ten_crop/cub200384_labels/1000.npy"
data = np.load(file_path)
print(data)
# 打印数据的基本信息
print("数据形状:", data.shape)
print("数据类型:", data.dtype)
print("数据内容:", data)

# 如果数据量很大，你可能只想看前几个元素
print("\n前几个元素:", data.flatten()[:10])