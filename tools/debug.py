import numpy as np

# 给定.npy文件路径
file_path = "/u/home/caoh/datasets/SemanticKITTI/dataset/pred/CGFormer/00/000030.npy"

# 加载.npy文件
data = np.load(file_path)

# 打印数据类型
print(data.dtype)


# python /u/home/caoh/projects/MA_Jiachen/3DPNA/tools/debug.py