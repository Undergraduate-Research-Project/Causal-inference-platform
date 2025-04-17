import numpy as np
import pandas as pd
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from gies.scores.gauss_int_l0_pen import GaussIntL0Pen
from gies import fit_bic

# 加载数据
data_file = "uploads/20241031092509_dataset2_2.csv"  # 替换为你的数据文件路径
data_df = pd.read_csv(data_file)

# 将数据转为 NumPy 格式
data = [data_df.values]

# 设置干预信息（如果无干预数据，则传入空列表）
interventions = [[]]

# 初始化背景知识（如果无特殊限制，则可以省略或使用空背景知识）
bck = BackgroundKnowledge()

# 运行 GIES 算法
adjacency_matrix, total_score = fit_bic(
    data,
    interventions,
)

# 打印结果
print("最终的邻接矩阵:")
print(adjacency_matrix)
print("总得分:", total_score)

# 提取因果关系边信息
print("因果关系边:")
for i in range(adjacency_matrix.shape[0]):
    for j in range(adjacency_matrix.shape[1]):
        if adjacency_matrix[i][j] == 1:  # 表示 i -> j 的因果关系
            print(f"{data_df.columns[i]} -> {data_df.columns[j]}")
