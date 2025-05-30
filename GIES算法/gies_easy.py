import argparse
import csv
import pandas as pd
from gies import fit_bic
import json
import numpy as np

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="GIES算法运行脚本")
    parser.add_argument('--data_file', required=True, help="数据文件路径")
    parser.add_argument('--output_file', required=True, help="输出结果的保存路径")
    parser.add_argument('--background_edge', required=True, help="666")
    parser.add_argument('--variable_names', required=True, help="包含变量名")

    # 解析参数
    args = parser.parse_args()
    output_file = args.output_file
    background_edge_json = args.background_edge
    variable_names = args.variable_names.split(',')

    print(background_edge_json)
    # 解析背景边信息
    background_edges = json.loads(background_edge_json)  # 将背景边的JSON字符串转化为Python对象

    # 构建邻接矩阵 A0
    A0 = np.zeros((len(variable_names), len(variable_names)))  # 初始化邻接矩阵
    for edge in background_edges:
        # 获取 'from' 和 'to' 字段对应的变量名
        i = variable_names.index(edge['from'])
        j = variable_names.index(edge['to'])
        A0[i, j] = 1  # 添加背景边，表示 i -> j

    print(A0)


    # 加载数据文件
    print("加载数据文件...")
    # 将逗号分隔的字符串转换为列表
    

    # 读取指定列的数据
    data_df = pd.read_csv(args.data_file, usecols=variable_names)
    variable_names = data_df.columns.tolist()  # 获取列名作为变量名
    data = [data_df.values]  # 转为 NumPy 数组（作为单环境数据传入）
    print(f"数据维度: {data_df.shape}")
    print(f"变量名称: {variable_names}")
    print("****************************")
    print(data)

    # 设置干预信息（如果没有干预，则传空列表）
    interventions = [[]]

    # 运行 GIES 算法
    print("开始运行GIES算法...")
    adjacency_matrix, total_score = fit_bic(
        data,
        interventions,
        A0 = A0 # 背景边传入
    )

    print("GIES算法运行完成！")
    print("总得分:", total_score)

    # 提取因果关系边的信息
    print("提取因果关系边的信息...")
    edges = []
    for i in range(adjacency_matrix.shape[0]):
        for j in range(adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] == 1:  # 表示 i -> j 的因果关系
                edges.append([variable_names[i], variable_names[j], "TAIL", "ARROW"])

    if not edges:
        print("未提取到任何因果关系边！")
    else:
        print(f"提取到的因果关系边数量: {len(edges)}")
        for edge in edges:
            print(f"{edge[0]} -> {edge[1]}")

    # 保存因果关系边到输出文件
    print("保存因果关系边信息中...")
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        csv_writer = csv.writer(csvfile)
        # 写入表头
        csv_writer.writerow(["Node1", "Node2", "Endpoint1", "Endpoint2"])
        # 写入边信息
        csv_writer.writerows(edges)
    print(f"因果关系已保存到文件: {output_file}")

if __name__ == "__main__":
    main()
