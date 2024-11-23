import argparse
import pandas as pd
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.GraphUtils import GraphUtils

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="PC算法运行脚本")
    parser.add_argument('--data_file', required=True, help="数据文件路径")
    parser.add_argument('--output_file', required=True, help="输出结果的保存路径")
    print(9,flush=True)
    # 解析参数
    args = parser.parse_args()
    data_file = args.data_file
    output_file = args.output_file

    # 加载数据文件
    print("加载数据文件...")
    data_df = pd.read_csv(data_file)
    variable_names = data_df.columns.tolist()  # 获取列名作为变量名
    data = data_df.values  # 转为 NumPy 数组
    print(f"数据维度: {data.shape}")
    print(f"变量名称: {variable_names}")

    # 初始化背景知识
    bck = BackgroundKnowledge()

    # 运行 PC 算法
    print("开始运行PC算法...")
    pc_result = pc(data=data, fisherz=fisherz, alpha=0.05, background_knowledge=bck, node_names=variable_names)

    # 提取因果关系边的信息
    print("提取因果关系边的信息...")
    edges = []
    for edge in pc_result.G.get_graph_edges():
        edges.append({
            'Node1': str(edge.node1),
            'Node2': str(edge.node2),
            'Endpoint1': edge.endpoint1,
            'Endpoint2': edge.get_endpoint2()
        })

    # 将边信息保存到指定的 output_file
    print("保存因果关系边信息中...")
    edge_df = pd.DataFrame(edges)
    edge_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"因果关系边信息已保存到 {output_file}")

if __name__ == "__main__":
    main()
