import argparse
import pandas as pd
import numpy as np
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.graph.GraphNode import GraphNode

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="PC算法运行脚本")
    parser.add_argument('--data_file', required=True, help="数据文件路径")
    parser.add_argument('--output_file', required=True, help="输出结果的保存路径")
    parser.add_argument('--background_edge', required=True, help="666")
    parser.add_argument('--variable_names', required=True, help="包含变量名")
    print(9,flush=True)
    # 解析参数
    args = parser.parse_args()
    output_file = args.output_file
    background_edge_json = args.background_edge
    print(background_edge_json)
    # 加载数据文件
    print("加载数据文件...")
    # 将逗号分隔的字符串转换为列表
    variable_names = args.variable_names.split(',')

    # 读取指定列的数据
    data_df = pd.read_csv(args.data_file, usecols=variable_names)

    # 打印读取的数据
    print(data_df)
    variable_names = data_df.columns.tolist()  # 获取列名作为变量名
    data = data_df.values  # 转为 NumPy 数组
    print(f"数据维度: {data.shape}")
    print(f"变量名称: {variable_names}")

    # 初始化背景知识
    bck = BackgroundKnowledge()
    bck.add_required_by_node(GraphNode('Diameter'), GraphNode('Viscera weight'))
    if background_edge_json:
        background_edge = eval(background_edge_json)
        print(f"接收到的 background_edge: {background_edge}")
        for edge in background_edge:
            node1 = GraphNode(str(edge['from']))
            node2 = GraphNode(str(edge['to']))
            print(node1, node2)
            bck.add_required_by_node(node1, node2)
            assert bck.required_rules_specs.__contains__((node1, node2))
            assert bck.is_required(node1, node2)

    print("bck is ",bck)

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
