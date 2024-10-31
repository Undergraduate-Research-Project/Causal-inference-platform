from guidance import models, gen,user,assistant
import pandas as pd
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from simple_model_suggester import SimpleModelSuggester
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Endpoint import Endpoint
import os

def find_keys_with_substring(dictionary, substring):
    return [key for key in dictionary if substring in key]

bck = BackgroundKnowledge()
modeler = SimpleModelSuggester()

# 读取CSV文件
df = pd.read_csv("C:\\Users\\32503\\Desktop\\gies+llm与pc+llm\\variable_dataset2.csv")
variables = df['detail'].tolist()
dic = dict(zip(df['name'],df['detail']))

bck = BackgroundKnowledge()

from causallearn.search.ConstraintBased.PC import pc

# 读取另一个CSV数据集
df1 = pd.read_csv("C:\\Users\\32503\\Desktop\\gies+llm与pc+llm\\dataset2_2.csv")
data = df1.to_numpy()

pos = 1
cnt = 0
cnt += 1

while pos:
    print("round:", cnt)
    cnt += 1
    pos = 0
    cg = pc(data=data, background_knowledge=bck, node_names=df['name'].tolist())
    for edge in cg.G.get_graph_edges():
        if bck.is_required(GraphNode(str(edge.node1)), GraphNode(str(edge.node2))) or bck.is_required(GraphNode(str(edge.node2)), GraphNode(str(edge.node1))) or (
                bck.is_forbidden(GraphNode(str(edge.node1)), GraphNode(str(edge.node2))) and bck.is_forbidden(GraphNode(str(edge.node2)), GraphNode(str(edge.node1)))):
            continue
        print(edge.node1, edge.node2, edge.endpoint1, edge.get_endpoint2())
        result = modeler.suggest_pairwise_relationship(dic[str(edge.node1)], dic[str(edge.node2)])
        print(result)
        if result[2].startswith("A"):
            if edge.endpoint1 != Endpoint.TAIL or edge.get_endpoint2() != Endpoint.ARROW:
                bck.add_required_by_node(GraphNode(str(edge.node1)), GraphNode(str(edge.node2)))
                bck.add_forbidden_by_node(GraphNode(str(edge.node2)), GraphNode(str(edge.node1)))
                print("A:", str(edge.node1), "->", str(edge.node2))
                pos = 1
        elif result[2].startswith("B"):
            if edge.endpoint1 != Endpoint.ARROW or edge.get_endpoint2() != Endpoint.TAIL:
                bck.add_required_by_node(GraphNode(str(edge.node2)), GraphNode(str(edge.node1)))
                bck.add_forbidden_by_node(GraphNode(str(edge.node1)), GraphNode(str(edge.node2)))
                print("B:", str(edge.node2), "->", str(edge.node1))
                pos = 1
        elif result[2].startswith("C"):
            bck.add_forbidden_by_node(GraphNode(str(edge.node1)), GraphNode(str(edge.node2)))
            bck.add_forbidden_by_node(GraphNode(str(edge.node2)), GraphNode(str(edge.node1)))
            print("C:", str(edge.node1), str(edge.node2))
            pos = 1

    for edge in cg.G.get_graph_edges():
        print(edge)


print()
print(6666666666)
print()

# 保存结果到文件
file_path = 'result2.csv'
print("Writing results to file...")

# 将因果图的边信息转换为字典列表
edges = []
for edge in cg.G.get_graph_edges():
    edges.append({
        'Node1': str(edge.node1),
        'Node2': str(edge.node2),
        'Endpoint1': edge.endpoint1,
        'Endpoint2': edge.get_endpoint2()
    })

# 将边信息保存为DataFrame
df_edges = pd.DataFrame(edges)
df_edges.to_csv(file_path, index=False, encoding='utf-8-sig')  # 指定UTF-8带BOM编码以避免乱码

# 验证文件是否成功写入
if os.path.exists(file_path):
    print(f"File '{file_path}' has been successfully created.")
    
    # 读取文件内容验证
    try:
        df_result = pd.read_csv(file_path, encoding='utf-8-sig')  # 使用同样的编码读取
        print("File content preview:")
        print(df_result.head())  # 显示前几行内容
    except Exception as e:
        print(f"Error reading file: {e}")
else:
    print(f"File '{file_path}' was not created.")