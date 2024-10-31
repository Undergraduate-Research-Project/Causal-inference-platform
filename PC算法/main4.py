import pandas as pd
import re
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Endpoint import Endpoint
from causallearn.search.ConstraintBased.PC import pc
from simple_model_suggester import SimpleModelSuggester
import argparse
import os

def causal_analysis(variable_file_path, data_file_path, output_file_path):
    def find_keys_with_substring(dictionary, substring):
        return [key for key in dictionary if substring in key]

    # 创建 BackgroundKnowledge 实例和模型建议器
    bck = BackgroundKnowledge()
    modeler = SimpleModelSuggester()

    # 读取 CSV 文件
    df = pd.read_csv(variable_file_path)
    variables = df['detail'].tolist()
    dic = dict(zip(df['name'], df['detail']))

    # 重新初始化 BackgroundKnowledge 实例
    bck = BackgroundKnowledge()

    # 读取另一个 CSV 数据集
    df1 = pd.read_csv(data_file_path)
    data = df1.to_numpy()

    # 初始化空的因果关系日志内容
    log_content = ""

    def update_log_content(log_content, new_relation):
        node1, node2 = new_relation.split(" --> ")
        conflict_pattern = rf"{node2} --> {node1}"
        if re.search(conflict_pattern, log_content):
            # 替换已有冲突关系
            log_content = re.sub(conflict_pattern, new_relation, log_content)
            print(f"Conflict detected. Replacing with updated relation: {new_relation}")
        else:
            log_content += new_relation + "\n"
        return log_content

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
            
            # 尝试调用 suggest_pairwise_relationship 并处理异常
            try:
                result = modeler.suggest_pairwise_relationship(dic[str(edge.node1)], dic[str(edge.node2)])
                print(result)
                
                # 在成功生成结果后，更新 log_content，替换冲突关系
                new_relation = f"{edge.node1} --> {edge.node2}"
                log_content = update_log_content(log_content, new_relation)
            except Exception as e:
                print(f"Error encountered in suggest_pairwise_relationship. Attempting to find a match from previous logs. Error: {e}")
                
                # 使用正则匹配日志记录
                node1, node2 = dic[str(edge.node1)], dic[str(edge.node2)]
                pattern = rf"{node1}.*{node2}|{node2}.*{node1}"
                match = re.search(pattern, log_content)
                if match:
                    result = match.group(0)
                    print(f"Using previous analysis result as default: {result}")
                else:
                    result = "No match found; using default relationship."
                    print(result)

            # 继续执行结果处理逻辑
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

    # 将因果图的边信息转换为字典列表
    edges = []
    for edge in cg.G.get_graph_edges():
        edges.append({
            'Node1': str(edge.node1),
            'Node2': str(edge.node2),
            'Endpoint1': edge.endpoint1,
            'Endpoint2': edge.get_endpoint2()
        })

    # 将边信息保存为 DataFrame 并写入 CSV 文件
    df_edges = pd.DataFrame(edges)
    df_edges.to_csv(output_file_path, index=False, encoding='utf-8-sig')  # 指定UTF-8带BOM编码以避免乱码

    # 验证文件是否成功写入
    if os.path.exists(output_file_path):
        print(f"File '{output_file_path}' has been successfully created.")
        
        # 读取文件内容验证
        try:
            df_result = pd.read_csv(output_file_path, encoding='utf-8-sig')  # 使用同样的编码读取
            print("File content preview:")
            print(df_result.head())  # 显示前几行内容
        except Exception as e:
            print(f"Error reading file: {e}")
    else:
        print(f"File '{output_file_path}' was not created.")


# 添加主函数
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run causal analysis")
    parser.add_argument('--variable_file', type=str, required=True, help="Path to the variable file")
    parser.add_argument('--data_file', type=str, required=True, help="Path to the data file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the output file")
    
    args = parser.parse_args()
    
    # 调用因果分析函数
    causal_analysis(args.variable_file, args.data_file, args.output_file)