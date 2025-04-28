import pandas as pd
import networkx as nx
from itertools import chain
from collections import defaultdict

# 第一步：读取CSV文件
file_path = '../data/csv/all_directed_edge_without_cycle.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(file_path)

# 第二步：创建有向图
G = nx.DiGraph()

# 第三步：为每个节点创建布尔变量，并添加到图中
for _, row in df.iterrows():
    d1, d2, corr = row['Dimension 1'], row['Dimension 2'], row['Correlation']
    # 创建有向边并附加权重（相关性）
    G.add_edge(d1, d2, correlation=corr)

# 旧版本
# def find_adjustment_vars(graph: nx.DiGraph, start: str, end: str):
#     # Step 1: Convert the directed graph to an undirected graph
#     undirected_graph = graph.to_undirected()
#
#     # Step 2: Find all paths from start to end
#     all_paths = list(nx.all_simple_paths(undirected_graph, source=start, target=end))
#
#     results = []  # To store all result lists for each path
#
#     # Step 3: Process each path
#     for path in all_paths:
#         result_list = []  # The result list for the current path
#
#         # Traverse the path
#         for i in range(len(path) - 1):
#             current_node = path[i]
#             next_node = path[i + 1]
#
#             if graph.has_edge(next_node, current_node):
#                 # Reverse edge (next -> current), add current node to result_list
#                 result_list.append(current_node)
#
#         # After finishing this path, store the result list
#         # 判定防止空集
#         if result_list:
#             results.append(set(result_list))  # Store as a set for easy union/intersection
#
#     # 记录已经选择的变量
#     chosen_variables = set()
#
#     while results:
#         # 统计每个变量阻断路径的数量
#         variable_count = defaultdict(int)
#         for path in results:
#             for variable in path:
#                 variable_count[variable] += 1
#
#         # 选择阻断最多路径的变量
#         best_variable = max(variable_count, key=variable_count.get)
#
#         # 将该变量加入选择集合
#         chosen_variables.add(best_variable)
#
#         # 移除所有包含该变量的路径（因为它们已经被阻断）
#         results = [path for path in results if best_variable not in path]
#
#     return chosen_variables


def find_adjustment_vars(G: nx.DiGraph, X: str, Y: str):
    # 找到所有共同祖先节点
    ancestors_of_X = nx.ancestors(G, X)
    ancestors_of_Y = nx.ancestors(G, Y)
    common_ancestors = ancestors_of_X.intersection(ancestors_of_Y)

    # 存储所有后门路径的节点集合
    backdoor_paths_nodes = []

    # 遍历每个共同祖先节点
    for ancestor in common_ancestors:
        # 找到从公共祖先到X的路径
        paths_to_X = list(nx.all_simple_paths(G, source=ancestor, target=X))
        # 找到从公共祖先到Y的路径
        paths_to_Y = list(nx.all_simple_paths(G, source=ancestor, target=Y))

        # 将从公共祖先到X的路径和从公共祖先到Y的路径组合成完整的后门路径
        for path_X in paths_to_X:
            for path_Y in paths_to_Y:
                # 从路径中去掉公共祖先和终点X、Y，防止重复
                full_path = set(path_X[:-1] + path_Y[1:-1])
                # 添加路径到后门路径集合
                if full_path:
                    backdoor_paths_nodes.append(full_path)

    # 找到覆盖所有路径的最小集合
    return minimum_cover(backdoor_paths_nodes)


# 求最小覆盖集
def minimum_cover(sets):
    # 展开集合中的所有元素
    elements = set(chain(*sets))
    cover = set()

    while sets:
        # 找到最常出现的元素
        most_common = max(elements, key=lambda e: sum(1 for s in sets if e in s))
        cover.add(most_common)

        # 移除包含该元素的所有集合
        sets = [s for s in sets if most_common not in s]
        elements.discard(most_common)

    return list(cover)

out = find_adjustment_vars(G, "单次词数量", "文本难度")
print(out)

