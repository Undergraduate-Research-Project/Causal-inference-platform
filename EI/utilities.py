import networkx as nx
import pandas as pd
from itertools import chain


def read_dot_graph_as_DG(filename):
    G = nx.nx_pydot.read_dot(filename)
    G = nx.DiGraph(G)
    for u, v in G.edges():
        G[u][v].clear()
    return G


def read_csv_graph(filename):
    df = pd.read_csv(filename)
    G = nx.DiGraph()
    # 使用iloc来选择第一列和第二列
    for _, row in df.iloc[:, :2].iterrows():
        d1, d2 = row[0], row[1]  # 使用索引0和1表示第一列和第二列
        G.add_edge(d1, d2)

    return G


def add_weights_to_graph(graph: nx.Graph, csv_file: str, F):
    # 读取 CSV 文件为 DataFrame
    df = pd.read_csv(csv_file)

    # 遍历图中所有的边
    for u, v in graph.edges():
        # 假设节点的名字直接对应 CSV 中的列名，找到这两列
        if u in df.columns and v in df.columns:
            col_u = df[u]  # 获取与节点 u 对应的列
            col_v = df[v]  # 获取与节点 v 对应的列

            # 计算函数 F(u_column, v_column) 的结果
            weight = F(col_u, col_v)

            # 将计算结果作为边的 'weight' 属性
            graph[u][v]['weight'] = weight

    return graph


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

        # 过滤，除了终点，里面含有X, Y是不能使用的
        paths_to_X = [sublist for sublist in paths_to_X if X not in sublist[:-1]]
        paths_to_X = [sublist for sublist in paths_to_X if Y not in sublist[:-1]]
        paths_to_Y = [sublist for sublist in paths_to_Y if X not in sublist[:-1]]
        paths_to_Y = [sublist for sublist in paths_to_Y if Y not in sublist[:-1]]

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