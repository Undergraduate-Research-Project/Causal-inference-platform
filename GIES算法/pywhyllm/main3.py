from guidance import models, gen,user,assistant
import pandas as pd
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from simple_model_suggester import SimpleModelSuggester
from causallearn.graph.Node import Node
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Endpoint import Endpoint

def find_keys_with_substring(dictionary, substring):
    return [key for key in dictionary if substring in key]


bck = BackgroundKnowledge()
# gpt = models.OpenAI("gpt-3.5-turbo")


modeler = SimpleModelSuggester()

# variables = ["ice cream sales", "temperature", "cavities"]
# results,resultlist = modeler.suggest_relationships(variables)
# print(results)
# print(resultlist)
#B: Diameter -> Length,Viscera weight -> Length,Shucked weight -> Whole weight
#C: Viscera weight Shell weight; C: Whole weight Height; C: Whole weight Viscera weight;
df = pd.read_csv("C:\\Users\\27497\\Desktop\\Causality\\因果发现\\variable_dataset2.csv")
variables = df['detail'].tolist()
print(variables)
dic = dict(zip(df['name'],df['detail']))
print(dic)

# resultlist = [['Rings of abalone', 'Shell weight(after being dried)'], ['Diameter(perpendicular to length)', 'Rings of abalone'], ['Rings of abalone', 'Height(with meat in shell)'], ['Whole weight of abalone', 'Rings of abalone'], ['Rings of abalone', 'Shucked weight(weight of meat)'], ['the longest shell measurement of abalone', 'Shell weight(after being dried)'], ['the longest shell measurement of abalone', 'Diameter(perpendicular to length)'], ['the longest shell measurement of abalone', 'Height(with meat in shell)']]

bck = BackgroundKnowledge()
from causallearn.search.ConstraintBased.PC import pc
import pyreadstat
# countryName = 'Hong Kong (China)'
# k = 20
# file_path = f"D:\data\原始数据\STU_QQQ_SPSS\country\{countryName}-v1(k={k}).sav"
#
# # 使用read_csv函数读取CSV文件
# df1, meta = pyreadstat.read_sav(file_path)
# print(df1)
df1 = pd.read_csv("C:\\Users\\27497\\Desktop\\Causality\\因果发现\\dataset2_2.csv")
data = df1.to_numpy()
pos = 1
cnt=0
cnt+=1



while pos:
    print("round:",cnt)
    cnt += 1
    pos = 0
    cg = pc(data=data,background_knowledge=bck,node_names=df['name'].tolist())
    for edge in cg.G.get_graph_edges():
        if bck.is_required(GraphNode(str(edge.node1)), GraphNode(str(edge.node2))) or bck.is_required(GraphNode(str(edge.node2)), GraphNode(str(edge.node1))) or (bck.is_forbidden(GraphNode(str(edge.node1)), GraphNode(str(edge.node2))) and bck.is_forbidden(GraphNode(str(edge.node2)), GraphNode(str(edge.node1)))):
            continue
        print(edge.node1,edge.node2,edge.endpoint1,edge.get_endpoint2())
        result = modeler.suggest_pairwise_relationship(dic[str(edge.node1)],dic[str(edge.node2)])
        print(result)
        if result[2].startswith("A"):
            if edge.endpoint1!=Endpoint.TAIL or edge.get_endpoint2()!=Endpoint.ARROW:
                bck.add_required_by_node(GraphNode(str(edge.node1)), GraphNode(str(edge.node2)))
                bck.add_forbidden_by_node(GraphNode(str(edge.node2)), GraphNode(str(edge.node1)))
                print("A:",str(edge.node1),"->",str(edge.node2))
                pos = 1
        elif result[2].startswith("B"):
            if edge.endpoint1 != Endpoint.ARROW or edge.get_endpoint2() != Endpoint.TAIL:
                bck.add_required_by_node(GraphNode(str(edge.node2)), GraphNode(str(edge.node1)))
                bck.add_forbidden_by_node(GraphNode(str(edge.node1)), GraphNode(str(edge.node2)))
                print("B:", str(edge.node2), "->",str(edge.node1))
                pos = 1
        elif result[2].startswith("C"):
            bck.add_forbidden_by_node(GraphNode(str(edge.node1)), GraphNode(str(edge.node2)))
            bck.add_forbidden_by_node(GraphNode(str(edge.node2)), GraphNode(str(edge.node1)))
            print("C:", str(edge.node1), str(edge.node2))
            pos = 1
    for edge in cg.G.get_graph_edges():
        print(edge)
print(cg.G)
print(cg.G.graph)
print(cg.G.graph.tofile('result2.csv'))
    # for ll in resultlist:
    #     if ll[0] not in dic:
    #         matching_keys = find_keys_with_substring(dic, ll[0])
    #         x=dic[matching_keys[0]]
    #     else:
    #         x=dic[ll[0]]
    #     if ll[1] not in dic:
    #         matching_keys = find_keys_with_substring(dic, ll[1])
    #         y=dic[matching_keys[0]]
    #     else:
    #         y=dic[ll[1]]
    #     print(x,y)
    #     bck.add_required_by_node(GraphNode(x),GraphNode(y))

