from openai import OpenAI  # 注意这里改为同步的OpenAI客户端
import os
from dotenv import load_dotenv
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
# Load environment variables
load_dotenv()

class DeepSeekClient:
    def __init__(self, csv_file='static/result.csv'):
        self.client = OpenAI(  # 使用同步客户端
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
        self.causal_graph = None  # 存储因果图
        
        # 正确找到csv文件的绝对路径
        current_dir = os.path.dirname(__file__)  # 当前deepseek.py所在目录 (api/)
        csv_path = os.path.join(current_dir, '..', csv_file)
        csv_path = os.path.abspath(csv_path)  # 转换成绝对路径

        self.csv_file = csv_path  # 保存绝对路径
    
    def get_response(self, messages):
        try:
            response = self.client.chat.completions.create(  # 同步调用
                model="deepseek-chat",
                messages=messages,
                stream=False
            )

            # Ensure proper response structure handling
            if hasattr(response.choices[0].message, 'content'):
                return {
                    "content": response.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }
            else:
                raise ValueError("Response missing 'content' field")
                
        except Exception as e:
            print(f"API request error: {str(e)}")
            return {"content": "no causal relationship"}

    def get_causal_relation(self, variable1, variable2):
        """判断两个变量间的因果关系"""
        prompt = f"""As a causal inference expert, rigorously evaluate the causal relationship between:
        Variable 1: {variable1}
        Variable 2: {variable2}
        
        Return ONLY one of these exact phrases:
        - "have causal relationship" (if "{variable1} causes {variable2}")
        - "no causal relationship" (if no plausible causal mechanism exists)"""

        messages = [
            {"role": "system", "content": "You are a rigorous causal inference expert"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.get_response(messages)
        content = response.get("content", "no causal relationship")
        return content.lower().strip()

    # def generate_causal_graph(self, variables):
    #     """生成因果图"""
    #     self.causal_graph = nx.DiGraph()
    #     self.causal_graph.add_nodes_from(variables)
        
    #     for i in range(len(variables)):
    #         for j in range(len(variables)):
    #             if i == j:
    #                 continue
                
    #             var1, var2 = variables[i], variables[j]
    #             relation = self.get_causal_relation(var1, var2)
    #             print(f"{var1} -> {var2}: {relation}")
    #             if "have causal relationship" in relation:
    #                 self.causal_graph.add_edge(var1, var2)
        
    #     return self.causal_graph


    def generate_causal_graph(self, variables):
        """根据已有PC输出文件，筛选无向边并生成因果图"""

        print("初始化因果图...")
        self.causal_graph = nx.DiGraph()
        self.causal_graph.add_nodes_from(variables)
        print(f"已添加节点: {variables}\n")

        # 统一设置路径
        current_dir = os.path.dirname(__file__)
        csv_path = os.path.abspath(os.path.join(current_dir, '..', 'static', 'result.csv'))

        print(f"正在读取PC算法输出文件 {csv_path} ...")
        df = pd.read_csv(csv_path)
        print(f"读取完成，边数量：{len(df)}\n")

        # 初始化收集所有最终要保存的边
        edges_for_output = []

        for idx, row in df.iterrows():
            node1 = row['Node1']
            node2 = row['Node2']
            endpoint1 = row['Endpoint1']
            endpoint2 = row['Endpoint2']

            print(f"处理第{idx+1}条边: {node1} —— {node2} ({endpoint1}, {endpoint2})")

            if endpoint1 == 'TAIL' and endpoint2 == 'TAIL':
                # 无向边，需要询问
                print(f"无向边，需要推断因果方向：{node1} -- {node2}")
                relation = self.get_causal_relation(node1, node2)
                print(f"推断结果: {relation}")

                if "have causal relationship" in relation:
                    self.causal_graph.add_edge(node1, node2)
                    edges_for_output.append({'Node1': node1, 'Node2': node2, 'Endpoint1': 'TAIL', 'Endpoint2': 'ARROW'})
                    print(f"添加有向边: {node1} → {node2}\n")
                elif "reverse causal relationship" in relation:
                    self.causal_graph.add_edge(node2, node1)
                    edges_for_output.append({'Node1': node2, 'Node2': node1, 'Endpoint1': 'TAIL', 'Endpoint2': 'ARROW'})
                    print(f"添加有向边: {node2} → {node1}\n")
                else:
                    print(f"没有因果关系，不添加边: {node1} -- {node2}\n")
            else:
                # 原本就有向的边，直接保留
                if endpoint1 == 'TAIL' and endpoint2 == 'ARROW':
                    self.causal_graph.add_edge(node1, node2)
                    edges_for_output.append({'Node1': node1, 'Node2': node2, 'Endpoint1': 'TAIL', 'Endpoint2': 'ARROW'})
                    print(f"保留已有向边: {node1} → {node2}\n")
                elif endpoint1 == 'ARROW' and endpoint2 == 'TAIL':
                    self.causal_graph.add_edge(node2, node1)
                    edges_for_output.append({'Node1': node2, 'Node2': node1, 'Endpoint1': 'TAIL', 'Endpoint2': 'ARROW'})
                    print(f"保留已有向边: {node2} → {node1}\n")
                else:
                    print(f"⚠️ 出现未处理的端点组合: {endpoint1}-{endpoint2}，跳过或后续处理\n")

        print("因果图生成完成！节点数:", self.causal_graph.number_of_nodes(), 
            "边数:", self.causal_graph.number_of_edges())

        # 保存新的因果关系到 static/result_new.csv
        output_path = os.path.abspath(os.path.join(current_dir, '..', 'static', 'result_new.csv'))
        output_df = pd.DataFrame(edges_for_output)
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"新因果边结果已保存到: {output_path}")

        # 返回生成好的因果图
        return self.causal_graph