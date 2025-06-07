from flask import jsonify 
from openai import OpenAI  # 注意这里改为同步的OpenAI客户端
import os
from dotenv import load_dotenv
import networkx as nx
import pandas as pd
import json
import logging
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
                    print(f"出现未处理的端点组合: {endpoint1}-{endpoint2}，跳过或后续处理\n")

        print("因果图生成完成！节点数:", self.causal_graph.number_of_nodes(), 
            "边数:", self.causal_graph.number_of_edges())

        # 保存新的因果关系到 static/result_new.csv
        output_path = os.path.abspath(os.path.join(current_dir, '..', 'static', 'result_new.csv'))
        output_df = pd.DataFrame(edges_for_output)
        output_df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"新因果边结果已保存到: {output_path}")

        # 返回生成好的因果图
        return self.causal_graph
    
    def backdoorAdjustment(self,data):
            try:
                
                # 验证数据结构
                required = ['cause_var', 'effect_var', 'nodes', 'edges']
                if not all(key in data for key in required):
                    return jsonify({"error": "缺少必要参数"}), 400
                    
                # 验证边格式
                if not all(isinstance(edge, dict) and 'from' in edge and 'to' in edge 
                        for edge in data['edges']):
                    return jsonify({"error": "edges格式错误"}), 400

                # 生成边列表字符串
                edges_str = '\n'.join([f"{e['from']} → {e['to']}" for e in data['edges']])

                # 构造提示词，明确要求严格JSON格式
                prompt = f"""作为因果推断专家，请基于以下因果图分析：
                                
                        【因果图结构】
                        变量列表：{', '.join(data['nodes'])}
                        因果边：
                        {edges_str}

                        【分析任务】
                        确定在估计 {data['cause_var']} 对 {data['effect_var']} 的因果效应时，需要调整的最小变量集合。

                        请严格遵循后门准则：
                        1. ​**阻断所有非因果路径**​（后门路径）:  
                        - 使用d-分离原则识别所有从 {data['cause_var']} 到 {data['effect_var']} 的 ​**非因果路径**​（即指向 {data['cause_var']} 的路径）。
                        - 确保调整集合阻断这些路径（如链结构 $i \rightarrow m \rightarrow j$ 或分叉结构 $i \leftarrow m \rightarrow j$ 需包含$m$，对撞结构 $i \rightarrow m \leftarrow j$ 需不含$m$及其后代）。

                        2. ​**不包含任何中介变量**​（即 {data['cause_var']} 的后代）:  
                        - 排除所有位于 {data['cause_var']} 到 {data['effect_var']} 因果路径上的变量。

                        请返回严格符合以下JSON格式的内容，不要包含任何其他文本或注释：
                        {{"adjustment_set": [...]}}"""

                # 调用大模型
                messages = [
                    {"role": "system", "content": "你是一个遵循Judea Pearl因果准则的专家"},
                    {"role": "user", "content": prompt}
                ]
                response = self.get_response(messages)
                model_response = response['content'].strip()
                
                logging.info(f"模型原始响应: {model_response}")
                
                # 预处理响应内容，去除可能的代码块标记
                if model_response.startswith('```json'):
                    model_response = model_response[7:-3].strip()
                elif model_response.startswith('```'):
                    model_response = model_response[3:-3].strip()

                # 防御性解析
                try:
                    result = json.loads(model_response)
                    if not isinstance(result.get('adjustment_set'), list):
                        raise ValueError("adjustment_set应为列表类型")
                        
                    return jsonify(result), 200
                except json.JSONDecodeError as e:
                    logging.error(f"JSON解析失败: {e}，响应内容: {model_response}")
                    return jsonify({"adjustment_set": [], "error": "响应格式无效"}), 200
                except Exception as e:
                    logging.error(f"解析错误: {e}，响应内容: {model_response}")
                    return jsonify({"adjustment_set": [], "error": "解析响应失败"}), 200

            except Exception as e:
                logging.error(f"后端错误: {str(e)}")
                return jsonify({"error": "分析服务不可用"}), 500
    
    def analyze_causal_data(self, prompt):
        """分析因果数据的方法"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的因果推理分析专家。请基于提供的因果关系数据，进行深入的分析并给出专业的结论和建议。"
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ]
            
            response = self.get_response(messages)
            return response.get("content", "分析失败")
            
        except Exception as e:
            logging.error(f"因果数据分析失败: {str(e)}")
            return None