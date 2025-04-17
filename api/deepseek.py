from openai import OpenAI  # 注意这里改为同步的OpenAI客户端
import os
from dotenv import load_dotenv
import networkx as nx
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

class DeepSeekClient:
    def __init__(self):
        self.client = OpenAI(  # 使用同步客户端
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1"
        )
        self.causal_graph = None  # 存储因果图
    
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
        """生成因果图"""
        self.causal_graph = nx.DiGraph()
        self.causal_graph.add_nodes_from(variables)
        
        for i in range(len(variables)):
            for j in range(len(variables)):
                if i == j:
                    continue
                
                var1, var2 = variables[i], variables[j]
                relation = self.get_causal_relation(var1, var2)
                print(f"{var1} -> {var2}: {relation}")
                if "have causal relationship" in relation:
                    self.causal_graph.add_edge(var1, var2)
        
        return self.causal_graph

