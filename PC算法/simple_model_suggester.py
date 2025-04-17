from typing import List, Tuple, Dict
from protocols import ModelerProtocol
import networkx as nx
import guidance
from enum import Enum
import re
import itertools
from guidance import models, gen,user,assistant,system,select
class SimpleModelSuggester:
    """
    A class that provides methods for suggesting causal relationships and confounding factors between variables.

    This class uses the guidance library to interact with LLMs, and assumes that guidance.llm has already been initialized to the user's preferred LLM 

    Methods:
    - suggest_pairwise_relationship(variable1: str, variable2: str) -> List[str]: 
        Suggests the causal relationship between two variables and returns a list containing the cause, effect, and a description of the relationship.
    - suggest_relationships(variables: List[str]) -> Dict[Tuple[str, str], str]: 
        Suggests the causal relationships between all pairs of variables in a list and returns a dictionary containing the cause-effect pairs and their descriptions.
    - suggest_confounders(variables: List[str], treatment: str, outcome: str) -> List[str]: 
        Suggests the confounding factors that might influence the relationship between a treatment and an outcome, given a list of variables that have already been considered.
    """

    def suggest_pairwise_relationship(self, variable1: str, variable2: str):
        """
        Suggests a cause-and-effect relationship between two variables.

        Args:
            variable1 (str): The name of the first variable.
            variable2 (str): The name of the second variable.

        Returns:
            list: A list containing the suggested cause variable, the suggested effect variable, and a description of the reasoning behind the suggestion.  If there is no relationship between the two variables, the first two elements will be None.
        """
        # prompt = """
        #         {{#system~}}
        #         You are a helpful assistant for causal reasoning.
        #         {{~/system}}
        #
        #         {{#user~}}
        #         Which cause-and-effect relationship is more likely?
        #         A. {{variable1}} causes {{variable2}}
        #         B. {{variable2}} causes {{variable1}}
        #         C. neither {{variable1}} nor {{variable2}} cause each other
        #         First let's very succinctly think about each option.
        #         Then I'll ask you to provide your final answer A, B, or C.
        #         {{~/user}}
        #
        #         {{#assistant~}}
        #         {{gen 'description'}}
        #         {{~/assistant}}
        #
        #         {{#user~}}
        #         Now what is your final answer: A, B, or C?
        #         {{~/user}}
        #
        #         {{#assistant~}}
        #         {{select 'answer'}}A{{or}}B{{or}}C{{/select}}
        #         {{~/assistant}}
        #     """
        #
        # # 执行模板，并传递变量
        # executed_program = guidance(prompt)(variable1=variable1, variable2=variable2)
        gpt = models.OpenAI("gpt-3.5-turbo")
        with system():
            lm = gpt + "You are a helpful assistant for causal reasoning."

        with user():
            lm += f'''Which cause-and-effect relationship is more likely?
                 A. {variable1} causes {variable2}
                 B. {variable2} causes {variable1}
                 C. neither {variable1} nor {variable2} cause each other
                 D. the cause-and-effect relationship between {variable1} and {variable2} is uncertain
                 First let's very succinctly think about each option.
                 '''

        with assistant():
            lm += gen('description')

        with user():
            lm += f'''Then analyze the probability of each option.'''

        with assistant():
            lm += gen('analyse')

        print(lm['analyse'])

        with user():
            lm += f'''I will ask you to provide your final answer A, B, C, or D. 
                    You will answer this question after I ask you, so do not rush to answer. '''

        with user():
            lm += f'''Please provide your final answer: A, B, C, or D.Please note that this is a multiple-choice question and you must answer it.
                    Again, you must answer this question.
                    Now what is your final answer: A, B, C, or D?'''

        with assistant():
            print(lm['description'])
            lm += select(['A','B','C','D'], name='answer')

        executed_program = lm
        print(f"Executed Program: {executed_program}")

        # 提取生成的描述和答案
        description = executed_program['description']
        answer = executed_program['answer']
        # 使用正则表达式匹配每个部分
        pattern = r"([A-D])\.(.*?)(?=\n[A-D]\.|$)"

        # 创建一个字典来存储匹配的部分
        parts = {m[0]: m[1].strip() for m in re.findall(pattern, description, re.DOTALL)}
        # 根据答案生成对应的返回值
        if answer == "A":
            return [variable1, variable2, 'A. '+parts[answer]]
        elif answer == "B":
            return [variable2, variable1, 'B. '+parts[answer]]
        elif answer == "C":
            return [None, None, 'C. '+parts[answer]]
        else:
            return [None, None, 'D. '+parts[answer]]

    def suggest_relationships(self, variables: List[str]):
        """
        Given a list of variables, suggests relationships between them by querying for pairwise relationships.

        Args:
            variables (List[str]): A list of variable names.

        Returns:
            dict: A dictionary of edges found between variables, where the keys are tuples representing the causal relationship between two variables,
            and the values are the strength of the relationship.
        """
        relationships = {}
        total = (len(variables) * (len(variables)-1)/2)
        i=0
        resultlist = []
        for(var1, var2) in itertools.combinations(variables, 2):
            i+=1
            print(f"{i}/{total}: Querying for relationship between {var1} and {var2}")
            y = self.suggest_pairwise_relationship(var1, var2)
            if( y[0] == None ):
                print(f"\tNo relationship found between {var1} and {var2}")
                continue
            print(f"\t{y[0]} causes {y[1]}")
            relationships[(y[0], y[1])] = y[2]
            resultlist.append([y[0],y[1]])
            print('resultlist:',resultlist)
        return relationships,resultlist
    

    def suggest_confounders(self, variables: List[str], treatment: str, outcome: str) -> List[str]:
        """
        Suggests potential confounding factors that might influence the relationship between the treatment and outcome variables.

        Args:
            variables (List[str]): A list of variables that have already been considered.
            treatment (str): The name of the treatment variable.
            outcome (str): The name of the outcome variable.

        Returns:
            List[str]: A list of potential confounding factors.
        """

        # prompt = """
        #     {{#system~}}
        #     You are a helpful assistant for causal reasoning.
        #     {{~/system}}
        #
        #     {{#user~}}
        #     What latent confounding factors might influence the relationship between {{treatment}} and {{outcome}}?
        #
        #     We have already considered the following factors {{variables}}.  Please do not repeat them.
        #
        #     List the confounding factors between {{treatment}} and {{outcome}} enclosing the name of each factor in <conf> </conf> tags.
        #     {{~/user}}
        #
        #     {{#assistant}}
        #     {{~gen 'latents'}}
        #     {{~/assistant}}
        #     """
        #
        #
        # program = guidance(prompt)
        #
        # executed_program = program(variables=str(variables), treatment=treatment, outcome=outcome)
        gpt = models.OpenAI("gpt-3.5-turbo")
        with system():
            lm = gpt + "You are a helpful assistant for causal reasoning."

        with user():
            lm += f'''What latent confounding factors might influence the relationship between {treatment} and {outcome}?

            We have already considered the following factors {variables}.  Please do not repeat them.

            List the confounding factors between {treatment} and {outcome} enclosing the name of each factor in <conf> </conf> tags.'''

        with assistant():
            lm += gen('latents')


        executed_program = lm
        print(f"Executed Program: {executed_program}")
        
        latents = executed_program['latents']
        latents_list = re.findall(r'<conf>(.*?)</conf>', latents)

        return latents_list


