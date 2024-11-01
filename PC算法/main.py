import guidance
# import openai
import os
from dotenv import load_dotenv
# api_base = "https://api.chatanywhere.com.cn/v1"
from guidance import models, gen,user,assistant
gpt = models.OpenAI("gpt-3.5-turbo")

from simple_model_suggester import SimpleModelSuggester

modeler = SimpleModelSuggester()
# result = modeler.suggest_pairwise_relationship('ice cream sales','shark sttacks')
#
# print(result)
#
# if result[0] is not None:
#     print(f"{result[0]} causes {result[1]}")
# else:
#     print(f"neither causes the other")

variables = ["ice cream sales", "temperature", "cavities"]
results,resultlist = modeler.suggest_relationships(variables)
print(results)
print(resultlist)
#
# variables = ["ice cream sales", "temperature", "cavities"]
# latents = modeler.suggest_confounders(variables, treatment="ice cream sales", outcome = "shark attacks")
#
# print(latents)
#
# latents = modeler.suggest_confounders([], treatment="vitamin c", outcome = "cardiovascular health")
#
# print(latents)

from simple_identification_suggester import SimpleIdentificationSuggester
identifier = SimpleIdentificationSuggester()

# variables = ["cigarette taxes", "rain", "car sales", "property taxes", "heart attacks"]
# ivs = identifier.suggest_iv(variables,
#                             treatment="smoking", outcome = "birth weight")
#
# print(ivs)
#
# variables = ["Age", "Sex", "HbA1c", "HDL", "LDL", "eGFR", "Prior MI",
#              "Prior Stroke or TIA", "Prior Heart Failure", "Cardiovascular medication",
#              "T2DM medication", "Insulin", "Morbid obesity", "First occurrence of Nonfatal myocardial infarction, nonfatal stroke, death from all cause",
#              "semaglutide treatment", "Semaglutide medication", "income", "musical taste"]
#
# backdoors = identifier.suggest_backdoor(variables,
#                             treatment="semaglutide treatment", outcome = "cardiovascular health")
#
# print(backdoors)
#
# frontdoors = identifier.suggest_frontdoor(variables,
#                             treatment="semaglutide treatment", outcome = "cardiovascular health")
#
# print(frontdoors)
#
# from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
# bck = BackgroundKnowledge()
#
#
# from causallearn.search.ConstraintBased.PC import pc
# import pandas as pd
# import numpy as np
# df = pd.read_csv("C:\\Users\\27497\\Desktop\\HongKong_self-efficacy_data.csv")
# data = df.to_numpy()
#
# # default parameters
# cg = pc(data)
#
# # # or customized parameters
# # cg = pc(data, alpha, indep_test, stable, uc_rule, uc_priority, mvpc, correction_name, background_knowledge, verbose, show_progress)
#
# # visualization using pydot
# cg.draw_pydot_graph()
#
# # or save the graph
# from causallearn.utils.GraphUtils import GraphUtils
#
# pyd = GraphUtils.to_pydot(cg.G)
# pyd.write_png('simple_test.png')
#
# visualization using networkx
# cg.to_nx_graph()
# cg.draw_nx_graph(skel=False)