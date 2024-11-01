import guidance
import os
from dotenv import load_dotenv
# api_base = "https://api.chatanywhere.com.cn/v1"


from guidance import models, gen,user,assistant
import pandas as pd

gpt = models.OpenAI("gpt-3.5-turbo")

from simple_model_suggester import SimpleModelSuggester

from simple_identification_suggester import SimpleIdentificationSuggester

modeler = SimpleModelSuggester()
# result = modeler.suggest_pairwise_relationship('ice cream sales','shark sttacks')
#
# print(result)
#
# if result[0] is not None:
#     print(f"{result[0]} causes {result[1]}")
# else:
#     print(f"neither causes the other")
#
# variables = ["ice cream sales", "temperature", "cavities"]
# results = modeler.suggest_relationships(variables)
# print(results)
#
# variables = ["ice cream sales", "temperature", "cavities"]
# latents = modeler.suggest_confounders(variables, treatment="ice cream sales", outcome = "shark attacks")
#
# print(latents)
#
# latents = modeler.suggest_confounders([], treatment="vitamin c", outcome = "cardiovascular health")
#
# print(latents)
#
identifier = SimpleIdentificationSuggester()
#
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

df = pd.read_csv("C:\\Users\\27497\\Desktop\\variable.csv")


variables = df['detail'].tolist()
print(variables)
dic = dict(zip(df['detail'], df['name']))
result_dict = {}
def find_keys_with_substring(dictionary, substring):
    return [key for key in dictionary if substring in key]
for item in variables:
    backdoors = identifier.suggest_backdoor(variables,
                            treatment=item, outcome = "SDLEFF")
    ll = []
    for i in backdoors:
        if i not in dic:
            matching_keys = find_keys_with_substring(dic, i)
            ll.append(dic[matching_keys[0]])
        else:
            ll.append(dic[i])
    result_dict[dic[item]] = ll
print(result_dict)