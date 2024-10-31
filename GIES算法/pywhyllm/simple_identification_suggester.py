from typing import List, Tuple, Dict
from protocols import ModelerProtocol
import networkx as nx
import guidance
from enum import Enum
import re
import itertools
from guidance import models, gen,user,assistant,system,select

class SimpleIdentificationSuggester:

    def suggest_backdoor(
        self,
        factors_list,
        treatment,
        outcome
    ):
        # prompt = """
        # {{#system~}}
        # You are a helpful assistant for causal reasoning.
        # {{~/system}}
        #
        # {{#user~}}
        # Which set or subset of factors in {{factors}} might satisfy the backdoor criteria for identifying the effect of {{treatment}} on {{outcome}}?
        #
        # List the factors satisfying the backdoor criteria enclosing the name of each factor in <backdoor> </backdoor> tags.
        # {{~/user}}
        #
        # {{#assistant}}
        # {{~gen 'backdoors'}}
        # {{~/assistant}}
        # """
        # program = guidance(prompt)
        #
        # executed_program = program(factors=str(factors_list), treatment=treatment, outcome=outcome)
        #
        # if( executed_program._exception is not None):
        #     raise executed_program._exception
        gpt = models.OpenAI("gpt-3.5-turbo")
        with system():
            lm = gpt + "You are a helpful assistant for causal reasoning."

        with user():
            lm += f'''Which set or subset of factors in {factors_list} might satisfy the frontdoor criteria for identifying the effect of {treatment} on {outcome}?
            List the factors satisfying the  backdoor criteria enclosing the name of each factor in <backdoor> </backdoor> tags.'''

        with assistant():
            lm += gen('backdoors')

        executed_program = lm
        print(f"Executed Program: {executed_program}")
        backdoors = executed_program['backdoors']
        backdoors_list = re.findall(r'<backdoor>(.*?)</backdoor>', backdoors)

        return backdoors_list


    def suggest_frontdoor(
        self,
        factors_list,
        treatment,
        outcome
    ):
        # prompt = """
        # {{#system~}}
        # You are a helpful assistant for causal reasoning.
        # {{~/system}}
        #
        # {{#user~}}
        # Which set or subset of factors in {{factors}} might satisfy the frontdoor criteria for identifying the effect of {{treatment}} on {{outcome}}?
        #
        # List the factors satisfying the frontdoor criteria enclosing the name of each factor in <backdoor> </backdoor> tags.
        # {{~/user}}
        #
        # {{#assistant}}
        # {{~gen 'frontdoors'}}
        # {{~/assistant}}
        # """
        # program = guidance(prompt)
        #
        # executed_program = program(factors=str(factors_list), treatment=treatment, outcome=outcome)
        #
        # if( executed_program._exception is not None):
        #     raise executed_program._exception
        gpt = models.OpenAI("gpt-3.5-turbo")
        with system():
            lm = gpt + "You are a helpful assistant for causal reasoning."

        with user():
            lm += f'''Which set or subset of factors in {factors_list} might satisfy the frontdoor criteria for identifying the effect of {treatment} on {outcome}?

            List the factors satisfying the frontdoor criteria enclosing the name of each factor in <frontdoors> </frontdoors> tags.'''

        with assistant():
            lm += gen('frontdoors')

        executed_program = lm
        print(f"Executed Program: {executed_program}")
        frontdoors = executed_program['frontdoors']
        frontdoors_list = re.findall(r'<frontdoor>(.*?)</frontdoor>', frontdoors)

        return frontdoors_list


    def suggest_iv(
        self,
        factors_list,
        treatment,
        outcome
    ):
        # prompt = """
        # {{#system~}}
        # You are a helpful assistant for causal reasoning.
        # {{~/system}}
        #
        # {{#user~}}
        # Which factors in {{factors}} might be valid instrumental variables for identifying the effect of {{treatment}} on {{outcome}}?
        #
        # List the factors that are possible instrumental variables in <iv> </iv> tags.
        # {{~/user}}
        #
        # {{#assistant}}
        # {{~gen 'ivs'}}
        # {{~/assistant}}
        # """
        # program = guidance(prompt)
        #
        # executed_program = program(factors=str(factors_list), treatment=treatment, outcome=outcome)
        #
        # if( executed_program._exception is not None):
        #     raise executed_program._exception
        gpt = models.OpenAI("gpt-3.5-turbo")
        with system():
            lm = gpt + "You are a helpful assistant for causal reasoning."

        with user():
            lm += f'''Which factors in {factors_list} might be valid instrumental variables for identifying the effect of {treatment} on {outcome}?
            List the factors that are possible instrumental variables in <iv> </iv> tags.'''

        with assistant():
            lm += gen('ivs')

        executed_program = lm
        print(f"Executed Program: {executed_program}")
        ivs = executed_program['ivs']
        ivs_list = re.findall(r'<iv>(.*?)</iv>', ivs)

        return ivs_list
