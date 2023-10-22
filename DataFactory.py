import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from util.data import *

class DataFactory():
    def __init__(self):
        self._creator = {}
        self._register_alias('Financial_1_shot', Financial_1_shot)
        self._register_alias('Financial_3_shot', Financial_3_shot)
        self._register_alias('Tomatoes_1_shot', Tomatoes_1_shot)
        self._register_alias('Tomatoes_3_shot', Tomatoes_3_shot)
        self._register_alias('Awesome', Awesome)
        self._register_alias('SQuAD', SQuAD)
        self._register_alias('SIQA', SIQA)
        self._register_alias('Roles', Roles)

    def _register_alias(self, name, creator):
        self._creator[name] = creator
    def get_dataset(self, name, train=True, num=16):
        creator = self._creator.get(name)
        return creator(train=train, num=num)

