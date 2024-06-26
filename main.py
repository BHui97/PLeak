from Attack import HotFlip
from Sampler import Sampler
import random
from datasets import load_dataset
import numpy as np
import torch
from datetime import datetime
from collections import OrderedDict
import os
from DataFactory import DataFactory
import sys

random.seed(0)
np.random.seed(0)
torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

dataset = sys.argv[1]
token_length = int(sys.argv[2])
shadow_model = sys.argv[3]
target_model = sys.argv[4]
train_num = int(sys.argv[5])
test_num = 1000

dataFactory = DataFactory()
trainset = dataFactory.get_dataset(dataset, train=True, num=train_num)
testset = dataFactory.get_dataset(dataset, train=False, num=test_num)
attack = HotFlip(trigger_token_length=token_length, shadow_model=shadow_model, template=trainset.template)
attack.replace_triggers(trainset)

triggers = attack.decode_triggers()

del attack
torch.cuda.empty_cache() 

sampler = Sampler(target_model=target_model, template=testset.template)
results = sampler.sample_sequence(testset, triggers=triggers)
Sampler.save_to_csv(f'results/{dataset}_{token_length}_{target_model}_{model}_{train_num}.csv', results, triggers)

sampler.evaluate(results, level='substring')
sampler.evaluate(results, level='em')
sampler.evaluate(results, level='edit')
sampler.evaluate(results, level='semantic')
