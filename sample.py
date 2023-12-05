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
model = sys.argv[2]
defense = sys.argv[3]
triggers = sys.argv[4]
test_num = 1000

dataFactory = DataFactory()
testset = dataFactory.get_dataset(dataset, train=False, num=test_num)

sampler = Sampler(target_model=model, template=testset.template, defense=defense)
results = sampler.sample_sequence(testset, triggers=triggers)
Sampler.save_to_csv(f'results/{dataset}_{model}_{triggers}.csv', results, triggers)

sampler.evaluate(results, level='substring')
sampler.evaluate(results, level='em')
sampler.evaluate(results, level='edit')
sampler.evaluate(results, level='semantic')
