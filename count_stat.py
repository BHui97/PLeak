import csv
from Sampler import Sampler
import random
from datasets import load_dataset
import numpy as np
import torch
from datetime import datetime
from collections import OrderedDict
from util.data import *
import os
from torchmetrics import ExtendedEditDistance, CatMetric

random.seed(0)
np.random.seed(0)
torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

train_num = 16
test_num = 1000
num_shots = 1
# trainset = Financial(True, num=train_num, num_shots=num_shots, with_instruction=True)
# testset = Tomatoes(False, num=test_num, num_shots=num_shots, with_instruction=True)

# trainset = Awesome(True, num=train_num)
testset = Awesome(False, num=test_num)
target_model = 'llama'
model = 'gptj'

metric = CatMetric()
attack = Sampler()

for data in testset:
    metric.update(len(attack.tokenizer.encode(data)))
std, mean = torch.std_mean(metric.compute())
print(f"edit distance mean: {mean.item()}, std: {std.item()}")
