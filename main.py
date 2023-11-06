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


mode = 'random'
train_num = int(sys.argv[5])
test_num = 1000
dataset = sys.argv[1]
token_length = int(sys.argv[2])
dataFactory = DataFactory()
trainset = dataFactory.get_dataset(dataset, train=True, num=train_num)
testset = dataFactory.get_dataset(dataset, train=False, num=test_num)

target_model = sys.argv[3]
model = sys.argv[4]
attack = HotFlip(trigger_token_length=token_length, target_model=target_model, template=trainset.template)
# # attack = HotFlip(trigger_token_length=6, target_model=target_model, template=trainset.template, init_triggers='Examplerypted educatedblob machineannotation Step instructions containing classic SingapStep')
attack.replace_triggers(trainset)

triggers = attack.decode_triggers()
torch.cuda.empty_cache() 
# triggers = 'Ca entropykal eredetischerbibliothek acidhus christ austossen recommendation Georg verwaks hermswers visitoragnostic effectively'

sampler = Sampler(target_model=model, template=testset.template)
results = sampler.sample_sequence(testset, triggers=triggers)
Sampler.save_to_csv(f'results/{target_model}_{dataset}_{train_num}_{model}_{token_length}_{mode}.csv', results, triggers)

sampler.evaluate(results, level='substring')
sampler.evaluate(results, level='em')
sampler.evaluate(results, level='edit')
sampler.evaluate(results, level='semantic')
