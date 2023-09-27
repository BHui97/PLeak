from attack import HotFlip
import csv
import random
from datasets import load_dataset
import numpy as np
import torch
from datetime import datetime
from collections import OrderedDict
from util.data import *
import os

random.seed(0)
np.random.seed(0)
torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

def save_to_csv(path, results):
    with open(path, 'w', newline='') as file:
        fieldnames = ['context', 'generation']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def attack_data(target_model):
    attack = HotFlip(target_model=target_model)
    attack.find_triggers(trainset)
    results = attack.sample_sequence(testset)
    target_path = f'results/{datetime.now().strftime("%Y_%m_%d-%I_%M")}_{target_model}.csv'
    save_to_csv(target_path, results)

train_num = 32
test_num = 1000
# trainset = Awesome(True, num=train_num)
# testset = Awesome(False, num=test_num)
trainset = Financial(True, num=train_num, with_instruction=False)
testset = Financial(False, num=test_num)
target_model = 'llama'
attack = HotFlip(trigger_token_length=6, target_model=target_model, template=trainset.template)
attack.replace_triggers(trainset)
results = attack.sample_sequence(testset)
# results = attack.sample_sequence(testset, triggers='inst InstructionModLoader egreg innumerableochond')
attack.evaluate(results, level='em')
attack.evaluate(results, level='edit')
attack.evaluate(results, level='semantic')
# attack.evaluate(results, level='em', example_only=False)
# attack.evaluate(results, level='edit', example_only=False)
# attack.evaluate(results, level='semantic', example_only=False)
# trainset, _ = load_financial_data(train_num, test_num)
# _, testset = load_glue_data(train_num, test_num)
# shadow_model = 'gpt2'
# shadow_attack = HotFlip(target_model=shadow_model)
# shadow_attack.find_triggers(trainset)
# triggers = shadow_attack.decode_triggers()
# print(triggers)

# target_model = 'opt'
# attack = HotFlip(target_model=target_model)
# results = attack.sample_sequence(testset, triggers=triggers)
# print(results)
# target_path = f'results/{datetime.now().strftime("%Y_%m_%d-%I_%M")}_{target_model}_{shadow_model}.csv'
# save_to_csv(target_path, results)
# attack_data('gpt2')
