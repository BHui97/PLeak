from attack import HotFlip
import csv
import random
from datasets import load_dataset
import numpy as np
import torch
from datetime import datetime
from collections import OrderedDict

random.seed(0)
np.random.seed(0)
torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

def load_financial_data(train_num, test_num):
    dataset = load_dataset("financial_phrasebank","sentences_allagree")
    trainset = random.choices(dataset["train"]["sentence"], k=train_num)
    testset = random.choices(dataset["train"]["sentence"], k=test_num)
    return trainset, testset

def load_squad_data(train_num, test_num):
    dataset = load_dataset("squad")
    # train_context = list(OrderedDict.fromkeys(dataset['train']['question']))
    train_context = dataset['train']['question']
    trainset = random.choices(train_context, k=train_num)
    testset = random.choices(train_context, k=test_num)
    return trainset, testset

def load_glue_data(train_num, test_num):
    dataset = load_dataset("glue", "ax")
    # train_context = list(OrderedDict.fromkeys(dataset['train']['question']))
    train_context = dataset['test']['premise']
    trainset = random.choices(train_context, k=train_num)
    testset = random.choices(train_context, k=test_num)
    return trainset, testset

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

train_num = 16
test_num = 1000

trainset, _ = load_financial_data(train_num, test_num)
_, testset = load_glue_data(train_num, test_num)
shadow_model = 'gpt2'
shadow_attack = HotFlip(target_model=shadow_model)
shadow_attack.find_triggers(trainset)
triggers = shadow_attack.decode_triggers()
print(triggers)

target_model = 'opt'
attack = HotFlip(target_model=target_model)
results = attack.sample_sequence(testset, triggers=triggers)
# print(results)
# target_path = f'results/{datetime.now().strftime("%Y_%m_%d-%I_%M")}_{target_model}_{shadow_model}.csv'
# save_to_csv(target_path, results)
# attack_data('gpt2')
