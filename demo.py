from attack import HotFlip
import csv
import random
from datasets import load_dataset
import numpy as np
import torch
from datetime import datetime

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

def attack_finacial_data(target_model):
    attack = HotFlip(target_model=target_model)
    attack.find_triggers(trainset)
    results = attack.sample_sequence(testset)
    print(results)
    target_path = f'results/{datetime.now().strftime("%Y_%m_%d-%I_%M")}_{target_model}.csv'
    save_to_csv(target_path, results)

train_num = 16
test_num = 1000
dataset = load_dataset("financial_phrasebank","sentences_allagree")
trainset = random.choices(dataset["train"]["sentence"], k=train_num)
testset = random.choices(dataset["train"]["sentence"], k=test_num)


# shadow_model = 'opt'
# shadow_attack = HotFlip(target_model=shadow_model)
# shadow_attack.find_triggers(trainset)
# triggers = shadow_attack.decode_triggers()
# print(triggers)

# target_model = 'gpt2'
# attack = HotFlip(target_model=target_model)
# results = attack.sample_sequence(testset, triggers=triggers)
# print(results)
# target_path = f'results/{datetime.now().strftime("%Y_%m_%d-%I_%M")}_{target_model}_{shadow_model}.csv'
# save_to_csv(target_path, results)
attack_finacial_data('gpt2')
