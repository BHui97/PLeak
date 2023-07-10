from torch.utils.data import Dataset
from datasets import load_dataset
import random

class Financial(Dataset):
    def __init__(self, train, num=16, prefix_1='sentence:', prefix_2='label:'):
        dataset = load_dataset("financial_phrasebank","sentences_allagree")
        self.dataset = random.choices(dataset["train"], k=num)
        self.label = ['negative', 'neutral', 'positive']
        self.prefix_1 = prefix_1
        self.prefix_2 = prefix_2
    
    def __getitem__(self, idx):
        ret = self.prefix_1 + self.dataset[idx]['sentence'] + self.prefix_2 + self.label[self.dataset[idx]['label']]

        return ret

    def __len__(self):
        return len(self.dataset)
        
        
# import ipdb;ipdb.set_trace()

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

