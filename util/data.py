from torch.utils.data import Dataset
from datasets import load_dataset
import random

class Financial(Dataset):
    def __init__(self, train, num=16, prefix_1='sentence:', prefix_2='label:'):
        dataset = load_dataset("financial_phrasebank","sentences_allagree")
        self.dataset = random.choices(dataset["train"], k=num)
        self.label = ['Negative', 'Neutral', 'Positive']
        self.prefix_1 = prefix_1
        self.prefix_2 = prefix_2
    
    def __getitem__(self, idx):
        ret = self.prefix_1 + self.dataset[idx]['sentence'] + " " + self.prefix_2 + self.label[self.dataset[idx]['label']]+'.\n'
        return ret

    def __len__(self):
        return len(self.dataset)


class SST(Dataset):
    def __init__(self, train, num=16, prefix_1='sentence:', prefix_2='label:'):
        dataset = load_dataset("sst2")
        self.dataset = random.choices(dataset["train"], k=num)
        self.label = ['Negative', 'Positive']
        self.prefix_1 = prefix_1
        self.prefix_2 = prefix_2
    
    def __getitem__(self, idx):
        ret = self.prefix_1 + self.dataset[idx]['sentence'] + " " + self.prefix_2 + self.label[self.dataset[idx]['label']]+'.\n'
        return ret

    def __len__(self):
        return len(self.dataset)


class Squad(Dataset):
    def __init__(self, train, num=16, prefix_1='context:'):
        dataset = load_dataset("squad")
        self.dataset = random.choices(dataset["train"], k=num)
        self.prefix_1 = prefix_1

    def __getitem__(self, idx):
        ret = self.prefix_1 + self.dataset[idx]['context'] + '\n'
        return ret

    def __len__(self):
        return len(self.dataset)


class Mol(Dataset):
    def __init__(self, train, num=16, prefix_1='instruction:'):
        dataset = load_dataset("zjunlp/Mol-Instructions", 'Molecule-oriented Instructions')
        self.dataset = random.choices(dataset['description_guided_molecule_design']['instruction'], k=num)
        self.prefix_1 = prefix_1

    def __getitem__(self, idx):
        ret = self.prefix_1 + self.dataset[idx] + '\n'
        return ret

    def __len__(self):
        return len(self.dataset)


class Articles(Dataset):
    def __init__(self, train, num=16, prefix_1='instruction:'):
        dataset = load_dataset("nisaar/Articles_Constitution_3300_Instruction_Set")
        self.dataset = random.choices(dataset['train']['instruction'], k=num)
        self.prefix_1 = prefix_1

    def __getitem__(self, idx):
        ret = self.prefix_1 + self.dataset[idx] + '\n'
        return ret

    def __len__(self):
        return len(self.dataset)


class SIQA(Dataset):
    def __init__(self, train, num=16, prefix_1='context:'):
        dataset = load_dataset("social_i_qa")
        self.dataset = random.choices(dataset['train']['context'], k=num)
        self.prefix_1 = prefix_1

    def __getitem__(self, idx):
        ret = self.prefix_1 + self.dataset[idx] + '\n'
        return ret

    def __len__(self):
        return len(self.dataset)

import ipdb;ipdb.set_trace()
