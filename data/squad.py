from torch.utils.data import Dataset
from datasets import load_dataset
import random

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
# import ipdb;ipdb.set_trace()

# def load_squad_data(train_num, test_num):
#     dataset = load_dataset("squad")
#     # train_context = list(OrderedDict.fromkeys(dataset['train']['question']))
#     train_context = dataset['train']['question']
#     trainset = random.choices(train_context, k=train_num)
#     testset = random.choices(train_context, k=test_num)
#     return trainset, testset
