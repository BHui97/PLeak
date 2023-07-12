from torch.utils.data import Dataset
from datasets import load_dataset
import random

class GLUE(Dataset):
    def __init__(self, train, num=16, prefix_1='premise:', prefix_2='hypothesis:'):
        dataset = load_dataset("glue", "ax")
        self.dataset = random.choices(dataset["test"], k=num)
        self.prefix_1 = prefix_1
        self.prefix_2 = prefix_2
    
    def __getitem__(self, idx):
        ret = self.prefix_1 + self.dataset[idx]['premise'] + ' '  + self.prefix_2 + self.dataset[idx]['hypothesis']+ '\n'

        return ret

    def __len__(self):
        return len(self.dataset)
