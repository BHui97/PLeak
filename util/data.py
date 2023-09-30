from torch.utils.data import Dataset
from datasets import load_dataset
from util.template import TextTemplate
import random

class Financial(Dataset):
    def __init__(self, train, num=16, prefix_1='sentence:', prefix_2='label:',with_instruction=True):
        dataset = load_dataset("financial_phrasebank","sentences_allagree")
        self.dataset = random.choices(dataset["train"], k=num)
        self.label = ['Negative', 'Neutral', 'Positive']
        self.template = TextTemplate(prefix_1 = prefix_1, prefix_2=prefix_2)
        self.instruction_prefix = "instruction:"
        self.instruction = (self.instruction_prefix+"Please analyze the sentiment of following sentences.\n\n") if with_instruction else ''
    
    def __getitem__(self, idx):
        return self.instruction + self.template(self.dataset[idx]['sentence'], self.label[self.dataset[idx]['label']])
        # return self.template(self.instruction+self.dataset[idx]['sentence'], self.label[self.dataset[idx]['label']])

    def __len__(self):
        return len(self.dataset)


class SST(Dataset):
    def __init__(self, train, num=16, prefix_1='sentence:', prefix_2='label:', with_instruction=False):
        dataset = load_dataset("sst2")
        self.dataset = random.choices(dataset["train"], k=num)
        self.label = ['Negative', 'Positive']
        self.template = TextTemplate(prefix_1 = prefix_1, prefix_2=prefix_2)
        self.instruction_prefix = "instruction:"
        self.instruction = (self.instruction_prefix+"Please analyze the sentiment of following sentences.\n\n") if with_instruction else ''
    
    def __getitem__(self, idx):
        return self.instruction + self.template(self.dataset[idx]['sentence'], self.label[self.dataset[idx]['label']])
    
    def __len__(self):
        return len(self.dataset)

class Tomatoes(Dataset):
    def __init__(self, train, num=16, prefix_1='sentence:', prefix_2='label:', with_instruction=True):
        dataset = load_dataset("rotten_tomatoes")
        self.dataset = random.choices(dataset["train"], k=num)
        self.label = ['Negative', 'Positive']
        self.template = TextTemplate(prefix_1 = prefix_1, prefix_2=prefix_2)
        self.instruction_prefix = "instruction:"
        self.instruction = (self.instruction_prefix+"Please analyze the sentiment of following sentences.\n\n") if with_instruction else ''
    
    def __getitem__(self, idx):
        return self.instruction + self.template(self.dataset[idx]['text'], self.label[self.dataset[idx]['label']])
    
    def __len__(self):
        return len(self.dataset)


class SQuAD(Dataset):
    def __init__(self, train, num=16, prefix_1='context:'):
        dataset = load_dataset("squad")
        self.dataset = random.choices(dataset["train"], k=num)
        self.prefix_1 = prefix_1
        self.template = TextTemplate(prefix_1 = prefix_1)

    def __getitem__(self, idx):
        return self.template(self.dataset[idx]['context'])

    def __len__(self):
        return len(self.dataset)


class SIQA(Dataset):
    def __init__(self, train, num=16, prefix_1='context:'):
        dataset = load_dataset("social_i_qa")
        self.dataset = random.choices(dataset['train']['context'], k=num)
        self.template = TextTemplate(prefix_1 = prefix_1)

    def __getitem__(self, idx):
        return self.template(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

class Awesome(Dataset):
    def __init__(self, train, num=16, prefix_1=''):
        dataset = load_dataset("csv", data_files="util/prompts.csv")
        self.dataset = random.choices(dataset['train']['prompt'], k=num)
        self.template = TextTemplate(prefix_1 = prefix_1, prefix_2='')
        self.instruction_prefix = ""

    def __getitem__(self, idx):
        return self.instruction_prefix+self.dataset[idx]+'\n'

    def __len__(self):
        return len(self.dataset)


class Harmful(Dataset):
    def __init__(self, train, num=16, prefix_1=''):
        dataset = load_dataset("csv", data_files="util/harmful_behaviors.csv")
        self.dataset = random.choices(dataset['train']['goal'], k=num)

    def __getitem__(self, idx):
        return self.dataset[idx]

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


class Alpaca(Dataset):
    def __init__(self, train, num=16, prefix_1='instruction:'):
        dataset = load_dataset("tatsu-lab/alpaca")
        self.dataset = random.choices(dataset['train']['instruction'], k=num)
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



