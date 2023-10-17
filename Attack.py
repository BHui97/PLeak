import torch
import numpy as np
from copy import deepcopy
import re
from util.template import TextTemplate
from torchmetrics import ExtendedEditDistance, CatMetric
from util.data import Harmful
from ModelFactory import ModelFactory

class HotFlip:
    def __init__(self, trigger_token_length=6, target_model='gpt2', template=None, init_triggers=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_model = target_model
        self.template = TextTemplate(prefix_1='') if template is None else template
        modelFactory = ModelFactory()
        self.model = modelFactory.get_model(target_model)
        self.tokenizer = modelFactory.get_tokenizer(target_model)
        self.vocab_size = modelFactory.get_vocab_size(target_model)
        self.embedding_weight = self.get_embedding_weight()
        self.trigger_tokens = self.init_triggers(trigger_token_length) if init_triggers==None else np.array(self.tokenizer.encode(init_triggers)[1:], dtype=int)
        self.step = 100

    def init_triggers(self, trigger_token_length):
        triggers = np.empty(trigger_token_length, dtype=int)
        for idx, t in enumerate(triggers):
            t = np.random.randint(self.vocab_size)
            while re.search("[^a-zA-Z0-9s\s]", self.tokenizer.decode(t)):
                t = np.random.randint(self.vocab_size)
            triggers[idx] = t
        return triggers

    def get_embedding_weight(self):
        for module in self.model.modules():
            if not isinstance(module, torch.nn.Embedding): continue
            if module.weight.shape[0] != self.vocab_size: continue
            module.weight.requires_grad = True
            return module.weight.detach()

    def get_triggers_grad(self):
        for module in self.model.modules():
            if not isinstance(module, torch.nn.Embedding): continue
            if module.weight.shape[0] != self.vocab_size: continue
            return module.weight.grad[self.trigger_tokens]
    
    def decode_triggers(self):
        return self.tokenizer.decode(self.trigger_tokens)

    def make_target(self, index, target_text, triggers):
        encoded_target_text = self.tokenizer.encode(target_text)
        encoded_trigger_prefix = self.tokenizer.encode(self.template.prefix_trigger)
        encoded_splash_n = self.tokenizer.encode('\n')
        if self.target_model == 'opt' or self.target_model == 'llama' or self.target_model=='vicuna': 
            encoded_target = encoded_target_text[1:]
            encoded_trigger_prefix = encoded_trigger_prefix[1:]
            encoded_splash_n = encoded_splash_n[1:]
        else: encoded_target = encoded_target_text

        encoded_text = encoded_target + encoded_trigger_prefix + triggers.tolist() + encoded_splash_n + encoded_target

        len_non_label = len(encoded_target)+len(encoded_trigger_prefix) + triggers.shape[0] + len(encoded_splash_n)
        encoded_label = [-100]*len_non_label + encoded_target

        label = torch.tensor([encoded_label], device=self.device, dtype=torch.long)
        lm_input= torch.tensor([encoded_text], device=self.device, dtype=torch.long)
        return lm_input, label

    def compute_loss(self, target_texts, trigger_tokens, idx_loss,  require_grad=False):
        total_loss = 0
        for index, text in enumerate(target_texts):
            lm_input, label = self.make_target(index, text, trigger_tokens) 
            loss = self.model(lm_input, labels=label)[0]/len(target_texts)
            total_loss += loss.item()
            if require_grad:
                loss.backward()
        return total_loss

    def hotflip_attack(self, averaged_grad, increase_loss=False, num_candidates=30):
        averaged_grad = averaged_grad
        embedding_matrix = self.embedding_weight
        averaged_grad = averaged_grad.unsqueeze(0)
        gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik",
                (averaged_grad, embedding_matrix))        
        gradient_dot_embedding_matrix *= -1 
        _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
        return best_k_ids.detach().squeeze().cpu().numpy()

    def replace_triggers(self, target_texts):
        print(f"init_triggers:{self.tokenizer.decode(self.trigger_tokens)}")
        self.max_len = 200
        idx_loss = 1
        while idx_loss * self.step < self.max_len:
            token_flipped = True
            while token_flipped:
                token_flipped = False
                with torch.set_grad_enabled(True):
                    self.model.zero_grad()
                    best_loss = self.compute_loss(target_texts, self.trigger_tokens, idx_loss, require_grad=True)
                print(f"current loss:{best_loss}, triggers:{self.tokenizer.decode(self.trigger_tokens)}")
                
                candidates = self.hotflip_attack(self.get_triggers_grad(), num_candidates=30)
                best_trigger_tokens = deepcopy(self.trigger_tokens)
                for i, token_to_flip in enumerate(self.trigger_tokens):
                    for cand in candidates[i]:
                        if re.search("[^a-zA-Z0-9s\s]", self.tokenizer.decode(cand)): continue
                        candidate_trigger_tokens = deepcopy(self.trigger_tokens)
                        candidate_trigger_tokens[i] = cand
                        self.model.zero_grad()
                        with torch.no_grad():
                            loss = self.compute_loss(target_texts, candidate_trigger_tokens, idx_loss, require_grad=False)
                        if best_loss <= loss: continue
                        token_flipped = True
                        best_loss = loss
                        best_trigger_tokens = deepcopy(candidate_trigger_tokens)
                self.trigger_tokens = deepcopy(best_trigger_tokens)
                if token_flipped: print(f"Loss: {best_loss}, triggers:{self.tokenizer.decode(self.trigger_tokens)}")
                else: print(f"\nNo improvement, ending iteration")
            idx_loss += 1
            print(f"Enter next iteration :{idx_loss}")

