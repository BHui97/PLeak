import torch
import numpy as np
from copy import deepcopy
import re
from util.template import TextTemplate
from torchmetrics import ExtendedEditDistance, CatMetric
from util.data import Harmful
from ModelFactory import ModelFactory

class HotFlip:
    def __init__(self, trigger_token_length=6, shadow_model='gpt2', step=100, template=None, init_triggers='', init_step=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_model = shadow_model
        self.template = TextTemplate(prefix_1='') if template is None else template
        modelFactory = ModelFactory()
        self.model = modelFactory.get_model(shadow_model)
        self.tokenizer = modelFactory.get_tokenizer(shadow_model)
        self.vocab_size = modelFactory.get_vocab_size(shadow_model)
        self.embedding_weight = self.get_embedding_weight()
        self.step = step
        self.init_step = init_step if init_step is not None else self.step
        self.user_prefix = ''
        self.trigger_tokens = self.init_triggers(trigger_token_length, init_triggers, self.user_prefix)

    def init_triggers(self, trigger_token_length, init_trigger='', user_prefix=''):
        init_tokens = self.tokenizer.encode(init_trigger)
        len_init = len(init_tokens)
        len_user = len(self.tokenizer.encode(user_prefix))
        if self.target_model == 'opt' or 'llama' in self.target_model or self.target_model=='vicuna': 
            init_tokens = init_tokens[1:]
            len_init -= 1
            len_user -= 1
        if len_init >= trigger_token_length:
            return np.asarray(init_tokens)
        triggers = np.empty(trigger_token_length-len_user, dtype=int)
        for idx, t in enumerate(triggers):
            if idx < len_init:
                triggers[idx] = init_tokens[idx]
            else:
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
        return self.user_prefix + self.tokenizer.decode(self.trigger_tokens)

    def make_target(self, index, idx_loss, target_text, triggers):
        encoded_target_text = self.tokenizer.encode(target_text)
        encoded_trigger_prefix = self.tokenizer.encode(self.template.prefix_trigger)
        encoded_splash_n = self.tokenizer.encode('\n')
        encoded_user_prefix = self.tokenizer.encode(self.user_prefix)
        if self.target_model == 'opt' or 'llama' in self.target_model or self.target_model=='vicuna': 
            encoded_target = encoded_target_text[1:]
            encoded_trigger_prefix = encoded_trigger_prefix[1:]
            encoded_splash_n = encoded_splash_n[1:]
            encoded_user_prefix = encoded_user_prefix[1:]
        else: encoded_target = encoded_target_text

        encoded_text = encoded_target + encoded_user_prefix+encoded_trigger_prefix + triggers.tolist() + encoded_splash_n + encoded_target

        len_non_label = len(encoded_target)+ len(encoded_user_prefix) + len(encoded_trigger_prefix) + triggers.shape[0] + len(encoded_splash_n)
        max_len = len(encoded_target)
        if max_len > self.max_len: self.max_len = max_len

        label_slice = self.init_step + idx_loss * self.step

        if label_slice > self.max_len:
            encoded_label = [-100]*len_non_label + encoded_target
        else:
            encoded_label = [-100]*len_non_label + encoded_target[:label_slice]

        encoded_text = encoded_text[:len(encoded_label)]
        label = torch.tensor([encoded_label], device=self.device, dtype=torch.long)
        lm_input= torch.tensor([encoded_text], device=self.device, dtype=torch.long)
        return lm_input, label

    def make_target_chat(self, index, idx_loss, target_text, triggers):
        target = [
                {"role": "system", "content": target_text},
                {"role": "user", "content": self.tokenizer.decode(triggers)},
                {"role": "assistant", "content": target_text},
            ]
        target = self.tokenizer.apply_chat_template(target)
        non_label = [
                {"role": "system", "content": target_text},
                {"role": "user", "content": self.tokenizer.decode(triggers)},
            ]
        non_label = self.tokenizer.apply_chat_template(non_label)
        label = [-100]*len(non_label) + target[len(non_label):]

        label = torch.tensor([label], device=self.device, dtype=torch.long)
        lm_input= torch.tensor([target], device=self.device, dtype=torch.long)
        return lm_input, label

    def make_adaptive_chat(self, index, idx_loss, target_text, triggers):
        text = target_text.split('\n')[::-1]
        text = " ".join(text)
        target = [
                {"role": "system", "content": target_text},
                {"role": "user", "content": self.tokenizer.decode(triggers)},
                {"role": "assistant", "content": text},
            ]
        target = self.tokenizer.apply_chat_template(target)
        non_label = [
                {"role": "system", "content": target_text},
                {"role": "user", "content": self.tokenizer.decode(triggers)},
            ]
        non_label = self.tokenizer.apply_chat_template(non_label)
        # lm_input = target[:len(non_label)] + target[len(non_label):][::-1]
        label = [-100]*len(non_label) + target[len(non_label):]

        label = torch.tensor([label], device=self.device, dtype=torch.long)
        lm_input= torch.tensor([target], device=self.device, dtype=torch.long)
        return lm_input, label

    def compute_loss(self, target_texts, trigger_tokens, idx_loss,  require_grad=False):
        total_loss = 0
        for index, text in enumerate(target_texts):
            lm_input, label = self.make_target(index, idx_loss, text, trigger_tokens) 
            loss = self.model(lm_input, labels=label)[0]/len(target_texts)
            if require_grad:
                loss.backward()
            total_loss += loss.item()
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
        print(f"init_triggers:{self.decode_triggers()}")
        self.max_len = self.step+10
        idx_loss = 0
        while idx_loss <= self.max_len//self.step:
            token_flipped = True
            while token_flipped:
                token_flipped = False
                with torch.set_grad_enabled(True):
                    self.model.zero_grad()
                    best_loss = self.compute_loss(target_texts, self.trigger_tokens, idx_loss, require_grad=True)
                print(f"current loss:{best_loss}, triggers:{self.decode_triggers()}")
                
                
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
                if token_flipped: print(f"Loss: {best_loss}, triggers:{self.decode_triggers()}")
                else: print(f"\nNo improvement, ending iteration")
            idx_loss += 1
            print(f"Enter next iteration :{idx_loss}")

