from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, OPTForCausalLM, GPTJForCausalLM
import torch
import numpy as np
from copy import deepcopy
import re
from nltk import pos_tag, word_tokenize
import nltk

class HotFlip:
    def __init__(self, trigger_token_length=6, target_model='gpt2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_model = target_model
        if target_model == 'gpt2':
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2').eval().to(self.device)
            self.vocab_size = 50257
        elif target_model == 'gptj':
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
            self.model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").eval().to(self.device)
            self.vocab_size=50257
        elif target_model == 'opt':
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
            self.model = OPTForCausalLM.from_pretrained("facebook/opt-350m").eval().to(self.device)
            self.vocab_size = 50272
        self.embedding_weight = self.get_embedding_weight()
        self.add_hook()
        self.trigger_tokens = np.random.randint(self.vocab_size, size=trigger_token_length)
        self.split_symbol = self.tokenizer.encode('.')[-1]
        self.trigger_tokens = np.insert(self.trigger_tokens, 0, self.split_symbol)

    def add_hook(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Embedding):
                if module.weight.shape[0] == self.vocab_size:
                    module.weight.requires_grad = True

    def get_embedding_weight(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Embedding):
                if module.weight.shape[0] == self.vocab_size:
                    return module.weight.detach()
    
    def decode_triggers(self):
        return self.tokenizer.decode(self.trigger_tokens)

    def make_target_batch(self, target_texts, triggers):
        # encode items and get the max length
        encoded_texts = []
        encoded_labels = []
        max_len = 0
        for target_text in target_texts:
            len_prefix = len(self.tokenizer.encode(": "))
            target_text = ": " +  target_text
            encoded_target_text = self.tokenizer.encode(target_text)
            # encoded_text = encoded_target_text + triggers.tolist() + self.tokenizer.encode('"') + encoded_target_text
            # encoded_label = [-100]*(len(encoded_target_text)+triggers.shape[0]+len(self.tokenizer.encode('"'))) + encoded_target_text 
            encoded_text = encoded_target_text + triggers.tolist() + encoded_target_text
            encoded_label = [-100]*(len(encoded_target_text)+triggers.shape[0] + len_prefix) + encoded_target_text[len_prefix:]
            encoded_texts.append(encoded_text)
            encoded_labels.append(encoded_label)
            if len(encoded_text) > max_len:
                max_len = len(encoded_text)

        # pad tokens, i.e., append -1 to the end of the non-longest ones
        for indx, encoded_text in enumerate(encoded_texts):
            if len(encoded_text) < max_len:
                current_len = len(encoded_text)
                encoded_texts[indx].extend([50256] * (max_len - current_len))
                encoded_labels[indx].extend([-100] * (max_len - current_len))
        labels = torch.tensor(encoded_labels, device=self.device, dtype=torch.long)
        lm_inputs= torch.tensor(encoded_texts, device=self.device, dtype=torch.long)
        
        return lm_inputs, labels

    def hotflip_attack(self, averaged_grad, trigger_token_ids, increase_loss=False, num_candidates=100):
        averaged_grad = averaged_grad.cpu()
        embedding_matrix = self.embedding_weight.cpu()
        trigger_token_embeds = torch.nn.functional.embedding(torch.LongTensor(trigger_token_ids),
                embedding_matrix).detach()
        averaged_grad = averaged_grad.unsqueeze(0)
        gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik",
                (averaged_grad, embedding_matrix))        
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1    # lower versus increase the class probability.  if num_candidates > 1: # get top k options
            _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
            return best_k_ids.detach().cpu().numpy()[0]
        _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
        return best_at_each_step[0].detach().cpu().numpy()

    def find_triggers(self, target_texts):
        token_last_change = 0
        print(f"init_triggers:{self.tokenizer.decode(self.trigger_tokens)}")
        while token_last_change < self.trigger_tokens.shape[0]:
            for i, token_to_flip in enumerate(self.trigger_tokens):
                if i == 0: continue
                token_flipped = False
                #Find Candidates for this token
                self.model.zero_grad()
                lm_inputs, labels = self.make_target_batch(target_texts, self.trigger_tokens)
                loss = self.model(lm_inputs, labels=labels)[0]
                loss.backward()
                best_loss = loss.item()
                
                if self.target_model == 'gpt2':
                    averaged_grad = self.model.transformer.wte.weight.grad[token_to_flip].unsqueeze(0)
                elif self.target_model == 'gptj':
                    averaged_grad = self.model.transformer.wte.weight.grad[token_to_flip].unsqueeze(0)
                elif self.target_model == 'opt':
                    averaged_grad = self.model.model.decoder.embed_tokens.weight.grad[token_to_flip].unsqueeze(0)

                # Use hotflip (linear approximation) attack to get the top num_candidates
                candidates = self.hotflip_attack(averaged_grad, [token_to_flip], num_candidates=100)[0]
                for cand in candidates:
                    candidate_trigger_tokens = deepcopy(self.trigger_tokens)
                    candidate_trigger_tokens[i] = cand

                    self.model.zero_grad()
                    lm_inputs, labels = self.make_target_batch(target_texts, candidate_trigger_tokens)
                    loss = self.model(lm_inputs, labels=labels)[0]
                    if best_loss > loss.item():
                        token_flipped = True
                        best_loss = loss.item()
                        self.trigger_tokens[i] = cand
                if token_flipped:
                    token_last_change = 0
                    print(f"Loss: {best_loss}, triggers:{self.tokenizer.decode(self.trigger_tokens)}")
                elif token_last_change < self.trigger_tokens.shape[0]:
                    token_last_change += 1
                else:
                    print(f"\nNo improvement, ending iteration")
                    break

    def sample_sequence(self, target_texts, triggers=None, length=100):
        results = []
        if triggers is None: triggers = self.tokenizer.decode(self.trigger_tokens)
        for target_text in target_texts:
            text =': ' + target_text + triggers + ':'
            # text = target_text + triggers
            target_tokens = torch.tensor([self.tokenizer.encode(text)], device=self.device, dtype=torch.long)
            # target_length = target_tokens.shape[1]
            target_length = len(self.tokenizer.encode(target_text))+self.trigger_tokens.shape[0]
            past = None
            with torch.no_grad():
                generated_tokens = []
                for i in range(length):
                    outputs= self.model(target_tokens, past_key_values=past)
                    logits = outputs.logits[:, -1, :]
                    log_probs = torch.nn.functional.softmax(logits, dim=-1)
                    pred = torch.argmax(log_probs, keepdim=True)
                    target_tokens = torch.cat((target_tokens, pred), dim=1)
                    if pred.item() == self.trigger_tokens[0]:
                    # if pred.item() == 50256:
                        break
                    generated_tokens.append(pred.item())
                generated_sent = self.tokenizer.decode(generated_tokens)
                results.append({'context': target_text, 'generation':generated_sent})
        self.evaluate(results)
        import ipdb;ipdb.set_trace()
        return results

    def sentence_to_tokens(self, sentence):
        ret_tokens = [word for word, pos in pos_tag(word_tokenize(sentence)) 
                    if pos.startswith('N') or pos.startswith('J') or pos.startswith('V')]
        return ret_tokens

    def sentence_to_char(self, sentence):
        ret_chars = re.sub('[^a-zA-Z]', '', sentence)
        return ret_chars

    def evaluate(self, results, level='token'):
        count = 0
        for result in results:
            if level == 'token':
                target = self.sentence_to_tokens(result['context'])
                pred = self.sentence_to_tokens(result['generation'])
            if level == 'char':
                target = self.sentence_to_char(result['context'])
                pred = self.sentence_to_char(result['generation'])
            if pred == target: count += 1
            else: import ipdb; ipdb.set_trace()

        print(count/len(results))



