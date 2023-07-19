from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, OPTForCausalLM, GPTJForCausalLM, AutoModelForCausalLM
import torch
import numpy as np
from copy import deepcopy
import re
from nltk import pos_tag, word_tokenize
import nltk
from torchmetrics import ExtendedEditDistance, CatMetric

class HotFlip:
    def __init__(self, trigger_token_length=6, target_model='gpt2', context_prefix='context:', trigger_prefix='input:'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_model = target_model
        if target_model == 'gptj':
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
            self.model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", device_map="auto", load_in_4bit=True).eval()
            self.vocab_size=50400
        elif target_model == 'opt':
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
            self.model = OPTForCausalLM.from_pretrained("facebook/opt-1.3b", device_map="auto", load_in_4bit=True).eval()
            self.vocab_size = 50272
        elif target_model == 'falcon':
            self.tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
            self.model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b-instruct", device_map="auto", load_in_4bit=True, trust_remote_code=True).eval()
            self.vocab_size = 65024

        self.embedding_weight = self.get_embedding_weight()
        self.add_hook()
        self.trigger_tokens = self.init_triggers(trigger_token_length)
        self.context_prefix = context_prefix
        self.trigger_prefix = trigger_prefix

    def init_triggers(self, trigger_token_length):
        triggers = np.empty(trigger_token_length, dtype=int)
        for idx, t in enumerate(triggers):
            t = np.random.randint(self.vocab_size)
            while re.search("[^a-zA-Z0-9s\s]", self.tokenizer.decode(t)):
                t = np.random.randint(self.vocab_size)
            triggers[idx] = t
        return triggers

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
            encoded_target_text = self.tokenizer.encode(target_text)
            encoded_trigger_prefix = self.tokenizer.encode(self.trigger_prefix)
            encoded_splash_n = self.tokenizer.encode('\n')
            if self.target_model == 'opt': 
                encoded_target = encoded_target_text[1:]
                encoded_trigger_prefix = encoded_trigger_prefix[1:]
                encoded_splash_n = encoded_splash_n[1:]
            else: encoded_target = encoded_target_text

            encoded_text = encoded_target_text + encoded_trigger_prefix + triggers.tolist() + encoded_splash_n + encoded_target
            len_non_label = len(encoded_target_text)+len(encoded_trigger_prefix) + triggers.shape[0] + len(encoded_splash_n)
            encoded_label = [-100]*len_non_label + encoded_target

            encoded_texts.append(encoded_text)
            encoded_labels.append(encoded_label)
            if len(encoded_text) > max_len:
                max_len = len(encoded_text)

        for indx, encoded_text in enumerate(encoded_texts):
            if len(encoded_text) < max_len:
                current_len = len(encoded_text)
                encoded_texts[indx].extend([self.tokenizer.eos_token_id] * (max_len - current_len))
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
                (averaged_grad.float(), embedding_matrix.float()))        
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
                # if i == 0: continue
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
                elif self.target_model == 'falcon':
                    averaged_grad = self.model.transformer.word_embeddings.weight.grad[token_to_flip].unsqueeze(0)

                # Use hotflip (linear approximation) attack to get the top num_candidates
                candidates = self.hotflip_attack(averaged_grad, [token_to_flip], num_candidates=100)[0]
                for cand in candidates:
                    if re.search("[^a-zA-Z0-9s\s]", self.tokenizer.decode(cand)):
                        continue
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
        for idx, target_text in enumerate(target_texts):
            text = target_text + self.trigger_prefix + triggers + '\n'
            target_tokens = torch.tensor([self.tokenizer.encode(text)], device=self.device, dtype=torch.long)
            target_length = len(self.tokenizer.encode(target_text))+self.trigger_tokens.shape[0]
            if target_length > 500: continue
            past = None
            with torch.no_grad():
                generated_tokens = []
                for i in range(length):
                    outputs= self.model(target_tokens, past_key_values=past)
                    logits = outputs.logits[:, -1, :]
                    log_probs = torch.nn.functional.softmax(logits, dim=-1)
                    pred = torch.argmax(log_probs, keepdim=True)
                    target_tokens = torch.cat((target_tokens, pred), dim=1)
                    pred_token = self.sentence_to_char(self.tokenizer.decode(pred.item()))
                    generated_tokens.append(pred.item())
                generation = self.tokenizer.decode(generated_tokens)
                generation = self.postprocess(generation)
                results.append({'context': target_text, 'generation':generation})
        return results
    
    def postprocess(self, text):
        ret = text
        for t in text.split(self.tokenizer.decode(self.trigger_tokens[0])):
            t = t.replace(self.context_prefix, '')
            t = t.replace(self.trigger_prefix, '')
            if t != '':
                ret = t
                break

        ret = self.context_prefix + ret
        print(ret)
        return ret

    def sentence_to_tokens(self, sentence):
        ret_tokens = [word for word, pos in pos_tag(word_tokenize(sentence), tagset='universal') if pos.startswith('N') or pos.startswith('A') or pos.startswith('V') or pos.startswith('X')]
        return ret_tokens

    def sentence_to_char(self, sentence):
        ret_chars = re.sub('[^a-zA-Z]', '', sentence.lower())
        return ret_chars
    
    def filter_tokens(self, sentence):
        ret_sentence = re.sub('[^a-zA-Z]', ' ', sentence.lower())
        filtered_sentence = ''.join(self.sentence_to_tokens(ret_sentence))

        return filtered_sentence

    def evaluate(self, results, prefix=None, level='char'):
        metric = CatMetric()
        if level == 'char':
            for result in results:
                target = self.filter_tokens(result['context'])
                pred = self.filter_tokens(result['generation'])
                if target in pred: 
                    metric.update(1)
                else: 
                    metric.update(0)
                mean = torch.mean(metric.compute())
            print(f"Acc: {mean.item()}")
        elif level == 'edit':
            EDD = ExtendedEditDistance()
            for result in results:
                dist = EDD([result['generation']], [result['context']])
                metric.update(dist)
            std, mean = torch.std_mean(metric.compute())
            import ipdb;ipdb.set_trace()
            print(f"edit distance mean: {mean.item()}, std: {std.item()}")
        elif level == 'semantic':
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            for result in results:
                embedding_1= model.encode(result['generation'], convert_to_tensor=True)
                embedding_2 = model.encode(result['context'], convert_to_tensor=True)

                sim = util.pytorch_cos_sim(embedding_1, embedding_2)
                metric.update(sim.to('cpu'))
            std, mean = torch.std_mean(metric.compute())
            import ipdb;ipdb.set_trace()
            print(f"semantic mean: {mean.item()}, std: {std.item()}")