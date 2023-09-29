from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, OPTForCausalLM, GPTJForCausalLM, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import numpy as np
from copy import deepcopy
import re
from nltk import pos_tag, word_tokenize
import nltk
from util.template import TextTemplate
from torchmetrics import ExtendedEditDistance, CatMetric
from fastchat.model import get_conversation_template
from util.data import Harmful

class HotFlip:
    def __init__(self, trigger_token_length=6, target_model='gpt2', template=None, init_triggers=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_model = target_model
        self.template = TextTemplate(prefix_1='') if template is None else template
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                               bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4')

        if target_model == 'gptj':
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
            self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b", device_map="auto", load_in_4bit=True, torch_dtype=torch.bfloat16, quantization_config=quantization_config).eval()
            self.vocab_size=50400
        elif target_model == 'gpt2':
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2", device_map="auto", load_in_4bit=True, torch_dtype=torch.bfloat16, quantization_config=quantization_config).eval()
            self.vocab_size=50257
        elif target_model == 'opt':
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-6.7B")
            self.model = OPTForCausalLM.from_pretrained("facebook/opt-6.7B", device_map="auto", load_in_4bit=True, torch_dtype=torch.bfloat16, quantization_config=quantization_config).eval()
            self.vocab_size = 50272
        elif target_model == 'llama':
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
            self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-hf", device_map="auto", load_in_4bit=True, torch_dtype=torch.bfloat16, quantization_config=quantization_config).eval()
            self.vocab_size = 32000
        # Load model directly

        self.embedding_weight = self.get_embedding_weight()
        self.add_hook()
        self.trigger_tokens = self.init_triggers(trigger_token_length) if init_triggers==None else np.array(self.tokenizer.encode(init_triggers), dtype=int)
        self.user_prefix = ""
        self.instruction = []

    def init_triggers(self, trigger_token_length):
        triggers = np.empty(trigger_token_length, dtype=int)
        for idx, t in enumerate(triggers):
            t = np.random.randint(self.vocab_size)
            while re.search("[^a-zA-Z0-9s\s]", self.tokenizer.decode(t)):
                t = np.random.randint(self.vocab_size)
            triggers[idx] = t
        return triggers
    
    def init_instruction(self, num_text):
        # for i in range(num_text):
        #     instruction = ''
        #     length = np.random.randint(20, 50)
        #     for i in range(length):
        #         t = np.random.randint(self.vocab_size)
        #         instruction += self.tokenizer.decode(t) + ' '
        #     self.instruction.append(instruction)
        self.instruction = Harmful(True, num_text, prefix_1='')


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

    def make_target(self, index, target_text, triggers):
        # encode items and get the max length
        target_text =  'instruction:' + self.instruction[index] + '\n\n' + target_text
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

    def hotflip_attack(self, averaged_grad, increase_loss=False, num_candidates=100):
        averaged_grad = averaged_grad
        embedding_matrix = self.embedding_weight
        averaged_grad = averaged_grad.unsqueeze(0)
        gradient_dot_embedding_matrix = torch.einsum("bij,kj->bik",
                (averaged_grad, embedding_matrix))        
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1    # lower versus increase the class probability.  if num_candidates > 1: # get top k options
            _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=2)
            return best_k_ids.detach().squeeze().cpu().numpy()
        _, best_at_each_step = gradient_dot_embedding_matrix.max(2)
        return best_at_each_step[0].detach().cpu().numpy()

    def replace_triggers(self, target_texts):
        token_flipped = True
        self.init_instruction(len(target_texts))
        print(f"init_triggers:{self.tokenizer.decode(self.trigger_tokens)}")
        while token_flipped:
            token_flipped = False
            with torch.set_grad_enabled(True):
                self.model.zero_grad()
                best_loss = self.compute_loss(target_texts, self.trigger_tokens, require_grad=True)
            print(f"current loss:{best_loss}")
            
            if self.target_model == 'gpt2':
                averaged_grad = self.model.transformer.wte.weight.grad[self.trigger_tokens]
            elif self.target_model == 'gptj':
                averaged_grad = self.model.transformer.wte.weight.grad[self.trigger_tokens]
            elif self.target_model == 'opt':
                averaged_grad = self.model.model.decoder.embed_tokens.weight.grad[self.trigger_tokens]
            elif self.target_model == 'llama' or 'vicuna':
                averaged_grad = self.model.model.embed_tokens.weight.grad[self.trigger_tokens]

            # Use hotflip (linear approximation) attack to get the top num_candidates
            candidates = self.hotflip_attack(averaged_grad, num_candidates=100)
            best_trigger_tokens = deepcopy(self.trigger_tokens)
            for i, token_to_flip in enumerate(self.trigger_tokens):
                for cand in candidates[i]:
                    if re.search("[^a-zA-Z0-9s\s]", self.tokenizer.decode(cand)):
                        continue
                    candidate_trigger_tokens = deepcopy(self.trigger_tokens)
                    candidate_trigger_tokens[i] = cand

                    self.model.zero_grad()
                    with torch.no_grad():
                        loss = self.compute_loss(target_texts, candidate_trigger_tokens, require_grad=False)
                    if best_loss > loss:
                        token_flipped = True
                        best_loss = loss
                        best_trigger_tokens = deepcopy(candidate_trigger_tokens)
            self.trigger_tokens = deepcopy(best_trigger_tokens)
            if token_flipped:
                print(f"Loss: {best_loss}, triggers:{self.tokenizer.decode(self.trigger_tokens)}")
            else:
                print(f"\nNo improvement, ending iteration")

    def compute_loss(self, target_texts, trigger_tokens, require_grad=False):
        total_loss = 0
        for index, text in enumerate(target_texts):
            lm_input, label = self.make_target(index, text, trigger_tokens) 
            loss = self.model(lm_input, labels=label)[0]/len(target_texts)
            total_loss += loss.item()
            if require_grad:
                loss.backward()
        return total_loss

    def sample_sequence(self, target_texts, triggers=None, length=333):
        results = []
        total_fail = 0
        if triggers is None: triggers = self.tokenizer.decode(self.trigger_tokens)
        for idx, target_text in enumerate(target_texts):
            text = target_text + self.template.format_trigger(triggers)
            target_tokens = self.tokenizer(text, return_tensors='pt').to(self.device)
            target_length = target_tokens.input_ids.shape[1]
            if target_length > 500: continue
            with torch.no_grad():
                gt = self.model.generate(**target_tokens, max_length=length, pad_token_id=self.tokenizer.eos_token_id, num_beams=3)
                generation = self.tokenizer.decode(gt[0, target_length:])
                generation = self.postprocess(generation, triggers)
                target = self.filter_tokens(target_text)
                pred = self.filter_tokens(generation)
                if target not in pred:
                    total_fail+=1
                    print(total_fail, idx)

                print(target_text, generation)
                results.append({'context': target_text, 'generation':generation})
        return results
    
    def postprocess(self, text, triggers):
        ret = text
        sentences = []
        sentences_filtered = [self.sentence_to_char(self.template.format_trigger(triggers))]
        for t in text.split('\n'):
            t_filtered = self.sentence_to_char(t.replace(self.template.prefix_trigger, ''))
            if t_filtered=='':
                continue
            elif t_filtered not in sentences_filtered and t_filtered != '':
                if t_filtered not in  ''.join(sentences_filtered):
                    sentences_filtered.append(t_filtered)
                    sentences.append(t)
            else:
                break
        if len(sentences)==0:
            ret = text.split('\n')
            if len(ret)>1: ret = ret[1]
            else: ret = ret[0]
        else:
            ret = ''.join(sentences)
        ret = ret.replace(self.tokenizer.eos_token, '')
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

    def evaluate(self, results, level='char'):
        metric = CatMetric()
        if level == 'em':
            for result in results:
                target_text = result['context']
                target = self.filter_tokens(target_text)
                pred = self.filter_tokens(result['generation'])
                if target in pred: 
                    metric.update(1)
                else: 
                    metric.update(0)
                    # import ipdb;ipdb.set_trace()
                mean = torch.mean(metric.compute())
            print(f"em Acc: {mean.item()}")
        elif level == 'edit':
            EDD = ExtendedEditDistance()
            for result in results:
                target_text = result['context']
                dist = EDD([result['generation']], [target_text])
                metric.update(dist)
            std, mean = torch.std_mean(metric.compute())
            print(f"edit distance mean: {mean.item()}, std: {std.item()}")
        elif level == 'semantic':
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            for result in results:
                target_text = result['context']
                embedding_1= model.encode(result['generation'], convert_to_tensor=True)
                embedding_2 = model.encode(target_text, convert_to_tensor=True)

                sim = util.pytorch_cos_sim(embedding_1, embedding_2)
                metric.update(sim.to('cpu'))
            std, mean = torch.std_mean(metric.compute())
            print(f"semantic mean: {mean.item()}, std: {std.item()}")
