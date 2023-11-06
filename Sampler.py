import torch
import csv
from ModelFactory import ModelFactory
from torchmetrics import ExtendedEditDistance, CatMetric
from torchmetrics.text import BLEUScore
from torchmetrics.functional.text import bleu_score
from util.template import TextTemplate
import re
from nltk import pos_tag, word_tokenize


class Sampler():
    def __init__(self, target_model='gptj', template=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_model = target_model
        self.template = TextTemplate(prefix_1='') if template is None else template
        modelFactory = ModelFactory()
        self.model = modelFactory.get_model(target_model)
        self.tokenizer = modelFactory.get_tokenizer(target_model)

    def sample_sequence(self, target_texts, triggers, length=50):
        results = []
        total_fail = 0
        if triggers is None: triggers = self.tokenizer.decode(self.trigger_tokens)
        for idx, target_text in enumerate(target_texts):
            text = target_text + self.template.format_trigger(triggers)

            target_tokens = self.tokenizer(text, return_tensors='pt').to(self.device)
            target_length = target_tokens.input_ids.shape[1]
            if target_length > 1000: continue
            with torch.no_grad():
                try:
                    if self.target_model == 'falcon':
                        gt = self.model.generate(target_tokens.input_ids, max_length = target_length*2+length, pad_token_id=self.tokenizer.eos_token_id, num_beams=3)
                    else:
                        gt = self.model.generate(**target_tokens, max_length=target_length*2+length, pad_token_id=self.tokenizer.eos_token_id, num_beams=3, temperature=0.9, top_p=0.6)
                    generation = self.tokenizer.decode(gt[0, target_length:])
                    generation = self.postprocess(generation, triggers)
                    results.append({'context': target_text, triggers:generation})
                    print(f'{idx=}\n{text=}\n{generation=}')
                    self.evaluate([{'context': target_text, triggers:generation}], level='substring')
                except RuntimeError as err:
                    print(f'{idx:} skip')
        return results
    
    def postprocess(self, text, triggers):
        ret = text
        sentences = []
        sentences_filtered = [self.sentence_to_char(self.template.format_trigger(triggers)), self.sentence_to_char('text:'+triggers)]
        text.replace('.', '\n')
        for t in text.split('\n')[:-1]:
            t_filtered = self.sentence_to_char(t.replace(self.template.prefix_trigger, ''))
            if t_filtered == '':continue
            if t_filtered not in sentences_filtered and t_filtered != '':
                if t_filtered in  ''.join(sentences_filtered): break
                sentences_filtered.append(t_filtered)
                sentences.append(t)
            # else: break
        if len(sentences)==0:
            ret = text.split('\n')
            ret = ret[1] if len(ret) > 1 else ret[0]
        else:
            ret = '.'.join(sentences)+'.'
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

    def evaluate(self, results, level='em'):
        metric = CatMetric()
        keys = list(results[0].keys())
        if level == 'em':
            for result in results:
                target_text = result[keys[0]]
                target = self.filter_tokens(target_text)
                pred = self.filter_tokens(result[keys[1]])
                if target == pred: 
                    metric.update(1)
                else: 
                    # import ipdb;ipdb.set_trace()
                    metric.update(0)
                mean = torch.mean(metric.compute())
            print(f"em Acc: {mean.item()}")
            return metric.compute()
        elif level == 'substring':
            for result in results:
                target_text = result[keys[0]]
                target = self.filter_tokens(target_text)
                pred = self.filter_tokens(result[keys[1]])
                if target in pred: 
                    metric.update(1)
                else: 
                    metric.update(0)
                mean = torch.mean(metric.compute())
            print(f"s Acc: {mean.item()}")
            return metric.compute()
        elif level == 'edit':
            EDD = ExtendedEditDistance()
            for result in results:
                target_text = result[keys[0]]
                dist = EDD([result[keys[1]]], [target_text])
                metric.update(dist)
            std, mean = torch.std_mean(metric.compute())
            print(f"edit distance mean: {mean.item()}, std: {std.item()}")
            return metric.compute()
        elif level == 'semantic':
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            for result in results:
                target_text = result[keys[0]]
                embedding_1= model.encode(result[keys[1]], convert_to_tensor=True)
                embedding_2 = model.encode(target_text, convert_to_tensor=True)

                sim = util.pytorch_cos_sim(embedding_1, embedding_2)
                metric.update(sim.to('cpu'))
            std, mean = torch.std_mean(metric.compute())
            print(f"semantic mean: {mean.item()}, std: {std.item()}")
            return metric.compute()
        elif level == 'bleu':
            for result in results:
                target_text = result[keys[0]]
                dist = bleu_score([result[keys[1]]], [target_text])
                # import ipdb;ipdb.set_trace()
                metric.update(1) if dist >= 0.6 else metric.update(0)
            std, mean = torch.std_mean(metric.compute())
            print(f"BLEU mean: {mean.item()}, std: {std.item()}")
            

    @staticmethod
    def save_to_csv(path, results, triggers):
        with open(path, 'w', newline='') as file:
            fieldnames = ['context', triggers]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
