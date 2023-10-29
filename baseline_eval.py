import torch
import csv
from torchmetrics import ExtendedEditDistance, CatMetric
import re
from nltk import pos_tag, word_tokenize
import nltk

def sentence_to_tokens(sentence):
    ret_tokens = [word for word, pos in pos_tag(word_tokenize(sentence), tagset='universal') if pos.startswith('N') or pos.startswith('A') or pos.startswith('V') or pos.startswith('X')]
    return ret_tokens

def sentence_to_char(sentence):
    ret_chars = re.sub('[^a-zA-Z]', '', sentence.lower())
    return ret_chars

def filter_tokens(sentence):
    ret_sentence = re.sub('[^a-zA-Z]', ' ', sentence.lower())
    filtered_sentence = ''.join(sentence_to_tokens(ret_sentence))

    return filtered_sentence

def evaluate(results, level='char'):
    metric = CatMetric()
    keys = list(results[0].keys())
    if level == 'em':
        for result in results:
            target_text = result[keys[0]]
            target = filter_tokens(target_text)
            text = ''
            for i in range(1, 6): text+=result[keys[i]]
            pred = filter_tokens(text)
            if target == pred: 
                metric.update(1)
            else: 
                metric.update(0)
            mean = torch.mean(metric.compute())
        print(f"em Acc: {mean.item()}")
    elif level == 'edit':
        EDD = ExtendedEditDistance()
        for result in results:
            target_text = result[keys[0]]
            dists = [EDD([result[keys[i]]], [target_text]) for i in range(1, 6)]
            metric.update(min(dists))
        std, mean = torch.std_mean(metric.compute())
        print(f"edit distance mean: {mean.item()}, std: {std.item()}")
    elif level == 'semantic':
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        for result in results:
            target_text = result[keys[0]]
            embeddings_1= [model.encode(result[keys[i]], convert_to_tensor=True) for i in range(1,6)]
            embedding_2 = model.encode(target_text, convert_to_tensor=True)

            sims = [util.pytorch_cos_sim(embedding_1, embedding_2) for embedding_1 in embeddings_1]
            metric.update(max(sims).to('cpu'))
        std, mean = torch.std_mean(metric.compute())
        print(f"semantic mean: {mean.item()}, std: {std.item()}")

results = []
with open('results/llama_Tomatoes_baseline_1.csv', mode ='r') as file:    
    csvFile = csv.DictReader(file)
                
    for lines in csvFile:
        results.append(lines)
evaluate(results, level='em')
evaluate(results, level='edit')
evaluate(results, level='semantic')
