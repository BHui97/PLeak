from Sampler import Sampler
import csv

results = []
with open('results/llama_SIQA_llama_16.csv', mode ='r') as file:    
    csvFile = csv.DictReader(file)
                
    for lines in csvFile:
        trigger = [*lines][1]
        # text = lines[trigger].split('context:')
        # lines[trigger] = 'context:' + text[1] if len(text) > 1 else text[0]
        # text =  lines[trigger].split(trigger[:10])
        # lines[trigger] = text[0]
        results.append(lines)
attack = Sampler()
attack.evaluate(results, level='em')
attack.evaluate(results, level='edit')
attack.evaluate(results, level='semantic')
# attack.evaluate(results, level='bleu')
