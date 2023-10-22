from Sampler import Sampler
import csv

results = []
with open('results/llama_Awesome_llama_12.csv', mode ='r') as file:    
    csvFile = csv.DictReader(file)
                
    for lines in csvFile:
        results.append(lines)
attack = Sampler()
attack.evaluate(results, level='em')
attack.evaluate(results, level='edit')
attack.evaluate(results, level='semantic')
# attack.evaluate(results, level='bleu')
