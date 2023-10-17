from Sampler import Sampler
import csv

results = []
with open('results/falcon_Financial_falcon_3_shots_12.csv', mode ='r') as file:    
    csvFile = csv.DictReader(file)
                
    for lines in csvFile:
        results.append(lines)
attack = Sampler()
attack.evaluate(results, level='em')
attack.evaluate(results, level='edit')
attack.evaluate(results, level='semantic')
