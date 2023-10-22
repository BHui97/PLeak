import csv
f = open("roles.txt", "r")

with open('roles.csv', 'w', newline='') as file:
    fieldnames = ['roles', 'instruction']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    for x in f:
        if x == '\n': continue
        x = x.split(' - ')
        result = {fieldnames[0]:x[0],  fieldnames[1]:x[1][:-1]}
        writer.writerow(result)
