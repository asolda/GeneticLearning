import sys
import pandas as pd
import numpy as np

if(len(sys.argv) != 2):
    print('Usage:', str(sys.argv[0]), 'population.csv')

def evaluate(individual):
    term1 = 0
    term2 = 0
    for elem in individual:
        term1 += 4*(elem**2)
        term2 += (elem - 5)**2
    return term1, term2

def load_from_file(filename):
    population = pd.read_csv(filename, dtype=np.float, header = 0)
    population = population.as_matrix()
    return [individual[1::] for individual in population[1::]]

population = load_from_file(str(sys.argv[1]))

term1 = 0
term2 = 0
for member in population:
    values = evaluate(member)
    term1 += values[0]
    term2 += values[1]

mean1 = term1/len(population)
mean2 = term2/len(population)

print(mean1, mean2)