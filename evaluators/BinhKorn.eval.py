import sys
import random
import pandas as pd
import numpy as np
from deap import creator, base, tools, algorithms

if(len(sys.argv) < 2 or len(sys.argv) > 3):
    print('Usage:', str(sys.argv[0]), 'population.csv (optional) skip (to skip nsga)')
    sys.exit()

#problem parameters
membersize = 100 #length for each indivudual
poplen = 1000 #population dimension
min_val = 0 #minimum value used when generating random individuals
max_val = 5 #maximum value used when generating random individuals

NGEN = 10

verbose = True

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

if(len(sys.argv) == 3):
    print("Classifier:", mean1, mean2)
    sys.exit()

if(verbose):
    print("Starting nsga runs")

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

#multi objective problem
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

toolbox.register("attr_float", uniform, min_val, max_val, membersize)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=min_val, up=max_val, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=min_val, up=max_val, eta=20.0, indpb=1.0/membersize)
toolbox.register("select", tools.selNSGA2)

pop = toolbox.population(n=poplen)

pop = toolbox.select(pop, len(pop))

CXPB = 0.9

training = []

# Begin the generational process
for gen in range(1, NGEN):
    if(verbose):
        print("Run", gen, "out of", NGEN)
    # Vary the population
    offspring = tools.selTournamentDCD(pop, len(pop))
    offspring = [toolbox.clone(ind) for ind in offspring]

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        if random.random() <= CXPB:
            toolbox.mate(ind1, ind2)
        toolbox.mutate(ind1)
        toolbox.mutate(ind2)
        del ind1.fitness.values, ind2.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Select the next generation population
    pop = toolbox.select(pop + offspring, poplen)

mean3 = 0
mean4 = 0

term1 = 0
term2 = 0
for member in pop:
    values = evaluate(member)
    term1 += values[0]
    term2 += values[1]

mean3 = term1/len(population)
mean4 = term2/len(population)

print("Classifier:", mean1, mean2)
print("NSGA:", mean3, mean4)