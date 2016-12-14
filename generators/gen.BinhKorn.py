import random
from deap import creator, base, tools, algorithms

import numpy as np

#problem parameters
membersize = 100 #length for each indivudual
poplen = 1000 #population dimension
min_val = 0 #minimum value used when generating random individuals
max_val = 5 #maximum value used when generating random individuals

NRUN = 5 #number of parallel indipendent runs
NGEN = 10 #number of generations for each separate run

verbose = True #debug messages

filename = "BinhKorn.dataset"
testfilename = "BinhKorn.testset"

#multi objective problem
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, min_val, max_val)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=membersize)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#fitness functions
def evaluate(individual):
    term1 = 0
    term2 = 0
    for elem in individual:
        term1 += 4*(elem**2)
        term2 += (elem - 5)**2
    return term1, term2

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

#population size
population = toolbox.population(n=poplen)

if(verbose):
    print("Starting algorithm runs")

#training generations
training = []
for n in range(NRUN):
    if(verbose):
        print("Run", n, "out of", NRUN)
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
    training.append(population)

#building training dataset
if(verbose):
    print("Building first part of training dataset...")
clfin = []
tops = []
for population in training:
    pareto = tools.sortNondominated(population, len(population))
    top = pareto[0] #the actual pareto front
    #using best and worst elements to build training data
    for member in top:
        #careful avoiding duplicates
        if member not in tops:
            tops.append(member)
        if member not in clfin:
            clfin.append(member)
    elems = len(clfin)
    for member in tools.selBest(population, k=len(population))[:-elems]:
        if member not in clfin:
            clfin.append(member)
        
##############
# running nsga

if(verbose):
    print("Starting nsga runs")

def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

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
for n in range(NRUN):
    if(verbose):
        print("NSGA run", n, "out of", NRUN)
    # Begin the generational process
    for gen in range(1, NGEN):
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
    training.append(pop)

#building training dataset
if(verbose):
    print("Building second part of training dataset...")
for population in training:
    pareto = tools.sortNondominated(population, len(population))
    top = pareto[0] #the actual pareto front
    #using best and worst elements to build training data
    for member in top:
        #careful avoiding duplicates
        if member not in tops:
            tops.append(member)
        if member not in clfin:
            clfin.append(member)
    elems = len(clfin)
    for member in tools.selBest(population, k=len(population))[:-elems]:
        if member not in clfin:
            clfin.append(member)

labels = []
ones = 0
zeros = 0

if(verbose):
    print("Shuffling dataset and setting labels")

np.random.shuffle(clfin)

for member in clfin:
    if member in tops:
        labels.append(1)
        ones += 1
    else:
        labels.append(0)
        zeros += 1

samples = len(clfin)

if(verbose):
    print("0:", zeros, "1:", ones)

filename += "." + str(samples) + "." + str(zeros) + "." + str(ones) + ".csv"

if(verbose):
    print("Writing dataset to", filename) 

with open(filename, 'w') as f:
    for index, individual in enumerate(clfin):
        for feature in individual:
            f.write(str(feature) + ", ")
        f.write(str(labels[index]) + "\n")

testfilename += "." + str(samples) + "." + str(zeros) + "." + str(ones) + ".csv"

if(verbose):
    print("Writing test dataset to", testfilename) 

with open(testfilename, 'w') as f:
    for index, individual in enumerate(clfin[:int(len(clfin)/2)]):
        for feature in individual:
            f.write(str(feature) + ", ")
        f.write(str(labels[index]) + "\n")
