import random
import sys
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from deap import creator, base, tools, algorithms

import numpy as np

if len(sys.argv) != 2:
    print("Usage:", str(sys.argv[0]), "model.pkl")
    exit()

modelfile = str(sys.argv[1])

#problem parameters
membersize = 100 #length for each indivudual
poplen = 1000 #population dimension
min_val = 0 #minimum value used when generating random individuals
max_val = 5 #maximum value used when generating random individuals

NGEN = 100 #number of generations for each run

verbose = True

clf = joblib.load(modelfile)

if(verbose):
    print("Classifier loaded")

#single objective function algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, min_val, max_val)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=membersize)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#fitness function based on classifier
def nneval(individual):
    y = np.array(individual).reshape(1, -1)
    cl = clf.predict(y)
    if cl[0] == 0:
        return 1 - clf.predict_proba(y)[0][0],
    else:
        return clf.predict_proba(y)[0][1],

toolbox.register("evaluate", nneval)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=poplen)

#performing genetic algorithm with classifier as fitness function
for gen in range(NGEN):
        if(verbose):
            print("Gen:", gen, "out of", NGEN)
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))

outfile = "sk.population." + str(NGEN) + ".csv"

if(verbose):
    print("Saving population to", outfile)

with open(outfile, 'w') as f:
    for param in population[0]:
        f.write(", ")
    f.write("Fitness value\n")
    for member in population:
        for value in member:
            f.write(str(value) + ", ")
        f.write(str(nneval(member)) + "\n")