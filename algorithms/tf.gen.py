import random
import sys
from deap import creator, base, tools, algorithms
import tensorflow as tf
import numpy as np
from keras.models import load_model

#problem parameters
membersize = 100 #length for each indivudual
poplen = 1000 #population dimension
min_val = 0 #minimum value used when generating random individuals
max_val = 5 #maximum value used when generating random individuals

NGEN = 100 #number of generations for each run

verbose = True

if len(sys.argv) != 2:
    print("Usage:", str(sys.argv[0]), "model.h5")
    exit()

modelfile = str(sys.argv[1])

model = load_model(modelfile)


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
    cl = model.predict(y)
    # one-hot encoded prediction
    if cl[0][0] == 1:
        return 1 - model.predict_proba(y, verbose = 0)[0][0],
    else:
        return model.predict_proba(y, verbose = 0)[0][1],

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

outfile = "tf.population." + str(NGEN) + ".csv"

if(verbose):
    print("Saving population to", outfile)

with open(outfile, 'w') as f:
    for elem in population:
        for value in elem[:len(elem) - 2]:
            f.write(str(value) + ', ')
        f.write(str(elem[len(elem) - 1]) + "\n")