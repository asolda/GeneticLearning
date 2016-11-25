# Super potato

## Informations
Author: Andrea Soldà, a.solda92@gmail.com

## What is this?
The purpose of this project is to show that a certain number of objective functions can be reduced to one neural network (NN) and that a properly trained NN can provide benefits when used to evaluate the fitness of individuals during the evolution process of a Genetic Algorithm (GA).

## How it works
### Step 1: Building the training set
An appropriate number of short runs of a multi objective GA is executed; each run is independent from the others and generates different solutions; each solution is then evaluated using the provided objective functions.
For each run all the Pareto Optimal solutions are gathered together, creating the first part of the training dataset. The rationale for this choice is that multiple run can provide different kind of solutions, with different configurations (especially for complex and difficult functions) and the classifier needs a wide range of different individuals to identify patterns in the best solutions.
Let n be the number of pareto optimal solutions for any given run; the n “worst” solutions are also gathered together to provide examples of what a “bad” solution looks like.
Optimal and worst solutions are then labeled, so that the classifier knows the mapping between a solution and its class: 0 is the label for bad solutions, 1 for optimal solutions.
Using this process a sufficiently large dataset is generated, and the learning process can take place.

### Step 2: Training the classifier
Is over the purpose of this document to go through the dynamics of how a classifier works; however, many different configurations are possible.
After the execution of this step, the classifier will be able to classify each solution as either a 0 or a 1. The classifier will also provide a confidence for each class; the confidence is the probability that the solution belongs to a certain class.

### Step 3: Creating a fitness function using the classifier
The fitness function for the new GA has the following form:

If prediction(individual) = 0 then fitness = 1 - confidence[0]

if prediction(individual) = 1 then fitness = confidence[1],

Where confidence[n] is the confidence score for the n-th class.
The GA will consequently be a single objective function algorithm, where the aim is to maximize the newly crafted objective function.

## Usage
First step is to generate a training set for the classifiers. You can do so using the generators in the "generators" folder.

Then you have to train a classifier. Different classifiers are provided in the "classifiers" folder. Eash classifier produces a model that will be used as a fitness function in a GA.

Pre-implemented genetic algorithms using the trained classifiers are available in the "algorithms" folder.  