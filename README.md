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

You can then evaluate the quality of the solution using the script inside the "evaluators" folder.

### Complete example

Let's say you want to use a DL classifier as a fitness function for the "single" objective function.

**Step 1**: Generate dataset
```
cd generators
python gen.single.py
```

You will read something like:
```
Starting algorithm runs
Run 0 out of 5
Run 1 out of 5
Run 2 out of 5
Run 3 out of 5
Run 4 out of 5
Building first part of training dataset...
Starting nsga runs
NSGA run 0 out of 5
NSGA run 1 out of 5
NSGA run 2 out of 5
NSGA run 3 out of 5
NSGA run 4 out of 5
Building second part of training dataset...
Shuffling dataset and setting labels
0: 992 1: 10
Writing dataset to single.dataset.1002.992.10.csv
Writing test dataset to single.testset.1002.992.10.csv
```

**Step 2**: Train the model. For the sake of order, create a new folder in the root of the project named `data` and move the datasets there:

From the root folder:
```
mkdir data
mv generators/single.dataset.1002.992.10.csv /data/
mv generators/single.testset.1002.992.10.csv /data/
mkdir models
```

Actual training:
```
cd classifiers
python3 tf.bmlp.clf.py ../data/single.dataset.1002.992.10.csv ../data/single.testset.1002.992.10.csv ../models
```

Output (truncated):
```
...

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
dense_1 (Dense)                  (None, 512)           51712       dense_input_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 512)           0           dense_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 512)           0           activation_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 512)           262656      dropout_1[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 512)           0           dense_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 512)           0           activation_2[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 2)             1026        dropout_2[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 2)             0           dense_3[0][0]
====================================================================================================
Total params: 315394
____________________________________________________________________________________________________
Train on 1000 samples, validate on 499 samples

...

Test accuracy: 1.0
Saving model to ../models/single.dataset.1002.992.10.csv.bmlp.h5
```

**Step 3**: Run GA.

From the root of your project:
```
cd algorithms
python tf.gen.py ../models/single.dataset.1002.992.10.csv.bmlp.h5
```

```
Classifier loaded
Gen: 0 out of 20
Gen: 1 out of 20
...
Gen: 19 out of 20
Saving population to tf.population.20.csv
```

**Step 4**: Evaluate population. Like we did in step 2, we create a new folder for the population.

From the root:
```
mkdir populations
mv algorithms/tf.population.20.csv populations/
```

We can now evaluate it:
```
cd evaluators
python single.eval.py ../populations/tf.population.20.csv skip
```

```
Classifier: -108.119238477
```