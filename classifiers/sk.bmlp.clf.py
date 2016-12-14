import sys
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import numpy as np
import pandas as pd

if len(sys.argv) != 3:
    print("Usage:", str(sys.argv[0]), "dataset.csv testset.csv")
    exit()

# Datasets
TRAINING = str(sys.argv[1])
TEST = str(sys.argv[2])

# Load datasets
training_set = pd.read_csv(TRAINING, dtype=np.float, header = 0)
test_set = pd.read_csv(TEST, dtype=np.float, header = 0)

num_features = training_set.shape[1] - 1
num_labels = 2 #0 or 1

training_set = training_set.as_matrix()
test_set = test_set.as_matrix()

x_train = [sample[1:num_features+1] for sample in training_set[1::]]
x_test = [sample[1:num_features+1] for sample in test_set[1::]]

y_train = [sample[-1] for sample in training_set[1::]]
y_test = [sample[-1] for sample in test_set[1::]]

x_train = np.array(x_train).reshape(len(x_train), -1)
x_test = np.array(x_test).reshape(len(x_test), -1)

#creating new classifier
clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(200, 400, 100), random_state=1)

#fitting classifier
clf.fit(x_train, y_train)

#testing our network
expZeros = 0
correctZeros = 0
expOnes = 0
correctOnes = 0

for index, testCase in enumerate(x_test):
    result = clf.predict(np.array(testCase).reshape(1, -1))
    expected = y_test[index]
    #print("Prediction: ", result, "Expected:", expected)
    if expected == 0:
        expZeros += 1
        if result[0] == expected:
            correctZeros += 1
    else:
        expOnes += 1
        if result[0] == expected:
            correctOnes += 1

print("0:", correctZeros, "out of", expZeros)
print("1:", correctOnes, "out of", expOnes)
print("Accuracy on 0:", "{0:.10f}".format(correctZeros/expZeros))
print("Accuracy on 1:", "{0:.10f}".format(correctOnes/expOnes))

# Saving on disk
modelfile = TRAINING + ".sk.bmlp.pkl"
print("Saving model to", modelfile)
joblib.dump(clf, modelfile) 