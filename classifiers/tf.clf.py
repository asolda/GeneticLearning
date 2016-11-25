from __future__ import division
import sys
import tensorflow as tf
import numpy as np
import time
import pandas as pd

if len(sys.argv) != 3:
    print("Usage:", str(sys.argv[0]), "dataset.csv testset.csv")
    exit()

# Datasets
TRAINING = str(sys.argv[1])
TEST = str(sys.argv[2])

# Load datasets
training_set = pd.read_csv(TRAINING, dtype=np.int, header = 0)
test_set = pd.read_csv(TEST, dtype=np.int, header = 0)

num_features = training_set.shape[1] - 1
num_labels = 2 #0 or 1

training_set = training_set.as_matrix()
test_set = test_set.as_matrix()

x_train = [sample[1:num_features+1] for sample in training_set[1::]]
x_test = [sample[1:num_features+1] for sample in test_set[1::]]

y_train = [sample[-1] for sample in training_set[1::]]
y_test = [sample[-1] for sample in test_set[1::]]

#represents labels array as a one-hot array
def toOneHot(list):
    onehot = []
    for item in list:
        if item == 0:
            onehot.append([1, 0])
        else:
            onehot.append([0, 1])
    return onehot

y_train = toOneHot(y_train)
y_test = toOneHot(y_test)

# TRAINING SESSION PARAMETERS
# number of times we iterate through training data
numEpochs = 27000
# a smarter learning rate for gradientOptimizer
learningRate = tf.train.exponential_decay(
    learning_rate=0.0008,
    global_step= 1,
    decay_steps=training_set.shape[0],
    decay_rate= 0.95,
    staircase=True
)

#placeholders
X = tf.placeholder(tf.float32 , [None, num_features])
yGold = tf.placeholder(tf.float32, [None, num_labels])

# Values are randomly sampled from a Gaussian with a standard deviation of:
#     sqrt(6 / (numInputNodes + numOutputNodes + 1))

weights = tf.Variable(tf.random_normal(
    [num_features, num_labels],
    mean=0,
    stddev=(np.sqrt(6/num_features + num_labels+1)),
    name="weights"
))

bias = tf.Variable(tf.random_normal(
    [1,num_labels],
    mean=0,
    stddev=(np.sqrt(6/num_features+num_labels+1)),
    name="bias"
))                      

# INITIALIZE our weights and biases
init_OP = tf.initialize_all_variables()

# PREDICTION ALGORITHM i.e. FEEDFORWARD ALGORITHM
apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias") 
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")

# COST FUNCTION i.e. MEAN SQUARED ERROR
cost_OP = tf.nn.l2_loss(activation_OP-yGold, name="squared_error_cost")


# OPTIMIZATION ALGORITHM i.e. GRADIENT DESCENT
training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)

# Create a tensorflow session
sess = tf.Session()

# Initialize all tensorflow variables
sess.run(init_OP)

# argmax(activation_OP, 1) gives the label our model thought was most likely
# argmax(yGold, 1) is the correct label
correct_predictions_OP = tf.equal(tf.argmax(activation_OP,1),tf.argmax(yGold,1))
# False is 0 and True is 1, what was our average?
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))
# Summary op for regression output
activation_summary_OP = tf.histogram_summary("output", activation_OP)
# Summary op for accuracy
accuracy_summary_OP = tf.scalar_summary("accuracy", accuracy_OP)
# Summary op for cost
cost_summary_OP = tf.scalar_summary("cost", cost_OP)
# Summary ops to check how variables (W, b) are updating after each iteration
weightSummary = tf.histogram_summary("weights", weights.eval(session=sess))
biasSummary = tf.histogram_summary("biases", bias.eval(session=sess))
# Merge all summaries
all_summary_OPS = tf.merge_all_summaries()
# Summary writer
writer = tf.train.SummaryWriter("summary_logs", sess.graph)

# Initialize reporting variables
cost = 0
diff = 1

# Training epochs
for i in range(numEpochs):
    if i > 1 and diff < .0001:
        print("change in cost %g; convergence."%diff)
        break
    else:
        # Run training step
        step = sess.run(training_OP, feed_dict={X: x_train, yGold: y_train})
        # Report occasional stats
        if i % 10 == 0:
            # Generate accuracy stats on test data
            summary_results, train_accuracy, newCost = sess.run(
                [all_summary_OPS, accuracy_OP, cost_OP], 
                feed_dict={X: x_train, yGold: y_train}
            )
            # Write summary stats to writer
            writer.add_summary(summary_results, i)
            # Re-assign values for variables
            diff = abs(newCost - cost)
            cost = newCost

            #generate print statements
            print("step %d, training accuracy %g"%(i, train_accuracy))
            print("step %d, cost %g"%(i, newCost))
            print("step %d, change in cost %g"%(i, diff))

# Using test data to find out accuracy
print("final accuracy on test set: %s" %str(sess.run(
    accuracy_OP, 
    feed_dict={X: x_test, yGold: y_test})))

modelfile = TRAINING + ".tfmodel.ckpt"
print("Saving model to", modelfile)
# Create Saver
saver = tf.train.Saver()
# Save variables to .ckpt file
saver.save(sess, modelfile)

# Close tensorflow session
sess.close()