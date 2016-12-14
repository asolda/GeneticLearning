from __future__ import division
import sys
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
import numpy as np
import time
import pandas as pd

if len(sys.argv) != 4:
    print("Usage:", str(sys.argv[0]), "dataset.csv testset.csv prefix_to_model")
    exit()

# Datasets
TRAINING = str(sys.argv[1])
TEST = str(sys.argv[2])
MODEL = str(sys.argv[3])

# Load datasets
training_set = pd.read_csv(TRAINING, dtype=np.float, header = 0)
test_set = pd.read_csv(TEST, dtype=np.float, header = 0)

num_features = training_set.shape[1] - 1
num_labels = 2 #0 or 1

batch_size = 128
nb_epoch = 20

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

x_train = np.array(x_train)
x_test = np.array(x_test)

# Model creation
model = Sequential()
model.add(Dense(512, input_shape=(num_features,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model_file_name = MODEL + TRAINING.rsplit('/', 1)[-1] + '.h5'

print("Saving model to", model_file_name)

model.save(model_file_name)