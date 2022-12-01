# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


#SETTING UP MODEL

#Loading dataset and defining inputs and outputs
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
input_vars = dataset[:, 0:8]
output_vars = dataset[:,8]

# Models in Keras are defined as a sequence of layers...
# We continue to add on layers until we are satisified with our model
model = Sequential()
# Read https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/
# When needing to experiment with model types

#Dense class: fully connected layers


# define the keras model
model = Sequential()
# First layer expects 8 inputs and has 12 nodes, 'relu' aids with performance
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))

#This is the output layer
model.add(Dense(1, activation='sigmoid'))

# Make sure to compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# FITTING/TRAINING MODEL

# Each epoch is split into batches
model.fit(input_vars, output_vars, epochs=150, batch_size=10)