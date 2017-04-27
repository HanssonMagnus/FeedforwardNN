#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 19:10:47 2017

@author: magnus
"""
import pydot
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
from keras.utils import plot_model

# For reproducibility.
np.random.seed(123)

# Generate data.
steps = 0.001
x = np.array([np.arange(0, 4*np.pi, steps)])

# Reshape input data from x.shape = (1, 1257) to x.shape = (1257, 1).
x = x.reshape(x.shape[1],1)

# Creating target data.
y = np.sin(x)


# Split the data into training and testing dataset.
train_size = int(len(x)*0.67)
test_size = len(x) - train_size

x_train = x[0:train_size]
y_train = y[0:train_size]

x_test = x[train_size:len(x)]
y_test = y[train_size:len(y)]


# Construct network model.
model = Sequential()
model.add(Dense(6, activation="relu", input_dim=1))


model.add(Dense(6, activation="relu", input_dim=6))

model.add(Dense(6, activation="relu", input_dim=6))
model.add(Dense(6, activation="relu", input_dim=6))
model.add(Dense(6, activation="relu", input_dim=6))
'''
model.add(Dense(6, activation="relu", input_dim=6))
model.add(Dense(6, activation="relu", input_dim=6))
model.add(Dense(6, activation="relu", input_dim=6))
'''
model.add(Dense(1)) # If you don't specify activation it becomes linear.
model.compile(loss = 'mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, nb_epoch=100, batch_size=100, verbose=0)

# Make predications.
train_predict = model.predict(x_train)
test_predict = model.predict(x_test)

# Shift train data predictions for plotting.
train_predict_plot = np.empty_like(x)
train_predict_plot[:, :] = np.nan
train_predict_plot[0:len(train_predict), :] = train_predict

test_predict_plot = np.empty_like(x)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict):len(x), :] = test_predict

# Plot.
plt.plot(y)
plt.plot(train_predict_plot)
plt.plot(test_predict_plot)
plt.show()

# Plot network graph.
#plot_model(model, to_file='model.png')



