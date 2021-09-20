#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 20:10:07 2021

@author: deepakd
"""

# example of plotting learning curves
import tensorflow as tf
from sklearn.datasets import make_classification
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import matplotlib
from sklearn.model_selection import KFold
import numpy as np
# create the dataset
m = 10000
n = 10000
trials = {i:None for i in range(20)} # Store results per trial.
X, y = make_classification(n_samples=n, n_classes=2, random_state=1)
kf = KFold(n_splits=20) # Split data into 20 train-test splits
# determine the number of input features
n_features = X.shape[1]
count = 0
# define model
for train_index, test_index in kf.split(X): # Validate 20 times
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    metric_list = [tf.keras.metrics.BinaryAccuracy()]
    model = Sequential()
    model.add(Dense(m, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    sgd = SGD(learning_rate=0.001, momentum=0.8)
    model.compile(optimizer=sgd, loss='binary_crossentropy',metrics=metric_list) 
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1) # Removed to see over epochs
    # fit the model
    # history = model.fit(X_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_data=(X_test, y_test), callbacks=[es])
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_data=(X_test, y_test))
    trials[count] = history.history
    count += 1

# Store results for plotting later
max_gen_error = list()

for k in trials.keys():
    train_acc = np.array(trials[k]['binary_accuracy'])
    val_acc = np.array(trials[k]['val_binary_accuracy'])
    gen_error = val_acc-train_acc
    max_gen_err = gen_error.max()
    max_gen_error.append(max_gen_err)
    
# plot curves for maximum loss
trial_ax = [i for i in range(len(trials))]
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.plot(max_gen_error,label='train-val error')
ax.set_title('Max Generalization Error over Trials (n={0}, m={1})'.format(n,m))
ax.set_xlabel('Trials')
ax.set_ylabel('Max Gen Error (train - val)')
ax.set_xticks([i for i in range(len(trials))])
plt.legend()
plt.savefig('cross_val_error/gen_error_early_stopping')
plt.show()

last_gen_error = list()
for k in trials.keys():
    train_acc = np.array(trials[k]['binary_accuracy'])
    val_acc = np.array(trials[k]['val_binary_accuracy'])
    gen_error = val_acc-train_acc
    last_gen_err = abs(gen_error[-1])
    last_gen_error.append(last_gen_err)
    
# plot curves for last loss
trial_ax = [i for i in range(len(trials))]
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.plot(last_gen_error,label='train-val error')
ax.set_title('Last Generalization Error over Trials (n={0}, m={1})'.format(n,m))
ax.set_xlabel('Trials')
ax.set_ylabel('Last Gen Error (train - test)')
ax.set_xticks([i for i in range(len(trials))])
plt.legend()
plt.savefig('cross_val_error/last_gen_error')
plt.show()

# Used to find VC dimension based generalization error probabilty, not used in plotting
def gen_error_func(N,D,eta):
    llog = np.log(2*N/D)
    rlog = np.log(eta/4)
    lipar = D*(llog+1)
    brack = lipar-rlog
    argu = brack/N
    print(argu)
    err = np.sqrt(argu)
    return err
    