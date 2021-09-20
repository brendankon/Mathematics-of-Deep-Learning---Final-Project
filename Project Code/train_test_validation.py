#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 05:03:48 2021

@author: deepakd
"""

import tensorflow as tf
from sklearn.datasets import make_classification
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import random
import math
import numpy as np
import matplotlib.plt as plt
import pandas as pd
import sys
import tensorflow.keras.backend as K
import time
import pickle
import os

# Vary width of nework.  m=10000 is "overparametrized"
widths = [10,100,1000,10000]
n = 10000
matrices = {m:None for m in widths}
X, y = make_classification(n_samples=n, n_classes=2, random_state=1)
# determine the number of input features
n_features = X.shape[1]
metric_list = [tf.keras.metrics.BinaryAccuracy()]
count = 0
start_time = time.time()
for m in widths:
    ep = 2000 # epochs
    count += 1
    # define model
    model = Sequential()
    model.add(Dense(m, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    sgd = SGD(learning_rate=0.001, momentum=0.8)
    model.compile(optimizer=sgd, loss='binary_crossentropy',metrics=metric_list)
    # fit the model
    history = model.fit(X, y, epochs=ep, batch_size=32, verbose=1, validation_split=0.2)
    matrices[m] = history.history
    temp_time = time.time()-start_time
    file = open("train_test_validation_2/time/{0}_{1}s_n={2}_m={3}.txt".format(count, temp_time, n, m),"a")

# To store results
pickle.dump(matrices,open('train_test_validation_2/matrices.pkl','wb'))         

# plot learning curves
colors = {10:['red','orangered'],100:['blue','deepskyblue'],1000:['green','lime'],10000:['purple','magenta']}
linestyles = {m:['solid','dashed'] for m in widths}
ep = 2000 # # of epochs to plot up to

# Plot loss
plt.title('Loss over Time\n(n={0})'.format(n))
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
for m in widths:
    x1 = matrices[m]['loss'][:ep]
    x2 = matrices[m]['val_loss'][:ep]
    plt.plot(x1,label='train m={0}'.format(m), c=colors[m][0],linestyle=linestyles[m][0])
    plt.plot(x2, label='val m={0}'.format(m), c=colors[m][1],linestyle=linestyles[m][1])
plt.legend()
plt.savefig('train_test_validation_2/Loss_n={0}_ep={1}'.format(n,ep))
plt.show()
# Plot accuracy
plt.title('Accuracy over Time\n(n={0})'.format(n))
plt.xlabel('Epoch')
plt.ylabel('Sparse Categorical Accuracy')
for m in widths:
    x3 = matrices[m]['binary_accuracy'][:ep]
    x4 = matrices[m]['val_binary_accuracy'][:ep]
    plt.plot(x3, label='train m={0}'.format(m), c=colors[m][0],linestyle=linestyles[m][0])
    plt.plot(x4, label='val m={0}'.format(m), c=colors[m][1],linestyle=linestyles[m][1])
plt.legend()
plt.savefig('train_test_validation_2/Acc_n={0}_ep={1}'.format(n,ep))
plt.show()