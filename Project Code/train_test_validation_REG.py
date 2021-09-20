#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 02:05:24 2021

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
import matplotlib.pyplot as plt
import pandas as pd
import sys
import tensorflow.keras.backend as K
import time
import pickle
import os

m = 10000
n = 10000
regs = [0,0.01,0.1]
ls = ['L1','L2']
matrices = {l:{r:None for r in regs} for l in ls} # Dictionary to store model history
X, y = make_classification(n_samples=n, n_classes=2, random_state=1)
# determine the number of input features
n_features = X.shape[1]
metric_list = [tf.keras.metrics.BinaryAccuracy()]
count = 0
start_time = time.time() 
for l in ls:
    for r in regs:
        # ep = 1000
        ep = 100
        count += 1
        # define model
        model = Sequential()
        if l == 'L1': # Choose correct regularizer
            l_reg = tf.keras.regularizers.L1(r)
        elif l == 'L2':
            l_reg = tf.keras.regularizers.L2(r)
        # l2_reg = tf.keras.regularizers.L2(r)
        model.add(Dense(m, activation='relu', kernel_initializer='he_normal', kernel_regularizer = l_reg, input_shape=(n_features,)))
        model.add(Dense(1, activation='sigmoid'))
        # compile the model
        sgd = SGD(learning_rate=0.001, momentum=0.8)
        model.compile(optimizer=sgd, 
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=metric_list)
        # fit the model
        history = model.fit(X, y, epochs=ep, batch_size=32, verbose=1, validation_split=0.2)
        matrices[l][r] = history.history
        temp_time = int(time.time()-start_time)
        file = open("train_test_validation_4/time/{0}_{1}s_ep{2}_l={3}_r={4}.txt".format(count, temp_time, ep, l, r),"a")

# Next 5 lines were just for storing variables.
pickle.dump(matrices,open('train_test_validation_4/matrices.pkl','wb'))         
matrices = pickle.load(open('train_test_validation_4/matrices_OLD.pkl','rb'))
# plot learning curves
colors = {0:['red','orangered'],0.01:['blue','deepskyblue'],0.1:['green','lime']}
linestyles = {r:['solid','dashed'] for r in regs}

# Some of our plotting code, slightly different from what produced some of our graphs.
ep = [50,500,500] # # of epochs to plot up to
ep2 = [100,550,1000]
regs = [0,0.01]
# regs = [0.01]
for i, e in enumerate(ep):
    for l in ls:
        plt.title('Loss over Time\n({0}-m={1}, n={2})'.format(l,m,n))
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        for r in regs:
            print(e,ep2[i])
            y1 = matrices[l][r]['loss'][e:ep2[i]]
            y2 = matrices[l][r]['val_loss'][e:ep2[i]]
            x_dim = np.arange(e,ep2[i])
            plt.plot(x_dim, y1,label='train {0}_r={1}'.format(l,r), c=colors[r][0],linestyle=linestyles[r][0])
            plt.plot(x_dim,y2, label='val {0}_r={1}'.format(l,r), c=colors[r][1],linestyle=linestyles[r][1])
        plt.legend()
        plt.savefig('train_test_validation_4/Loss_{0}_EPR={1}-{2}'.format(l,e,ep2[i]))
        plt.show()
    
    
    for l in ls:
        plt.title('Accuracy over Time\n({0}-m={1}, n={2})'.format(l,m,n))
        plt.xlabel('Epoch')
        plt.ylabel('Binary Accuracy')
        for r in regs:
            print(e,ep2[i])
            y3 = matrices[l][r]['binary_accuracy'][e:ep2[i]]
            y4 = matrices[l][r]['val_binary_accuracy'][e:ep2[i]]
            x_dim = np.arange(e,ep2[i])
            plt.plot(x_dim,y3,label='train {0}_r={1}'.format(l,r), c=colors[r][0],linestyle=linestyles[r][0])
            plt.plot(x_dim, y4, label='val {0}_r={1}'.format(l,r), c=colors[r][1],linestyle=linestyles[r][1])
        plt.legend()
        plt.savefig('train_test_validation_4/Acc_{0}_EPR={1}-{2}'.format(l,e,ep2[i]))
        plt.show()
