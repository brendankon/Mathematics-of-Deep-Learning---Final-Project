#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 17:11:37 2021

@author: deepakd
"""

import tensorflow as tf
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

#Activation for scaling network output (LAZY TRAINING)
class ScaledActivation:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def scaled_output(self,x):
        return x * self.alpha
        
class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, learning_rate=.0005):
        self.weights_dict = {}
        self.lr_dict = {}
        self.learning_rate = learning_rate
    
    # Function which gets called at the end of each training epoch
    def on_epoch_end(self, epoch, logs=None):
        self.weights_dict.update({epoch:self.model.get_weights()})
        
        for var in self.model.optimizer.variables():
            if 'kernel/accumulator' in var.name:
                self.lr_dict.update({epoch:self.learning_rate/tf.sqrt(var.numpy() + 1e-7)}) #Effective learning rate from adagrad
            if 'kernel/rms' in var.name:
                self.lr_dict.update({epoch:self.learning_rate/tf.sqrt(var.numpy() + 1e-7)}) #Effective learning rate from rmsprop

        #print(self.model.optimizer.variables())


# Display plot of weight changes
def plotWeightChanges(title, epochs_matrix, norm_matrix, m_arr, alphas_arr):
    colors = ['black', 'red', 'green', 'blue', 'yellow', 'orange']
    linestyles = ['solid','solid','solid','dashed','dashed','dashed']
    plt.figure(figsize=(7,5))
    for i in range(len(norm_matrix)):
        m_label = "m=" + str(m_arr[i]) + ', alpha=' + str(alphas_arr[i]) 
        plt.plot(epochs_matrix[i], norm_matrix[i], label=m_label, c=colors[i],linestyle=linestyles[i])

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Maximum Distances")
    plt.title("scaled_loss_analysis.py: \nWeight Changes over Epochs (Optimizer: " + title + ")")
    plt.savefig('Plots_scaling_loss_analysis/Weight_Changes_{0}'.format(title))
    plt.show()
# Plot learning rate changes
def plotLearningRateChanges(title, epochs_matrix, lr_matrix, m_arr):
    colors = ['black', 'red', 'green', 'blue']
    linestyles = ['solid','solid','solid','dashed','dashed','dashed']
    plt.figure(figsize=(7,5))
    for i in range(len(lr_matrix)):
        m_label = "stddev=" + str(m_arr[i])
        plt.plot(epochs_matrix[i], lr_matrix[i], label=m_label, c=colors[i],linestyle=linestyles[i])

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Average Adaptive Learning Rates")
    plt.title("scaled_loss_analysis.py: \nLearning Rate Changes over Epochs (Optimizer: " + title + ")")
    plt.savefig('Plots_scaling_loss_analysis/LearningRate_Changes_{0}'.format(title))
    plt.show()
# Plot loss over epochs
def plotLossHistory(title, epochs_matrix, loss_matrix, m_arr, alphas_arr):
    colors = ['black', 'red', 'green', 'blue', 'yellow', 'orange']
    linestyles = ['solid','solid','solid','dashed','dashed','dashed']
    plt.figure(figsize=(7,5))
    for i in range(len(loss_matrix)):
        m_label = "m=" + str(m_arr[i]) + ', alpha=' + str(alphas_arr[i]) 
        plt.plot(epochs_matrix[i], loss_matrix[i], label=m_label, c=colors[i],linestyle=linestyles[i])

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("log(Training Errors)")
    plt.title("scaled_loss_analysis.py: \nLoss History over Epochs (Optimizer: " + title + ")")
    plt.savefig('Plots_scaling_loss_analysis/Loss_History_{0}'.format(title))
    plt.show()

# SCALED LOSS FUNCTION FOR LAZY TRAINING EXPERIMENTATION, AS PER CHIZAT PAPER
class Loss:
    def __init__(self, alpha):
        self.alpha = alpha

    def scaled_loss(self, y_true, y_pred):
        squared_difference =  tf.square(y_true - (y_pred))/(self.alpha**2)
        return tf.reduce_mean(squared_difference, axis=-1)

def main():
    start_time = time.time()
    print("\n\nSTART TIME: ", time.ctime(start_time))
    
    xList = []
    yList = []
    d = 1000 # Dimension of input
    n = 1000 # Number of training samples

    # Generate training data
    #   - xi's generated from U(-1,1)
    #   - yi's generated from N(0,1)
    for i in range(n):
        currX = []
        for j in range(d):
            currXi = random.uniform(-1,1)
            currX.append(currXi)
        xList.append(currX)
        yList.append(random.gauss(0,1))

    xArr = np.array(xList)
    yArr = np.array(yList)
    
    widths = [10,100,1000] #Widths of layers
    alphas = [1,10] #Model scaling
    num_epochs = 150
    
    # Try different optimizers
    custCallback = CustomCallback(learning_rate=.0005)
    sgd = tf.keras.optimizers.SGD(learning_rate=.0005)
    adgd = tf.keras.optimizers.Adagrad(learning_rate=.0005)
    rmsprop = tf.keras.optimizers.RMSprop(learning_rate=.0005)
    adadelta = tf.keras.optimizers.Adadelta(learning_rate=.0005)
    adam = tf.keras.optimizers.Adam(learning_rate=.0005)
    
    optim_list = ['sgd','adagrad','rmsprop','adadelta','adam']
    optimizers = {'sgd':sgd,'adagrad':adgd,'rmsprop':rmsprop,'adadelta':adadelta,'adam':adam}
    matrix_list = ['epochs_matrix','norm_matrix','m_arr','loss_matrix','lr_matrix','alphas_arr']
    
    matrices = {k:{kk:None for kk in matrix_list} for k in optim_list}
    for opt in optim_list:
        norm_matrix = []
        epochs_matrix = []
        loss_matrix = []
        m_arr = []
        alphas_arr = []
        lr_matrix = []
#    while a < len(alphas):
        for a,aa in enumerate(alphas):
            cust_loss = Loss(alphas[a])
            output_activation = ScaledActivation(alphas[a])
    #        while m < len(widths):
            for m,mm in enumerate(widths):
                model = tf.keras.models.Sequential()
                model.add(tf.keras.layers.Dense(widths[m], input_dim=d, activation='relu'))
                # Add scaled output as per lazy training
                model.add(tf.keras.layers.Dense(1, activation=output_activation.scaled_output))
                # model.layers[1].trainable = False
                
                # Objective Functions:
                
                # Compile with custom loss function for lazy training
                model.compile(optimizer=optimizers[opt], loss=cust_loss.scaled_loss, metrics=[tf.keras.metrics.MeanSquaredError()])
                history = model.fit(x=xArr, y=yArr, epochs = num_epochs, batch_size=10, callbacks=custCallback)
    
                # Lists for storing inital model weights
                zero_weights_1 = []
                norm_arr = []
                epochs = []
                loss_arr = []
    
                for epoch_loss in history.history['mean_squared_error']:
                    loss_arr.append(math.log(epoch_loss))
                loss_arr.pop(0)
    
                for epoch,weights in custCallback.weights_dict.items():
                    if epoch == 0:
                        zero_weights_1 = np.transpose(weights[0])
                    else:
                        max_norm_1 = 0
    
                        for i in range(len(zero_weights_1)):
                            wi_norm = np.linalg.norm(x=(zero_weights_1[i] - np.transpose(weights[0])[i])) # Calculate L2 norm of difference between w_t and w_0
                            if wi_norm > max_norm_1:
                                max_norm_1 = wi_norm # Record max norm
    
                        print("Norm difference for epoch " , epoch , " layer 1: ", max_norm_1)
    
                        norm_arr.append(max_norm_1)
                        epochs.append(epoch)
    
                avg_lr = []
                for epoch,rates in custCallback.lr_dict.items():
                    if(epoch != 0):
                        curr_lr = np.mean(rates)
                        avg_lr.append(curr_lr)
                        print("Norm LR for epoch ", epoch, " :" , curr_lr)
    
                epochs_matrix.append(epochs)
                norm_matrix.append(norm_arr)
                loss_matrix.append(loss_arr)
                lr_matrix.append(avg_lr)
                m_arr.append(widths[m])
                alphas_arr.append(alphas[a])
                temp_time = int(time.time()-start_time)
                file = open("Time_Counts/{0}s_{1}_{2}_{3}.txt".format(temp_time, opt, aa, mm),"a")
    #            m = m + 1
    #        a = a + 1
    #        m = 0
        matrices[opt]['epochs_matrix'] = epochs_matrix
        matrices[opt]['loss_matrix'] = loss_matrix
        matrices[opt]['lr_matrix'] = lr_matrix
        matrices[opt]['m_arr'] = m_arr
        matrices[opt]['norm_matrix'] = norm_matrix
        matrices[opt]['alphas_arr'] = alphas_arr
#        plotWeightChanges(epochs_matrix, norm_matrix, m_arr, alphas_arr)
#        plotLossHistory(epochs_matrix, loss_matrix, m_arr, alphas_arr)
    pickle.dump(matrices, open("Plots_scaling_loss_analysis/matrices.pkl","wb")) # Save results
    for opt in optim_list:
        plotWeightChanges(opt, matrices[opt]['epochs_matrix'], matrices[opt]['norm_matrix'], matrices[opt]['m_arr'], matrices[opt]['alphas_arr'])
        plotLossHistory(opt, matrices[opt]['epochs_matrix'], matrices[opt]['loss_matrix'], matrices[opt]['m_arr'], matrices[opt]['alphas_arr'])
#        plotLearningRateChanges(opt, matrices[opt]['epochs_matrix'], matrices[opt]['lr_matrix'], matrices[opt]['m_arr'])
    
    finish_time = time.time()
    print("\n\nFINISH TIME: ", time.ctime(finish_time))
    print("\n\nTIME ELAPSED: ", finish_time-start_time)
   
    return matrices
matrices = main()



# with open('Final_Project/Plots_scaling_loss_analysis/matrices.pkl','rb') as f:
#     matrices = pickle.load(f)