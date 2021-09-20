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

# Callback to save checkpoints
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

# Display plot of weight changes
def plotWeightChanges(title, epochs_matrix, norm_matrix, m_arr):
    plt.figure(figsize=(7,5))
    colors = ['black', 'red', 'green', 'blue']
    for i in range(len(norm_matrix)):
        m_label = "stddev=" + str(m_arr[i])
        plt.plot(epochs_matrix[i], norm_matrix[i], label=m_label, c=colors[i])

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Maximum Distances")
    plt.title("scaling_initialization_analysis.py: \nWeight Changes over Epochs (Optimizer: " + title + ")")
    plt.savefig('Plots_scaling_initializations_analysis/Weight_Changes_{0}'.format(title))
    plt.show()
# Display loss over epochs
def plotLossHistory(title, epochs_matrix, loss_matrix, m_arr):
    plt.figure(figsize=(7,5))
    colors = ['black', 'red', 'green', 'blue']
    for i in range(len(loss_matrix)):
        m_label = "stddev=" + str(m_arr[i])
        plt.plot(epochs_matrix[i], loss_matrix[i], label=m_label, c=colors[i])

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("log(Training Errors)")
    plt.title("scaling_initialization_analysis.py: \nLoss History over Epochs (Optimizer: " + title + ")")
    plt.savefig('Plots_scaling_initializations_analysis/Loss_History_{0}'.format(title))
    plt.show()
# Display learning rate changes
def plotLearningRateChanges(title, epochs_matrix, lr_matrix, m_arr):
    plt.figure(figsize=(7,5))
    colors = ['black', 'red', 'green', 'blue']
    for i in range(len(lr_matrix)):
        m_label = "stddev=" + str(m_arr[i])
        plt.plot(epochs_matrix[i], lr_matrix[i], label=m_label, c=colors[i])

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Average Adaptive Learning Rates")
    plt.title("scaling_initialization_analysis.py: \nLearning Rate Changes over Epochs (Optimizer: " + title + ")")
    plt.savefig('Plots_scaling_initializations_analysis/LearningRate_Changes_{0}'.format(title))
    plt.show()
# Bar chart for training-test loss
def plotTrainTestLoss(title, trainArr, testArr, std_dev_arr):
    plt.figure(figsize=(7,5))
    X = np.arange(len(std_dev_arr))
    fig, ax = plt.subplots()
    ax.bar(X + 0.00, trainArr, color = 'b', width = 0.25, label="training")
    ax.bar(X + 0.25, testArr, color = 'g', width = 0.25, label="testing")
    ax.set_xticks(X)
    ax.set_xticklabels(std_dev_arr)
    plt.xlabel("Standard Deviation of Weight Initializations")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("scaled_initializations_analysis.py: \nTrain-Test Loss (Optimizer: " + title + ")")
    plt.savefig('Plots_scaling_initializations_analysis/Train_Test_Loss_{0}'.format(title))
    plt.show()

def main(): # Main method to run training
    start_time = time.time()
    print("\n\nSTART TIME: ", time.ctime(start_time))
    xList = []
    yList = []
    xListTest = []
    yListTest = []
    d = 10 # Dimension of input
    n = 100 # Number of training samples

    # Generate training data
    #   - xi's generated from U(0,1)
    #   - yi's generated from summation of sin(pi*x_i)
    for i in range(n): # 100 iterations
        output = 0
        currX = []
        currXTest = []
        outputTest = 0
        for j in range(d): # 10 iterations
            currXi = random.uniform(0,1)
            output += math.sin(math.pi * currXi)
            currX.append(currXi)

            currXiTest = random.uniform(0,1)
            outputTest += math.sin(math.pi * currXiTest)
            currXTest.append(currXiTest)

        xList.append(currX)
        yList.append(output)
        xListTest.append(currXTest)
        yListTest.append(outputTest)


    xArr = np.array(xList)
    yArr = np.array(yList)
    xArrTest = np.array(xListTest)
    yArrTest = np.array(yListTest)
    

    std_dev = [.01,.1,1] #std_dev of weight initializations
    m = 0
    num_epochs = 300
    
    # Various optimization functions
    custCallback = CustomCallback(learning_rate=.0005)
    sgd = tf.keras.optimizers.SGD(learning_rate=.0005)
    adgd = tf.keras.optimizers.Adagrad(learning_rate=.0005)
    rmsprop = tf.keras.optimizers.RMSprop(learning_rate=.0005)
    adadelta = tf.keras.optimizers.Adadelta(learning_rate=.0005)
    adam = tf.keras.optimizers.Adam(learning_rate=.0005)
    
    optim_list = ['sgd','adagrad','rmsprop','adadelta','adam']
    optimizers = {'sgd':sgd,'adagrad':adgd,'rmsprop':rmsprop,'adadelta':adadelta,'adam':adam}
    matrix_list = ['norm_matrix','epochs_matrix','loss_matrix','std_dev_arr','trainErrors','testErrors']
    
    matrices = {k:{kk:None for kk in matrix_list} for k in optim_list}
    
    for opt in optim_list:
        norm_matrix = []
        epochs_matrix = []
        loss_matrix = []
        std_dev_arr = []
        trainErrors = []
        testErrors = []
    #    while m < len(std_dev):
        for m,mm in enumerate(std_dev):
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Dense(100, input_dim=d, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0, stddev=mm)))
            model.add(tf.keras.layers.Dense(1))
            model.layers[1].trainable = True
    
            model.compile(optimizer=sgd, loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])
            history = model.fit(x=xArr, y=yArr, epochs = num_epochs, batch_size=10, callbacks=custCallback)
            resultsTest = model.evaluate(xArrTest, yArrTest)
            resultsTrain = model.evaluate(xArr, yArr)
            testErrors.append(resultsTest[0])
            trainErrors.append(resultsTrain[0])
    
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
    
            epochs_matrix.append(epochs)
            norm_matrix.append(norm_arr)
            loss_matrix.append(loss_arr)
            std_dev_arr.append(std_dev[m])
            temp_time = int(time.time()-start_time)
            file = open("Time_Counts/{0}_{1}_{2}_s.txt".format(temp_time, opt, mm),"a")
        matrices[opt]['norm_matrix'] = norm_matrix
        matrices[opt]['epochs_matrix'] = epochs_matrix
        matrices[opt]['loss_matrix'] = loss_matrix
        matrices[opt]['std_dev_arr'] = std_dev_arr
        matrices[opt]['trainErrors'] = trainErrors
        matrices[opt]['testErrors'] = testErrors
        
#        matrices[opt]['lr_matrix'] = lr_matrix
#        matrices[opt]['m_arr'] = m_arr
        
    for opt in optim_list:
        plotWeightChanges(opt, matrices[opt]['epochs_matrix'], matrices[opt]['norm_matrix'], matrices[opt]['std_dev_arr'])
        plotLossHistory(opt, matrices[opt]['epochs_matrix'], matrices[opt]['loss_matrix'], matrices[opt]['std_dev_arr'])
        plotTrainTestLoss(opt, matrices[opt]['trainErrors'], matrices[opt]['testErrors'], matrices[opt]['std_dev_arr'])
        # plotWeightChanges(epochs_matrix, norm_matrix, std_dev_arr)
        # plotLossHistory(epochs_matrix, loss_matrix, std_dev_arr)
        # plotTrainTestLoss(trainErrors, testErrors, std_dev_arr)
    return matrices

matrices = main()
# Store results in hard rive
pickle.dump(matrices, open("Plots_scaling_initializations_analysis/matrices.pkl","wb"))
