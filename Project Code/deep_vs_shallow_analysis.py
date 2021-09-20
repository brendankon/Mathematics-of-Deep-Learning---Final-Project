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

#Activation for scaling network output (LAZY TRAINING SCALING)
class ScaledActivation:
    def __init__(self, alpha=1):
        self.alpha = alpha

    def scaled_output(self,x):
        return x * self.alpha

class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, learning_rate=.0005):
        self.weights_dict = {}
    
    # Function which gets called at the end of each training epoch
    def on_epoch_end(self, epoch, logs=None):
        self.weights_dict.update({epoch:self.model.get_weights()})


# Display plot of weight changes
def plotWeightChanges(title, epochs_matrix, norm_matrix, alphas_arr, depths_arr):
    colors = ['orangered','lime','deepskyblue','red','green','blue','maroon','darkgreen','midnightblue']
    linestyles = ['solid','dashed','dashdot']
    linestyles = [l for l in linestyles for i in range(len(linestyles))]
    plt.figure(figsize=(15,10),dpi=300)
    for i in range(len(norm_matrix)):
        m_label = "alpha={0}, d={1}".format(alphas_arr[i],depths_arr[i])
        plt.plot(epochs_matrix[i], norm_matrix[i], label=m_label, c=colors[i], linestyle=linestyles[i])
    bx = plt.subplot(111)
    box = bx.get_position()
    bx.set_position([box.x0, box.y0, box.width*0.9, box.height])
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 0.75))
    plt.xlabel("Epochs")
    plt.ylabel("Maximum Distances")
    plt.title("deep_vs_shallow_analysis.py: \nWeight Changes over Epochs \n{0}".format(title))
    plt.savefig("deep_vs_shallow_analysis/latest_plots/Weight_Changes_"+title)   
    plt.show()
    
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*1.0, box.height])
    legend_x = 1.1
    legend_y = 0.5
    plt.legend(["blue", "green"], loc='center left', bbox_to_anchor=(legend_x, legend_y))

def plotLossHistory(title, epochs_matrix, loss_matrix, alphas_arr, depths_arr):
    colors = ['orangered','lime','deepskyblue','red','green','blue','maroon','darkgreen','midnightblue']
    linestyles = ['solid','dashed','dashdot']
    linestyles = [l for l in linestyles for i in range(len(linestyles))]
    plt.figure(figsize=(15,10),dpi=300)
    for i in range(len(loss_matrix)):
        m_label = "alpha={0}, d={1}".format(alphas_arr[i],depths_arr[i])
        plt.plot(epochs_matrix[i], loss_matrix[i], label=m_label, c=colors[i], linestyle=linestyles[i])
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.9, box.height])
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 0.75))
    plt.xlabel("Epochs")
    plt.ylabel("log(Training Errors)")
    plt.title("deep_vs_shallow_analysis.py: \nLoss History over Epochs \n{0}".format(title))
    plt.savefig("deep_vs_shallow_analysis/latest_plots/Loss_History_"+title)   
    plt.show()

class Loss: # AS PER CHIZAT, LAZY TRAINING CUSTOM LOSS
    def __init__(self, alpha):
        self.alpha = alpha

    def scaled_loss(self, y_true, y_pred):
        squared_difference =  tf.square(y_true - y_pred)/(self.alpha**2)
        return tf.reduce_mean(squared_difference, axis=-1)

def main():
    count = 0
    start_time = time.time()
    print("\n\nSTART TIME: ", time.ctime(start_time))
    dim = 1000 # Dimension of input
    n_s = [1000,10000] # Number of training samples
    xArr = {n:None for n in n_s}
    yArr = {n:None for n in n_s}
    # Generate training data
    #   - xi's generated from U(-1,1)
    #   - yi's generated from N(0,1)
    for n in xArr.keys():
        xList = []
        yList = []
        for i in range(n):
            currX = []
            for j in range(dim):
                currXi = random.uniform(-1,1)
                currX.append(currXi)
            xList.append(currX)
            yList.append(random.gauss(0,1))
        xArr[n] = np.array(xList)
        yArr[n] = np.array(yList)
        
    alphas = [10,100,1000] #Model scaling
    widths = [10,100,1000]
    depths = [1,5,10] #Hidden Layers
    
    num_epochs = 150
    norm_matrix = []
    epochs_matrix = []
    loss_matrix = []
    alphas_arr = []
    depths_arr = []
    m_arr = []
    param_list = []
    sizes = [1000,10000]
    matrix_list = ['epochs_matrix','norm_matrix','m_arr','loss_matrix','alphas_arr','depths_arr']
    matrices = {s:{l:None for l in matrix_list} for s in sizes}
    custCallback = CustomCallback(learning_rate=.0005)
    sgd = tf.keras.optimizers.SGD(learning_rate=.0005)
    for n in xArr.keys():
        for a,aa in enumerate(alphas):
            cust_loss = Loss(alphas[a])
            output_activation = ScaledActivation(alphas[a])
            for w,ww in enumerate(widths):
                for d,dd in enumerate(depths):
                    model = tf.keras.models.Sequential()
                    model.add(tf.keras.layers.Dense(ww, input_dim=dim, activation='relu'))
        
                    i = 1
                    while i < depths[d]:
                        model.add(tf.keras.layers.Dense(widths[w], activation='relu'))
                        i += 1
        
                    model.add(tf.keras.layers.Dense(1, activation=output_activation.scaled_output))
        
                    
                    
                    model.compile(optimizer=sgd, loss=cust_loss.scaled_loss, metrics=[tf.keras.metrics.MeanSquaredError()])
                    history = model.fit(x=xArr[n], y=yArr[n], epochs = num_epochs, batch_size=10, callbacks=custCallback)
        
                    # Lists for storing inital model weights
                    zero_weights = []
                    norm_arr = []
                    epochs = []
                    loss_arr = []
        
                    for epoch_loss in history.history['mean_squared_error']:
                        loss_arr.append(math.log(epoch_loss))
                    loss_arr.pop(0)
        
                    for epoch,weights in custCallback.weights_dict.items():
                        if epoch == 0:
                            for i in range(len(weights)-1):
                                zero_weights.append(np.transpose(weights[i]))
                        else:
                            weight_norms = []
                            for j in range(len(zero_weights)):
                                max_norm = 0
        
                                for i in range(len(zero_weights[j])):
                                    wi_norm = np.linalg.norm(x=(zero_weights[j][i] - np.transpose(weights[j])[i])) # Calculate L2 norm of difference between w_t and w_0
                                    if wi_norm > max_norm:
                                        max_norm = wi_norm # Record max norm
        
                                weight_norms.append(max_norm)
        
                            norm_arr.append(sum(weight_norms)/len(weight_norms))
                            epochs.append(epoch)
        
                    epochs_matrix.append(epochs)
                    norm_matrix.append(norm_arr)
                    loss_matrix.append(loss_arr)
                    m_arr.append(widths[w])
                    alphas_arr.append(alphas[a])
                    depths_arr.append(depths[d])
                    count += 1
                    temp_time = int(time.time()-start_time)
                    file = open("deep_vs_shallow_analysis/Time/{0}_{1}s_n{2}_a{3}_w{4}_d{5}.txt".format(count, temp_time, n, aa, ww, dd),"a")
                    param_list.append([n,alphas[a],widths[w],depths[d]])
        matrices[n]['epochs_matrix'] = epochs_matrix
        matrices[n]['loss_matrix'] = loss_matrix
        matrices[n]['m_arr'] = m_arr
        matrices[n]['norm_matrix'] = norm_matrix
        matrices[n]['alphas_arr'] = alphas_arr
        matrices[n]['depths_arr'] = depths_arr 
        
    pickle.dump(matrices, open("deep_vs_shallow_analysis/matrices.pkl","wb"))
    pickle.dump(param_list, open("deep_vs_shallow_analysis/param_list.pkl","wb"))
    # plotWeightChanges(epochs_matrix, norm_matrix, m_arr, alphas_arr)
    # plotLossHistory(epochs_matrix, loss_matrix, m_arr, alphas_arr)
    return matrices, param_list

# BELOW COMMENTED CODE WAS NEEDED TO PROCESS DATA FOR GRAPHING
# matrices, param_list = main()
# matrices = pickle.load(open('deep_vs_shallow_analysis/matrices.pkl','rb'))
# param_list = pickle.load(open('deep_vs_shallow_analysis/param_list.pkl','rb'))
# Needed to resolve issues with mismatched data (swap width and alphas, for graphing purposes)
# new_param_list = list()
# new_widths = [10,100,1000]
# new_n = [1000,10000]
# for i,n in enumerate(new_n):
#     for j,w in enumerate(new_widths):
#         for k,p in enumerate(param_list):
#             if(p[0]==n and p[2]==w):
#                 print(w, j)
#                 new_param_list.append(p)

# new_matrices = {n:{new_param_list[i][1:]:None for i in range(int(len(param_list)/2))} for n in new_n}

# temp_matrices = {n:{} for n in matrices.keys()}
# for n in matrices.keys():
#     if n==1000:
#         for k in matrices[n].keys():
#             temp_matrices[n][k]=matrices[n][k][:27]
#     elif n==10000:
#         for k in matrices[n].keys():
#             temp_matrices[n][k]=matrices[n][k][27:]

# m_keys = ['i','nawd_params','alphas_arr','depths_arr','epochs_matrix','loss_matrix','m_arr','norm_matrix']  
# new_matrices = {}
# for n in new_n:
#     new_matrices[n]={w:{m:[] for m in m_keys} for w in new_widths}


# count = 0
# for n in new_matrices.keys():
#     for w in new_matrices[n].keys():
#         for i in range(int(len(param_list)/2)):
#             k = param_list[i]
#             if k[2]==w:
#                 print(i, "-->", k)
#                 new_matrices[n][w]['i'].append(i)
#                 new_matrices[n][w]['nawd_params'].append(k)
#                 new_matrices[n][w]['alphas_arr'].append(temp_matrices[n]['alphas_arr'][i])
#                 new_matrices[n][w]['depths_arr'].append(temp_matrices[n]['depths_arr'][i])
#                 new_matrices[n][w]['epochs_matrix'].append(temp_matrices[n]['epochs_matrix'][i])
#                 new_matrices[n][w]['loss_matrix'].append(temp_matrices[n]['loss_matrix'][i])
#                 new_matrices[n][w]['m_arr'].append(temp_matrices[n]['m_arr'][i])
#                 new_matrices[n][w]['norm_matrix'].append(temp_matrices[n]['norm_matrix'][i])

# pickle.dump(new_matrices, open("deep_vs_shallow_analysis/new_matrices.pkl","wb"))

new_matrices = pickle.load(open('deep_vs_shallow_analysis/new_matrices.pkl','rb'))
param_list = pickle.load(open('deep_vs_shallow_analysis/param_list.pkl','rb'))
size_list = new_matrices.keys()
for n in size_list:
    for w in new_matrices[n].keys():
        title = "n = {0}, Width = {1}".format(n,w)
        plotWeightChanges(title, new_matrices[n][w]['epochs_matrix'], new_matrices[n][w]['norm_matrix'], new_matrices[n][w]['alphas_arr'], new_matrices[n][w]['depths_arr'])
        plotLossHistory(title, new_matrices[n][w]['epochs_matrix'], new_matrices[n][w]['loss_matrix'], new_matrices[n][w]['alphas_arr'], new_matrices[n][w]['depths_arr'])
            # plotLearningRateChanges(opt, matrices[opt]['epochs_matrix'], matrices[opt]['lr_matrix'], matrices[opt]['m_arr'])
