#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 05:03:48 2021

@author: deepakd
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import tensorflow.keras.backend as K
import time
import pickle
import os

# Up to line 44 is for converting the imported MNIST set for training
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

widths = [100,1000,10000] # Hidden layer width
matrices = {m:None for m in widths} # Store results
ep = 30 # Epochs
count = 1
start_time = time.time()
for m in widths:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(m,activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    
    history = model.fit(ds_train, epochs=ep, validation_data=ds_test,)
    
    matrices[m]=history.history
    
    temp_time = int(time.time()-start_time)
    file = open("MNIST/time/{0}_{1}s_m={2}.txt".format(count, temp_time, m),"a")
    count += 1

# Save results
pickle.dump(matrices,open('MNIST/matrices.pkl','wb'))  
matrices = pickle.load(open('MNIST/matrices_OLD.pkl','rb'))
# plot learning curves
colors = {100:['red','orangered'],1000:['blue','deepskyblue'],10000:['green','lime']}
linestyles = {m:['solid','dashed'] for m in widths}

plt.title('Loss over Time')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')

for m in widths:
    x1 = matrices[m]['loss']
    x2 = matrices[m]['val_loss']
    plt.plot(x1,label='train-m={0}'.format(m), c=colors[m][0],linestyle=linestyles[m][0])
    plt.plot(x2, label='val-m={0}'.format(m), c=colors[m][1],linestyle=linestyles[m][1])
plt.legend()
plt.savefig('MNIST/Loss')
plt.show()

plt.title('Accuracy over Time')
plt.xlabel('Epoch')
plt.ylabel('Sparse Categorical Accuracy')
for m in widths:
    x3 = matrices[m]['sparse_categorical_accuracy']
    x4 = matrices[m]['val_sparse_categorical_accuracy']
    plt.plot(x3, label='train-m={0}'.format(m), c=colors[m][0],linestyle=linestyles[m][0])
    plt.plot(x4, label='val-m={0}'.format(m), c=colors[m][1],linestyle=linestyles[m][1])
plt.legend()
plt.savefig('MNIST/Acc')
plt.show()