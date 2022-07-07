#!/usr/bin/env python3
# -*- coding: utf-8 -*-



"""
Author: M Arshad Zahangir Chowdhury
Email: arshad.zahangir.bd[at]gmail[dot]com
Definitions of various models tested for voc-net.
"""

import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers

import tensorflow_docs.plots
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling



import pathlib
import shutil
import tempfile

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

def get_callbacks(name):
    return [
    tfdocs.modeling.EpochDots(),
#     tf.keras.callbacks.EarlyStopping(monitor='val_SparseCatCrossentropy', patience=100),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]

def get_optimizer():
    return tf.keras.optimizers.Adam()

def compile_and_fit(model, name, x_train, y_train, x_test, y_test, STEPS_PER_EPOCH,  optimizer=None, max_epochs=200):
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[
              tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True, name='SparseCatCrossentropy'),
              'accuracy'])

    model.summary()

    history = model.fit(
    x_train,
    y_train,
    steps_per_epoch = STEPS_PER_EPOCH,
    epochs=max_epochs,
    validation_data=(x_test, y_test),
    callbacks=get_callbacks(name),
    verbose=0)
    
    return history

# def simplest_model():
def C1f1k3_AP1_D12():

    model = models.Sequential()

    # C1 Convolutional Layer
    model.add(layers.Conv1D(filters = 1 , kernel_size=3, activation='relu', input_shape=(229, 1), name = 'C1') )

    # S2 Subsampling Layer
    model.add(layers.AveragePooling1D(pool_size = 2, strides = 2, padding = 'valid', name = 'S2'))

    # Flatten the CNN output to feed it with fully connected layers
    model.add(layers.Flatten())
    
    
    model.add(layers.Dense(12))  # number of dense layer would be equal to number of classess
    model.summary()
    
    return model


# def simplest_with_max_pool_model():
def C1f1k3_MP1_D12():

    model = models.Sequential()

    # C1 Convolutional Layer
    model.add(layers.Conv1D(filters = 1 , kernel_size=3, activation='relu', input_shape=(229, 1), name = 'C1') )

    # S2 Subsampling Layer
    model.add(layers.MaxPooling1D(pool_size = 2, strides = 2, padding = 'valid', name = 'S2'))

    # Flatten the CNN output to feed it with fully connected layers
    model.add(layers.Flatten())
    
    
    model.add(layers.Dense(12))  # number of dense layer would be equal to number of classess
    model.summary()
    
    return model

# def multi_conv_model():
def C2f1k3_AP1_D12():

    model = models.Sequential()

    # C1 Convolutional Layer
    model.add(layers.Conv1D(filters = 1 , kernel_size=3, activation='relu', input_shape=(229, 1), name = 'C1') )

    # S2 Subsampling Layer
    model.add(layers.AveragePooling1D(pool_size = 2, strides = 2, padding = 'valid', name = 'S2'))
    
    # C3 Convolutional Layer
    model.add(layers.Conv1D(filters = 1 , kernel_size=3, activation='relu', name = 'C3') )

    # Flatten the CNN output to feed it with fully connected layers
    model.add(layers.Flatten())
    
    
    model.add(layers.Dense(12))  # number of dense layer would be equal to number of classess
    model.summary()
    
    return model

# def multi_conv_with_dense_model():
def C2f1k3_AP1_D48_D12():

    model = models.Sequential()

    # C1 Convolutional Layer
    model.add(layers.Conv1D(filters = 1 , kernel_size=3, activation='relu', input_shape=(229, 1), name = 'C1') )

    # S2 Subsampling Layer
    model.add(layers.AveragePooling1D(pool_size = 2, strides = 2, padding = 'valid', name = 'S2'))
    
    # C3 Convolutional Layer
    model.add(layers.Conv1D(filters = 1 , kernel_size=3, activation='relu', name = 'C3') )

    # Flatten the CNN output to feed it with fully connected layers
    model.add(layers.Flatten())
    
    model.add(layers.Dense(48, activation='relu')) 
    model.add(layers.Dense(12))  # number of dense layer would be equal to number of classess
    model.summary()
    
    return model

# def multi_conv_with_second_pool_model():
def C2f1k3_AP2_D48_D12():

    model = models.Sequential()

    # C1 Convolutional Layer
    model.add(layers.Conv1D(filters = 1 , kernel_size=3, activation='relu', input_shape=(229, 1), name = 'C1') )

    # S2 Subsampling Layer
    model.add(layers.AveragePooling1D(pool_size = 2, strides = 2, padding = 'valid', name = 'S2'))
    
    # C3 Convolutional Layer
    model.add(layers.Conv1D(filters = 1 , kernel_size=3, activation='relu', name = 'C3') )
    
    # S4 Subsampling Layer
    model.add(layers.AveragePooling1D(pool_size = 2, strides = 2, padding = 'valid', name = 'S4'))

    # Flatten the CNN output to feed it with fully connected layers
    model.add(layers.Flatten())
    
    model.add(layers.Dense(48, activation='relu'))
    model.add(layers.Dense(12))  # number of dense layer would be equal to number of classess
    model.summary()
    
    return model


# def multi_conv_with_dense_multi_filter_model():
def C2f3k3_AP1_D48_D12():

    model = models.Sequential()

    # C1 Convolutional Layer
    model.add(layers.Conv1D(filters = 3 , kernel_size=3, activation='relu', input_shape=(229, 1), name = 'C1') )

    # S2 Subsampling Layer
    model.add(layers.AveragePooling1D(pool_size = 2, strides = 2, padding = 'valid', name = 'S2'))
    
    # C3 Convolutional Layer
    model.add(layers.Conv1D(filters = 3 , kernel_size=3, activation='relu', name = 'C3') )

    # Flatten the CNN output to feed it with fully connected layers
    model.add(layers.Flatten())
    
    model.add(layers.Dense(48, activation='relu')) 
    model.add(layers.Dense(12))  # number of dense layer would be equal to number of classess
    model.summary()
    
    return model

# def multi_conv_with_dense_multi_filter_bottleneck_model():
def C2f3k3_AP1_D6_D12():

    model = models.Sequential()

    # C1 Convolutional Layer
    model.add(layers.Conv1D(filters = 3 , kernel_size=3, activation='relu', input_shape=(229, 1), name = 'C1') )

    # S2 Subsampling Layer
    model.add(layers.AveragePooling1D(pool_size = 2, strides = 2, padding = 'valid', name = 'S2'))
    
    # C3 Convolutional Layer
    model.add(layers.Conv1D(filters = 3 , kernel_size=3, activation='relu', name = 'C3') )

    # Flatten the CNN output to feed it with fully connected layers
    model.add(layers.Flatten())
    
    model.add(layers.Dense(6, activation='relu'))
    model.add(layers.Dense(12))  # number of dense layer would be equal to number of classess
    model.summary()
    
    return model


# def simplest_reg_dropout_model():
def C1f1k3_AP1_RD50_D12():

    model = models.Sequential()

    # C1 Convolutional Layer
    model.add(layers.Conv1D(filters = 1 , kernel_size=3, activation='relu', input_shape=(229, 1), name = 'C1') )

    # S2 Subsampling Layer
    model.add(layers.AveragePooling1D(pool_size = 2, strides = 2, padding = 'valid', name = 'S2'))

    # Flatten the CNN output to feed it with fully connected layers
    model.add(layers.Flatten())
    
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(12))  # number of dense layer would be equal to number of classess
    model.summary()
    
    return model


# def simplest_reg_L2_model():
def C1f1k3_AP1_D48_RL1_D12():

    model = models.Sequential()

    # C1 Convolutional Layer
    model.add(layers.Conv1D(filters = 1 , kernel_size=3, activation='relu', input_shape=(229, 1), name = 'C1') )

    # S2 Subsampling Layer
    model.add(layers.AveragePooling1D(pool_size = 2, strides = 2, padding = 'valid', name = 'S2'))

    # Flatten the CNN output to feed it with fully connected layers
    model.add(layers.Flatten())
    
    
    model.add(layers.Dense(48, activation='relu', kernel_regularizer=regularizers.l2(0.01))) # tanh yielded good result
    model.add(layers.Dense(12,kernel_regularizer=regularizers.l2(0.01)))  # number of dense layer would be equal to number of classess
    model.summary()
    
    return model


# def multi_conv_with_dense_multi_filter_reg_model():
def C2f3k3_AP1_D48_RD50_D12():

    model = models.Sequential()

    # C1 Convolutional Layer
    model.add(layers.Conv1D(filters = 3 , kernel_size=3, activation='relu', input_shape=(229, 1), name = 'C1') )

    # S2 Subsampling Layer
    model.add(layers.AveragePooling1D(pool_size = 2, strides = 2, padding = 'valid', name = 'S2'))
    
    # C3 Convolutional Layer
    model.add(layers.Conv1D(filters = 3 , kernel_size=3, activation='relu', name = 'C3') )

    # Flatten the CNN output to feed it with fully connected layers
    model.add(layers.Flatten())
    
    model.add(layers.Dense(48, activation='relu')) 
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(12))  # number of dense layer would be equal to number of classess
    model.summary()
    
    return model


# def multi_conv_with_dense_multi_filter_reg_L2_model():
def C2f3k3_AP1_D48_RL1_D12():

    model = models.Sequential()

    # C1 Convolutional Layer
    model.add(layers.Conv1D(filters = 3 , kernel_size=3, activation='relu', input_shape=(229, 1), name = 'C1') )

    # S2 Subsampling Layer
    model.add(layers.AveragePooling1D(pool_size = 2, strides = 2, padding = 'valid', name = 'S2'))
    
    # C3 Convolutional Layer
    model.add(layers.Conv1D(filters = 3 , kernel_size=3, activation='relu', name = 'C3') )

    # Flatten the CNN output to feed it with fully connected layers
    model.add(layers.Flatten())
    
    model.add(layers.Dense(48, activation='relu',kernel_regularizer=regularizers.l2(0.01))) 
    model.add(layers.Dense(12,kernel_regularizer=regularizers.l2(0.01)))  # number of dense layer would be equal to number of classess
    model.summary()
    
    return model


# def multi_conv_with_dense_multi_filter_reg_L2_dropout_model():
def C2f3k3_AP1_D48_RL1_RD50_D12():

    model = models.Sequential()

    # C1 Convolutional Layer
    model.add(layers.Conv1D(filters = 3 , kernel_size=3, activation='relu', input_shape=(229, 1), name = 'C1') )

    # S2 Subsampling Layer
    model.add(layers.AveragePooling1D(pool_size = 2, strides = 2, padding = 'valid', name = 'S2'))
    
    # C3 Convolutional Layer
    model.add(layers.Conv1D(filters = 3 , kernel_size=3, activation='relu', name = 'C3') )

    # Flatten the CNN output to feed it with fully connected layers
    model.add(layers.Flatten())
    
    model.add(layers.Dense(48, activation='relu',kernel_regularizer=regularizers.l2(0.01))) 
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(12,kernel_regularizer=regularizers.l2(0.01)))  # number of dense layer would be equal to number of classess
    model.summary()
    
    return model

