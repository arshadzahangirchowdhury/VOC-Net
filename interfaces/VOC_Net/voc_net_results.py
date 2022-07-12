#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: M Arshad Zahangir Chowdhury

Identify experiments, plot roc, pr curve, grad cam maps etc.

"""

import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from ipywidgets import interactive
import seaborn as sns  #heat map
import glob # batch processing of images

if '../../' not in sys.path:
    sys.path.append('../../')

import math
from scipy import signal
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve 
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

from sklearn.metrics import PrecisionRecallDisplay
from itertools import cycle
from sklearn.model_selection import train_test_split


import itertools

from vocnet.misc.utils import classifier_internals
from vocnet.misc.utils import clf_post_processor


from vocnet.spectral_datasets.IR_datasets import IR_data
from vocnet.spectral_datasets.IR_datasets import spectra_to_img
from vocnet.spectral_datasets.THz_datasets import THz_data

from vocnet.misc.aperture import publication_fig
from vocnet.misc.voc_net_utils import multiclass_roc_auc_score
from vocnet.misc.voc_net_utils import plot_raw_scores
from vocnet.misc.voc_net_utils import simple_spectrum_fig
from vocnet.misc.voc_net_utils import simple_plot_raw_scores

from vocnet.misc.voc_net_utils import plot_sequential_group_prediction



import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from PIL import Image



# GPU_mem_limit=1.0
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_mem_limit*1000.0)])

#     except RuntimeError as e:
#         print(e)        



import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import tensorflow_docs.modeling
from tensorflow.keras import regularizers

from vocnet.models.voc_net_models import get_callbacks
from vocnet.models.voc_net_models import get_optimizer
from vocnet.models.voc_net_models import compile_and_fit


from vocnet.models.voc_net_models import C1f1k3_AP1_D12
from vocnet.models.voc_net_models import C1f1k3_MP1_D12

from vocnet.models.voc_net_models import C2f1k3_AP1_D12
from vocnet.models.voc_net_models import C2f1k3_AP1_D48_D12
from vocnet.models.voc_net_models import C2f1k3_AP2_D48_D12

from vocnet.models.voc_net_models import C2f3k3_AP1_D48_D12
from vocnet.models.voc_net_models import C2f3k3_AP1_D6_D12

from vocnet.models.voc_net_models import C1f1k3_AP1_RD50_D12
from vocnet.models.voc_net_models import C1f1k3_AP1_D48_RL1_D12
from vocnet.models.voc_net_models import C2f3k3_AP1_D48_RD50_D12
from vocnet.models.voc_net_models import C2f3k3_AP1_D48_RL1_D12
from vocnet.models.voc_net_models import C2f3k3_AP1_D48_RL1_RD50_D12

    

from tensorflow import keras
import keras_tuner as kt

import random


#Set random seed
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(42)  
tf.random.get_global_generator().reset_from_seed(42)
np.random.seed(42)
random.seed(42)



# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")



s = THz_data(resolution=0.016, verbosity = False)
s.load_THz_data()
# s.dataset_info()
X = s.spectra
y = s.targets

X=np.expand_dims(X,-1)

#split intro train and test set


TRAIN_SIZE=0.70
TEST_SIZE=1-TRAIN_SIZE

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE,
                                                   test_size=TEST_SIZE,
                                                   random_state=786,
                                                   stratify=y
                                                   )

print("All:", np.bincount(y) / float(len(y))*100  )
print("Training:", np.bincount(y_train) / float(len(y_train))*100  )
print("Testing:", np.bincount(y_test) / float(len(y_test))*100  )


# build the best model here and other functions

def voc_net():

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
    


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                 metrics=[
              tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True, name='SparseCatCrossentropy'),
              'accuracy'])
    
    model.summary()
    
    return model

def grad_cam(layer_name, data):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    last_conv_layer_output, preds = grad_model(data)

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(data)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0))

    last_conv_layer_output = last_conv_layer_output[0]

    heatmap = last_conv_layer_output * pooled_grads
    heatmap = tf.reduce_mean(heatmap, axis=(1))
    heatmap = np.expand_dims(heatmap,0)
    return heatmap


s.load_experiments()
Xexp = s.exp_spectra
yexp = s.exp_targets
SpectraFrame = pd.DataFrame(s.exp_spectra)
SpectraFrame['labels'] = [s.labels[i] for i in s.exp_targets]
SpectraFrame['targets'] =  s.exp_targets
spectraframe = SpectraFrame 
Xexp = np.expand_dims(Xexp,-1)
print(yexp)

print(s.labels)
print([s.labels[i] for i in s.exp_targets])

model = voc_net()

# run on CPU for reproducibility, best epoch is 4.
with tf.device('/CPU:0'):
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='SparseCatCrossentropy', patience=5)
    history = model.fit(x_train, y_train, epochs=4, validation_data=(x_test, y_test), callbacks=[stop_early])



plt.figure(dpi=300)
plt.scatter(history.epoch,history.history['accuracy'], color = 'red', label = 'training')
plt.scatter(history.epoch,history.history['val_accuracy'], color = 'blue', label = 'validation')
plt.legend(loc=4)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig(r'RESULTS/results_figures/accuracies.png', bbox_inches='tight')

plt.figure(dpi=300)
plt.scatter(history.epoch,history.history['SparseCatCrossentropy'], color = 'red', label = 'training')
plt.scatter(history.epoch,history.history['val_SparseCatCrossentropy'],color = 'blue', label = 'validation')
plt.legend(loc=1)
plt.xlabel('Epoch')
plt.ylabel('Sparse categorical crossentropy loss')
plt.savefig(r'RESULTS/results_figures/sparse_cat_losses.png', bbox_inches='tight')


# softmax scores

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(x_test)



print("\ntest/validation\n")

pred_y=np.argmax(model.predict(x_test), axis=-1)

cm = confusion_matrix(y_test, pred_y)

group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]

annot_labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_counts,group_percentages)]

fig = plt.figure(figsize=(16,10), dpi = 600);
# plt.title('Confusion matrix);

# ax = sns.heatmap(cm, annot=True, cmap='PiYG');   #cmap='coolwarm' also good
ax = sns.heatmap(cm/np.sum(cm), annot=np.asarray(annot_labels).reshape(12,12), fmt = '', cmap='RdBu_r',cbar=False);   #cmap='coolwarm' also good
#ax = sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues') #Shows percentage
ax.set_xticklabels(s.labels);
ax.set_yticklabels(s.labels);
plt.xlabel('Predicted Molecule');
plt.ylabel('Actual Moelcule');
plt.xticks(rotation=45);
plt.yticks(rotation=0);
plt.savefig(r'RESULTS/results_figures/cm_test_data.png', bbox_inches='tight')

classifier_internals(pred_y, y_test, y_train, 'simple_CNN')

fig = plot_sequential_group_prediction(np.squeeze(x_test), y_test, predictions, s.frequencies,s.labels, 201, dpi = 300)
plt.close()
fig.savefig(r'RESULTS/results_figures/softmax_figures_test_spectra_start_idx_201.png', bbox_inches='tight')

fig = plot_sequential_group_prediction(np.squeeze(x_test), y_test, predictions, s.frequencies,s.labels, 520, dpi = 300)
plt.close()
fig.savefig(r'RESULTS/results_figures/softmax_figures_test_spectra_start_idx_520.png', bbox_inches='tight')


print('ROC AUC score:', multiclass_roc_auc_score(y_test, pred_y, s.labels)[0])

fig = multiclass_roc_auc_score(y_test, pred_y, s.labels)[1]
plt.close()
fig.savefig(r'RESULTS/results_figures/ROC_test_data.png', bbox_inches='tight')

print("\ntrain\n")
pred_y_train=np.argmax(model.predict(x_train), axis=-1)

cm = confusion_matrix(y_train, pred_y_train)

group_counts = ["{0:0.0f}".format(value) for value in
                cm.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in
                     cm.flatten()/np.sum(cm)]

annot_labels = [f"{v1}\n{v2}" for v1, v2 in
          zip(group_counts,group_percentages)]

fig = plt.figure(figsize=(16,10), dpi = 600);
# plt.title('Confusion matrix);

# ax = sns.heatmap(cm, annot=True, cmap='PiYG');   #cmap='coolwarm' also good
ax = sns.heatmap(cm/np.sum(cm), annot=np.asarray(annot_labels).reshape(12,12), fmt = '', cmap='RdBu_r',cbar=False);   #cmap='coolwarm' also good
#ax = sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues') #Shows percentage
ax.set_xticklabels(s.labels);
ax.set_yticklabels(s.labels);
plt.xlabel('Predicted Molecule');
plt.ylabel('Actual Moelcule');
plt.xticks(rotation=45);
plt.yticks(rotation=0);
plt.savefig(r'RESULTS/results_figures/cm_train_data.png', bbox_inches='tight')

classifier_internals(pred_y_train, y_train, y_train, 'simple_CNN')


print('ROC AUC score:', multiclass_roc_auc_score(y_train, pred_y_train, s.labels)[0])

fig = multiclass_roc_auc_score(y_train, pred_y_train, s.labels)[1]
plt.close()
fig.savefig(r'RESULTS/results_figures/ROC_train_data.png', bbox_inches='tight')


predictions_on_train = probability_model.predict(x_train)



fig = plot_sequential_group_prediction(np.squeeze(x_train), y_train, predictions_on_train, s.frequencies,s.labels, 337, dpi = 300)
plt.close()
fig.savefig(r'RESULTS/results_figures/softmax_figures_train_spectra_start_idx_337.png', bbox_inches='tight')

fig = plot_sequential_group_prediction(np.squeeze(x_train), y_train, predictions_on_train, s.frequencies,s.labels, 1190, dpi = 300)
plt.close()
fig.savefig(r'RESULTS/results_figures/softmax_figures_train_spectra_start_idx_1190.png', bbox_inches='tight')

pred_y_exp=np.argmax(model.predict(Xexp), axis=-1)
print(pred_y_exp)

predictions_exp = probability_model.predict(Xexp) # softmax scores for experiment

print("\nexperiments\n")

fig = plot_sequential_group_prediction(Xexp, yexp, predictions_exp, s.frequencies, 
                                 s.labels, 0)

fig.savefig(r'RESULTS/results_figures/softmax_figures_exp_spectra_start_idx_0_to_11.png', bbox_inches='tight')
plt.close()
fig = plot_sequential_group_prediction(Xexp, yexp, predictions_exp, s.frequencies, 
                                 s.labels, 12)
plt.close()
fig.savefig(r'RESULTS/results_figures/softmax_figures_exp_spectra_start_idx_12_to_23.png', bbox_inches='tight')

fig = plot_sequential_group_prediction(Xexp, yexp, predictions_exp, s.frequencies, 
                                 s.labels, 24)
plt.close()
fig.savefig(r'RESULTS/results_figures/softmax_figures_exp_spectra_start_idx_24_to_36.png', bbox_inches='tight')

print([s.labels[i] for i in s.exp_targets])

cm = confusion_matrix(yexp, pred_y_exp)

# group_counts = ["{0:0.0f}".format(value) for value in
#                 cm.flatten()]

# group_percentages = ["{0:.2%}".format(value) for value in
#                      cm.flatten()/np.sum(cm)]

# annot_labels = [f"{v1}\n{v2}" for v1, v2 in
#           zip(group_counts,group_percentages)]

# fig = plt.figure(figsize=(16,10), dpi = 600);
# # plt.title('Confusion matrix);

# # ax = sns.heatmap(cm, annot=True, cmap='PiYG');   #cmap='coolwarm' also good
# ax = sns.heatmap(cm/np.sum(cm), annot=np.asarray(annot_labels).reshape(6,6), fmt = '', cmap='RdBu_r',cbar=False);   #cmap='coolwarm' also good
# #ax = sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues') #Shows percentage
# ax.set_xticklabels(np.unique([s.labels[i] for i in s.exp_targets]));
# ax.set_yticklabels(np.unique([s.labels[i] for i in s.exp_targets]));
# plt.xlabel('Predicted Molecule');
# plt.ylabel('Actual Moelcule');
# plt.xticks(rotation=45);
# plt.yticks(rotation=0);
# plt.savefig(r'RESULTS/results_figures/cm_exp_data.png', bbox_inches='tight')


print('ROC AUC score:', multiclass_roc_auc_score(yexp, pred_y_exp, s.labels_exp)[0])

fig = multiclass_roc_auc_score(yexp, pred_y_exp, s.labels_exp)[1]
fig.savefig(r'RESULTS/results_figures/ROC_exp_data.png', bbox_inches='tight')


### GRAD-CAM

## PR Curve for test and exp_data are given in notebook



layer_name = "C3"

font = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
        }


count = 0
for i,j in zip(Xexp,yexp):
    
    data = np.expand_dims(i,0)
    heatmap = grad_cam(layer_name,data)

    fig = plt.figure(figsize=(30,4),dpi=300)
    plt.imshow(np.expand_dims(heatmap,axis=2),cmap='YlGnBu', aspect="auto", interpolation='nearest',extent=[0,229,i.min(),i.max()], alpha=0.8)
    
    ticklist = range(0,229)
    plt.xticks(ticklist[::30], np.round(s.frequencies.tolist()[::30], decimals=1) ) # tick every 40th frequency
    plt.plot(i,'k')
    plt.title(f'{s.labels[j]}', fontdict=font)
    plt.colorbar()
    plt.clim(np.min(heatmap),np.max(heatmap))
    
#     plt.show()
    plt.close()
    fig.savefig(r'RESULTS/class_activation_maps/CAM_exp' + str(count) + '.png', bbox_inches='tight')
    count = count + 1


im0 = Image.open(r'RESULTS/class_activation_maps/CAM_exp' + str(0) + '.png') 
im1 = Image.open(r'RESULTS/class_activation_maps/CAM_exp' + str(6) + '.png') 
im2 = Image.open(r'RESULTS/class_activation_maps/CAM_exp' + str(12) + '.png') 
im3 = Image.open(r'RESULTS/class_activation_maps/CAM_exp' + str(18) + '.png') 
im4 = Image.open(r'RESULTS/class_activation_maps/CAM_exp' + str(24) + '.png') 
im5 = Image.open(r'RESULTS/class_activation_maps/CAM_exp' + str(30) + '.png') 
  

fig, ax = plt.subplots(6)
fig.set_size_inches(16, 16)
fig.set_dpi(300)
plt.axis(False)

ax[0].imshow(im0)
ax[0].axis(False)
plt.close()
ax[1].imshow(im1)
ax[1].axis(False)
plt.close()
ax[2].imshow(im2)
ax[2].axis(False)
plt.close()
ax[3].imshow(im3)
ax[3].axis(False)
plt.close()
ax[4].imshow(im4)
ax[4].axis(False)
plt.close()
ax[5].imshow(im5)
ax[5].axis(False)
plt.close()

fig.savefig(r'RESULTS/results_figures/Class_activation_maps_combined_V1.png', bbox_inches='tight')