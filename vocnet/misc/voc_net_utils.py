#!/usr/bin/env python3
# -*- coding: utf-8 -*-



"""
Author: M Arshad Zahangir Chowdhury
Email: arshad.zahangir.bd[at]gmail[dot]com
Definitions of various functions for plotting results for voc-net.
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

import itertools

from aimos.misc.utils import classifier_internals
from aimos.misc.utils import clf_post_processor




from aimos.misc.aperture import publication_fig




def multiclass_roc_auc_score(y_test, y_pred, target, average="macro", figsize = (12, 8), dpi=150 ):
    '''function for scoring roc auc score for multi-class'''
    
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    
    fig, c_ax = plt.subplots(1,1, figsize = figsize, dpi=dpi)

    for (idx, c_label) in enumerate(target):
        fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    
    c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')    
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate', weight ='bold', fontsize = 15)
    c_ax.set_ylabel('True Positive Rate', weight ='bold', fontsize = 15)
    
    c_ax.xaxis.set_tick_params(which='major', 
                                 size=12, 
                                 width=1, 
                                 direction='out',
                                 labelsize=15)


    c_ax.yaxis.set_tick_params(which='major', 
                             size=12, 
                             width=1, 
                             direction='out',
                             labelsize=15)

    

    plt.close()
    
    return roc_auc_score(y_test, y_pred, average=average), fig, c_ax



def plot_raw_scores(i, predictions_array, true_label, all_unique_labels, 
                    fig_prop = {'figsize':(2,0.7), 'dpi':300, 'ax_rect': [0,0,1,1]}
                   ):
    fig = plt.figure(figsize=fig_prop['figsize'],dpi=fig_prop['dpi'])
    ax = fig.add_axes(fig_prop['ax_rect'])
    
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(12),all_unique_labels)
    plt.yticks([])
    thisplot = plt.bar(range(12), predictions_array[i], color="#777777")
    plt.ylim([0, 1])
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    predicted_label = np.argmax(predictions_array[i])
    
    ax.set_xticklabels(all_unique_labels);
    ax.set_xlabel('Labels', labelpad = 4, 
                  fontsize = 'small', 
                  fontweight = 'bold')
    ax.set_ylabel('Softmax scores', labelpad = 4, 
                  fontsize = 'small', 
                  fontweight = 'bold')
    ax.xaxis.set_tick_params(which='major', 
                             size=5, 
                             width=1, 
                             direction='out',
                             labelsize=7)
    ax.yaxis.set_tick_params(which='major', 
                             size=5, 
                             width=1, 
                             direction='out',
                             labelsize=7)
    
    
    
    plt.xticks(rotation=90, fontweight='bold');

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    plt.close()
    return fig, ax


def simple_spectrum_fig(frequencies, absorbances):
    
    spectrum_plot = plt.plot(frequencies, absorbances/max(absorbances), linewidth = 0.5, color = 'black')
    plt.xlabel('Frequency ($cm^{-1}$)')
    plt.ylabel('Norm. Abs.')
    plt.xlim([frequencies[0], frequencies[-1]])
    
def simple_plot_raw_scores(i, predictions_array, true_label,all_unique_labels):
    
    true_label = true_label[i]
    plt.grid(False)
    plt.yticks(range(12),all_unique_labels)

    scoreplot = plt.barh(range(12), predictions_array[i], color="#777777")
    
    plt.xlim([0, 1])
    predicted_label = np.argmax(predictions_array[i])
    plt.yticks(fontsize = 7);
    plt.tick_params(axis = 'y', direction = 'out') # , pad =-335
    
    plt.ylabel('label')
    plt.xlabel('softmax score')
    scoreplot[predicted_label].set_color('red')
    scoreplot[true_label].set_color('blue')
    


def plot_sequential_group_prediction(x,y, predictions, frequencies, all_unique_labels, start=0,dpi=300):
    '''Plot the first 12 test spectra, their predicted labels, and the true labels.
Color correct predictions in blue and incorrect predictions in red.'''
    num_rows = 4
    num_cols = 3
    num_images = num_rows*num_cols
    fig = plt.figure(figsize=(2*2*num_cols, 2*num_rows),dpi=dpi)

    if start<0:
        start=0

    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        simple_spectrum_fig(frequencies, x[i+int(start)])
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        simple_plot_raw_scores(i+int(start), predictions, y,all_unique_labels)
    plt.tight_layout()
#     plt.show()
    
    return fig
    