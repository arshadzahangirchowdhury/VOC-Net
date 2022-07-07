#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: M Arshad Zahangir Chowdhury

Utility functions for processing classification and regression results

"""
import matplotlib as mpl
import matplotlib.pyplot as plt

import sys
import os

import numpy as np
import pandas as pd

from scipy import signal
from ipywidgets import interactive
import seaborn as sns 
import glob 
import datetime

#import metrics to evaluate classifiers
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

    
if '../../' not in sys.path:
    sys.path.append('../../')
    
  

    
    
def classifier_internals(pred_y_internal, test_y_internal, train_y_internal, classifier_name):
    '''
    Function to calculate certain classification metrics. Shows misclassified molecules' indices.
    '''
    
    print('----------------------------',classifier_name, '-------------------------------')
    print("Fraction Correct[Accuracy]:")
    print(np.sum(pred_y_internal == test_y_internal) / float(len(test_y_internal))) #proportion of correct predictions
    

    print("Samples Correctly Classified:")
    correct_idx = np.where(pred_y_internal == test_y_internal)
    print(correct_idx)

    print("Samples Incorrectly Classified:")
    incorrect_idx = np.where(pred_y_internal != test_y_internal) 
    print(incorrect_idx)
    
    #troubleshoot the model

    print('All Train y with label identifier', train_y_internal,'\n')


    print('Data misidentified:\n')
    print('Test y with incorrect indexes label identifier', test_y_internal[incorrect_idx], '\n')
    print('Predicted y with incorrect indexes label identifier', pred_y_internal[incorrect_idx],'\n')

    print('Correct data identified:\n')

    print('Test y with correct indexes label identifier', test_y_internal[correct_idx], '\n')
    print('Predicted y correct indexes label identifier', pred_y_internal[correct_idx],'\n')

    print('All Test y with label identifier', test_y_internal,'\n')
    print('All Pred y with label identifier', pred_y_internal, '\n')
    
def clf_post_processor(pred_y, test_y, labels, cm_title='Confusion Matrix',cmap_color='coolwarm', verbose = True):
    '''
    Function to plot confusion matrix. 
    
    '''
        
    FCA=np.sum(pred_y == test_y) / float(len(test_y))
    print("Fraction Correct[Accuracy]:", FCA)

    print('Accuracy Score', accuracy_score(test_y, pred_y))
    
    cm = confusion_matrix(test_y, pred_y);
    
    if verbose:
        
        fig = plt.figure();
        plt.title(cm_title);
        ax = sns.heatmap(cm, annot=True, cmap=cmap_color);   #cmap='coolwarm' also good
        #ax = sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues') #Shows percentage
        ax.set_xticklabels(labels);
        ax.set_yticklabels(labels);
        plt.xlabel('Predicted Molecule');
        plt.ylabel('Actual Moelcule');
        plt.xticks(rotation=90);
        plt.yticks(rotation=0);

        print(classification_report(test_y, pred_y))

    return FCA, cm



def exp_classifier_internals(pred_y_internal, test_y_internal, classifier_name, labels):
    '''
    same as classifier_internals functions but for experimental spectra and viewing the results in notebook.
    '''
    print('--',classifier_name,'--')
    accc=float((100*np.sum(pred_y_internal == test_y_internal))) / float(len(test_y_internal))
    print("[Accuracy]: ", "%.2f" % accc)
    
    

    
    correct_idx = np.where(pred_y_internal == test_y_internal)
    

    
    incorrect_idx = np.where(pred_y_internal != test_y_internal) 
    
    
#     #troubleshoot the model
#     print('Exp.      Pred. ')
    
#     i = 0
#     while i < len(pred_y_internal):
#         print(labels[test_y_internal[i]],'   ', labels[pred_y_internal[i]]  )
#         i = i+1

   
#        troubleshoot the model
    print('Pred. ')
    
    i = 0
    while i < len(pred_y_internal):
        print(labels[pred_y_internal[i]]  )
        i = i+1


    #print('Data misidentified:\n')
    #print('Test y with incorrect indexes label identifier', test_y_internal[incorrect_idx], '\n')
    #print('Predicted y with incorrect indexes label identifier', pred_y_internal[incorrect_idx],'\n')

    #print('Correct data identified:\n')

    #print('Experimental y with correct indexes label identifier', test_y_internal[correct_idx], '\n')
    #print('Predicted y correct indexes label identifier', pred_y_internal[correct_idx],'\n')
    print('\n')
    print('Experimental y labels: ', test_y_internal,'\n')
    print('Predicted    y labels: ', pred_y_internal, '\n')
    print('------------------\n')
    

def exp_classifier_internals_to_file(pred_y_internal, test_y_internal, classifier_name, labels):
    '''
    Write results to file through function
    
    '''
    #internal means local variables.
    file2 = open("Experimental_Validation_Matrix_Results.txt","a") 
    
    LineI="\n" + classifier_name + "\n"
    

    file2.writelines([LineI])
    
#   print('--',classifier_name,'--')
    accc=float((100*np.sum(pred_y_internal == test_y_internal))) / float(len(test_y_internal))
    LineII="[Accuracy]: " + "%.2f" % accc +"\n"
    file2.writelines([LineII])
    
    

    
    correct_idx = np.where(pred_y_internal == test_y_internal)
    

    
    incorrect_idx = np.where(pred_y_internal != test_y_internal) 
    
    
#     #troubleshoot the model
     
#     file2.writelines(["Exp.      Pred. " + "\n"])
    
#     i = 0
#     while i < len(pred_y_internal):
#         print(labels[test_y_internal[i]],'   ', labels[pred_y_internal[i]]  )
#         file2.writelines([labels[test_y_internal[i]] + '   ' + labels[pred_y_internal[i]] + "\n"  ])
#         i = i+1

#only writing predictions
    i = 0
    while i < len(pred_y_internal):
        print(labels[pred_y_internal[i]]  )
        file2.writelines(labels[pred_y_internal[i]] + "\n")
        i = i+1

   

    file2.close() #to change file access modes 
    
        

def simple_plotter(x,y,linewidth=0.5,color='black',label='$molecule$', 
                   majorsize=6,minorsize=2,width=1, labelsize=8,legendsize=3,legendloc=2, 
                   labelpad=4,fontsize='medium',fontweight='bold',
                  xmajormplloc=0.5,xminormplloc=0.2, tickdirection='out',dpi=300):
    
    '''
    Makes a high resolution plot of a spectra. 
    '''
    
    #plotting bases

    fig = plt.figure(figsize=(2,0.7),dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])

    # ax_CO2 = fig.add_axes([0, 0.6, 1, 0.4])

    plt.rc('font', weight='bold')
    ax.plot(x, y, linewidth=linewidth, color=color, label=label)
    ax.legend(loc=legendloc, prop={'size': legendsize})

    # ax.plot(wavenumbers, X[25]/max(X[25]), linewidth=2, color=colors(1), label='Sample 2')
    # Set the axis limits
    ax.set_xlim(x[0], x[-1])
    # ax.set_ylim(0, 1.0)

    ax.xaxis.set_tick_params(which='major', size=majorsize, width=width, direction=tickdirection,labelsize=labelsize)
    ax.xaxis.set_tick_params(which='minor', size=minorsize, width=width, direction=tickdirection,labelsize=labelsize)
    # ax.tick_params(direction='out', length=6, width=2, colors='r',
    #                grid_color='r', grid_alpha=0.2)

    ax.yaxis.set_tick_params(which='major', size=majorsize, width=width, direction=tickdirection,labelsize=labelsize)
    ax.yaxis.set_tick_params(which='minor', size=minorsize, width=width, direction=tickdirection,labelsize=labelsize)


    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(xmajormplloc))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(xminormplloc))
    # ax_C2H5OH.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.01))
    # ax_C2H5OH.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.005))

    # Add the x and y-axis labels
    ax.set_ylabel(r'Abs.', labelpad=labelpad, fontsize=fontsize, fontweight=fontweight)
    ax.set_xlabel(r'Frequency, $\nu$ ($cm^{-1}$)', labelpad=labelpad, fontsize=fontsize, fontweight=fontweight)
    # ax_H2O.set_xlabel(r'Frequency, $\nu$ ($cm^{-1}$)', labelpad=4, fontsize='medium', fontweight='bold')

    plt.show()
    #fig.savefig('MasterNormalized\H2O.jpg', bbox_inches='tight')

    
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()
        

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    plt.gcf().set_dpi(300)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw, orientation='horizontal')
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="baseline")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on bottom.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right",
             rotation_mode="anchor")
    ax.xaxis.set_label_position('top') 
    ax.set_xlabel(r'Component Species', labelpad=4, fontsize = '12',  fontweight='bold')

    # Turn spines off and create white grid.
#     ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:%0.3f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
#             text = im.axes.text(j, i, valfmt(data[i, j],  None), **kw, size=5)
            text = im.axes.text(j, i, abs(np.around(data[i, j],  decimals=3)), **kw, size=5) #fixes the t in the heatmap
            texts.append(text)

    return texts    

def svm_clf_post_processor(pred_y, test_y, labels, figure_save_file):
    '''
    A function to postprocess the results of svm classifier only.
    '''
    
    print(np.sum(pred_y == test_y) / float(len(test_y))) 
    FCA_OVR=np.sum(pred_y == test_y) / float(len(test_y))

    

    cm_OVR = confusion_matrix(test_y, pred_y)
    
    fig = plt.figure(figsize=(16,10));
    
    ax = sns.heatmap(cm_OVR,linewidths=2, annot=True, cmap='RdPu');  
    
    ax.set_xticklabels(labels);
    ax.set_yticklabels(labels);
    plt.xlabel('Predicted Molecule',fontsize='medium', fontweight='bold');
    plt.ylabel('Actual Moelcule',fontsize='medium', fontweight='bold');
    plt.xticks(rotation=90, fontweight='bold');
    plt.yticks(rotation=0, fontweight='bold');
    plt.title(' Accuracy={0:0.2f}%\n'.format(FCA_OVR*100), fontsize='medium', fontweight='bold');
    fig.savefig(figure_save_file, bbox_inches='tight',dpi=300)
    
    return FCA_OVR, cm_OVR


    
if __name__ == "__main__":
    
    print('just some classes and functions!')