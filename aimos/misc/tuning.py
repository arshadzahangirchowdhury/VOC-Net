#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: M Arshad Zahangir Chowdhury

Functions needed for hyperparameter tuning
"""



import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix    #confusion matrix
import seaborn as sns  #heat map
import glob # batch processing of images
from scipy import signal
from sklearn.metrics import *
import matplotlib as mpl
from math import *
if '../../' not in sys.path:
    sys.path.append('../../')
from aimos.spectral_datasets.IR_datasets import IR_data
from aimos.misc.utils import *
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier #Shift + tab will show detains of the classifier
from sklearn.svm import SVC
import datetime

def hyperparameter_tuning(model_to_set, parameters, train_X, train_y ):
    '''
    A function to calculate scores over a range of soft margin constant values.
    '''
    t_start = datetime.datetime.now()

    model_tuning = GridSearchCV(model_to_set, param_grid=parameters,
                                 scoring='f1_weighted')

    model_tuning.fit(train_X, train_y)

    print(model_to_set.estimator.kernel)
    print(model_to_set.estimator)
    print(model_tuning.best_score_)
    print(model_tuning.best_params_)
    
    #sorted(model_tunning.cv_results_.keys())
    
    t_end = datetime.datetime.now()
    delta = t_end - t_start
    
    print('Time elaspsed (s): ', delta.total_seconds()) 

    
    return model_tuning.cv_results_['mean_test_score']