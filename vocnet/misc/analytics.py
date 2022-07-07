#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: M Arshad Zahangir Chowdhury

Useful functions for data analytics
"""

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from ipywidgets import interactive
import seaborn as sns  #heat map
import glob # batch processing of images

if '../../' not in sys.path:
    sys.path.append('../../')

def plot_compound_counts(df, title = 'spectra', color = 'blue', save_to_file = True):
    fig, ax = plt.subplots()
    fig.set_figheight(9)
    fig.set_figwidth(16)
    ax = sns.countplot(x="labels", color=color,  data=df)
    ax.set_xlabel('Compound')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 45);
    
    
    plt.tight_layout()
    
    if save_to_file == True:
        plt.savefig('RESULTS/Analytics_Figures/' + title + '.png', dpi=300)
    
    
    
    return fig, ax

def plot_dataset_property(df, y_prop, title = 'spectra',  ylabel = 'Maximum Absorbance', color = 'red', save_to_file = True):    
#     sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots()
    fig.set_figheight(8)
    fig.set_figwidth(16)
    ax = sns.barplot(x='labels', y=y_prop, color=color,  data=df,capsize=.2)
    ax.set_xlabel('Compound')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 45);
    
    if save_to_file == True:
        plt.savefig('RESULTS/Analytics_Figures/' + title + '.png', dpi=300)
        
def load_exp_spectra(path_exp, spectrum_name):
    exp_spectrum = pd.read_excel(path_exp + spectrum_name, sheet_name="Sheet1")
    exp_spectrum.columns = ['wavenumbers', 'absorbance'] 
    
    return exp_spectrum['wavenumbers'].to_numpy(), exp_spectrum['absorbance'].to_numpy()
    