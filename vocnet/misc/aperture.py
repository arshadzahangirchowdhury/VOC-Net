#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: M Arshad Zahangir Chowdhury

Useful functions for creating high quality plots of spectra. 
"""

import numpy as np
import pandas as pd
import sys
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pylab import cm
import seaborn as sns  
import glob 
from scipy import signal


def publication_fig(frequencies, 
                   absorbances, 
                   xlim_low,
                   xlim_high, 
                   ylim_low,
                   ylim_high,
                   fig_prop = {'figsize':(2,0.7), 'dpi':300, 'ax_rect': [0,0,1,1]},
                   plot_prop = {'linewidth':0.25, 'color':'black' , 'fontweight':'bold', 'label' : 'compound'},
                   legend_prop = {'loc':2, 'size':6},
                   major_tick_params =  {'which':'major', 'size':6, 'width':1, 'direction':'out','labelsize':8},
                   minor_tick_params =  {'which':'minor', 'size':2, 'width':1, 'direction':'out','labelsize':8},
                   tick_locator = {'xmajor':0.5, 'xminor':1.5, 'ymajor':0.2, 'yminor':0.4},
                   ylabelinfo = {'ylabel': r'Norm. Abs.', 'labelpad':4, 'fontsize':'medium', 'fontweight':'bold'},
                   xlabelinfo = {'xlabel': r'Frequency, $\nu$ ($cm^{-1}$)', 'labelpad':4, 'fontsize':'medium', 'fontweight':'bold'},
                   twin_xlabelinfo = {'xlabel': 'Wavelength ($mm$)', 'labelpad':4, 'fontsize':'medium', 'fontweight':'bold'}, 
                   plot_flags = {'twinx': True, 'legend':True}
                   ):
    
    plt.rc('font', weight=plot_prop['fontweight'])
    fig = plt.figure(figsize=fig_prop['figsize'],dpi=fig_prop['dpi'])
    ax = fig.add_axes(fig_prop['ax_rect'])
    ax.plot(frequencies, absorbances/max(absorbances), linewidth=plot_prop['linewidth'], color=plot_prop['color'], label=plot_prop['label'])
    
    if plot_flags['legend'] == True:
        ax.legend(loc=legend_prop['loc'],prop={'size': legend_prop['size']})
    
    
    ax.set_xlim(xlim_low, xlim_high)
    ax.set_ylim(ylim_low, ylim_high)
    
    
    ax.xaxis.set_tick_params(which=major_tick_params['which'], 
                             size=major_tick_params['size'], 
                             width=major_tick_params['width'], 
                             direction=major_tick_params['direction'],
                             labelsize=major_tick_params['labelsize'])
    
    ax.xaxis.set_tick_params(which=minor_tick_params['which'], 
                             size=minor_tick_params['size'], 
                             width=minor_tick_params['width'], 
                             direction=minor_tick_params['direction'],
                             labelsize=minor_tick_params['labelsize'])

    ax.yaxis.set_tick_params(which=major_tick_params['which'], 
                             size=major_tick_params['size'], 
                             width=major_tick_params['width'], 
                             direction=major_tick_params['direction'],
                             labelsize=major_tick_params['labelsize'])
    
    ax.yaxis.set_tick_params(which=minor_tick_params['which'], 
                             size=minor_tick_params['size'], 
                             width=minor_tick_params['width'], 
                             direction=minor_tick_params['direction'],
                             labelsize=minor_tick_params['labelsize'])
    
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(tick_locator['xmajor']))
    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(tick_locator['xminor']))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(tick_locator['ymajor']))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(tick_locator['yminor']))
    
    ax.set_ylabel(ylabelinfo['ylabel'], 
                  labelpad = ylabelinfo['labelpad'], 
                  fontsize = ylabelinfo['fontsize'], 
                  fontweight = ylabelinfo['fontweight'])
    
    
    ax.set_xlabel(xlabelinfo['xlabel'], 
                  labelpad = xlabelinfo['labelpad'], 
                  fontsize = xlabelinfo['fontsize'], 
                  fontweight = xlabelinfo['fontweight'])
    
    if plot_flags['twinx'] == True:
    
        # Create new axes object by cloning the y-axis of the first plot
        ax2 = ax.twiny()
        # Edit the tick parameters of the new x-axis
        ax2.xaxis.set_tick_params(which=major_tick_params['which'], 
                                 size=major_tick_params['size'], 
                                 width=major_tick_params['width'], 
                                 direction=major_tick_params['direction'],
                                 labelsize=major_tick_params['labelsize'])

        ax2.xaxis.set_tick_params(which=minor_tick_params['which'], 
                                 size=minor_tick_params['size'], 
                                 width=minor_tick_params['width'], 
                                 direction=minor_tick_params['direction'],
                                 labelsize=minor_tick_params['labelsize'])


        ax2.set_xlabel(twin_xlabelinfo['xlabel'], 
                      labelpad = twin_xlabelinfo['labelpad'], 
                      fontsize = twin_xlabelinfo['fontsize'], 
                      fontweight = twin_xlabelinfo['fontweight'])

        ax2.set_xlim(xlim_high/8, xlim_low/8)

        plt.close()
    
    
    return fig, ax


