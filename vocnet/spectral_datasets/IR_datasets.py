#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: M Arshad Zahangir Chowdhury

Creates IR spectra dataset for use by various functions and subroutines
"""

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


from random import seed, gauss


# if '../' not in sys.path:
#     sys.path.append('../')
    
if '../../' not in sys.path:
    sys.path.append('../../')


class IR_data:
    '''
    
    A class to handle IR dataset consisting of different resolutions.
    
    resolution : float, 0.016, 0.001, 0.0001, 0.00004 1/cm are available resolutions
    verbosity : boolean, to get description
    cv_type: string, 'none', 'pressure', 'concentration'
    
    '''
    
    def __init__(self, data_start = 400, data_end = 4000, resolution = 1, verbosity = False, cv_type='none'):
        
        self.resolution = float(resolution)
        self.data_start = data_start
        self.data_end = data_end
        self.cv_type = cv_type
        
        self.labels = labels = ['H2O','CO2','O3','N2O','CO',
         'CH4','NO','SO2','NO2','NH3',
          'HNO3','HF','HCl','HBr','HI',
          'OCS','H2CO','HOCl','HCN','CH3Cl',
          'H2O2','C2H2','C2H6','PH3','H2S',
          'HCOOH','C2H4','CH3OH','CH3Br','CH3CN',
          'C4H2','HC3N','SO3','COCl2'
         ]
        self.label_id = np.array([0,1,2,3,4,
                             5,6,7,8,9,
                             10,11,12,13,14,
                            15,16,17,18,19,
                            20,21,22,23,24,
                            25,26,27,28,29,
                            30,31,32,33])
        self.n_compounds= 34 # total no. of compounds
        
        
        if self.cv_type == 'none':
            self.n_spectrum= 42 # total no. of individual spectrum for a single compound
        
        elif self.cv_type == 'pressure':
            self.n_spectrum= 10 
        
        elif self.cv_type == 'concentration':
            self.n_spectrum= 7 
        
        else :

            raise ValueError('cv_type must be none, pressure or concentration.')

        
        self.n_spectra = self.n_compounds* self.n_spectrum
        
        self.front_trim_amount = (self.data_start - 400)/self.resolution #float, convert to int
        print("Front trim :", self.front_trim_amount)

        self.end_trim_amount = (4000- self.data_end)/self.resolution #float, convert to int
        print("End trim :", self.end_trim_amount)

        
        #experimental spectra
        self.n_exp_compounds=6 # total no. of compounds
        self.n_exp_spectrum=6 # total no. of individual spectrum for a single compound
        self.n_exp_spectra=self.n_exp_compounds*self.n_exp_spectrum # Total number of spectra

        
        self.verbosity = verbosity
        
        
        


           

    def load_IR_data(self) :
        ''' 
        
        Loads the IR spectra and the targets based on the resolution and training type.
        
        '''


        verbose = self.verbosity

        pathMaster1 = r'../../data/IR_DATA_0d01wvnstep_10um_decades_Ex' #Resoliution = 0.01 per cm
        pathMaster2 = r'../../data/IR_DATA_0d1wvnstep_10um_decades_Ex' #Resoliution = 0.1 per cm
        pathMaster3 = r'../../data/IR_DATA_1wvnstep_10um_decades_Ex' #Resoliution = 1 per cm
        # pathMaster4=r'../../data/IR_DATA_2wvnstep_10um_decades' #Resoliution = 2 per cm
        # pathMaster5=r'../../data/IR_DATA_4wvnstep_10um_decades' #Resoliution = 4 per cm
        # pathMaster6=r'../../data/IR_DATA_8wvnstep_10um_decades' #Resoliution = 8 per cm
        pathMaster_CV_P = r'../../data/IR_DATA_1wvnstep_10um_decades_CV_P' #Resoliution = 1 per cm
        pathMaster_CV_C = r'../../data/IR_DATA_1wvnstep_10um_decades_CV_X' #Resoliution = 1 per cm

        

        path_H2O = r'/H2O'
        path_CO2 = r'/CO2'
        path_O3 = r'/O3'
        path_N2O = r'/N2O'
        path_CO = r'/CO'
        path_CH4 = r'/CH4'
        path_NO = r'/NO'
        path_SO2 = r'/SO2'
        path_NO2 = r'/NO2'
        path_NH3 = r'/NH3'
        path_HNO3 = r'/HNO3'
        path_HF = r'/HF'
        path_HCl = r'/HCl'
        path_HBr = r'/HBr'
        path_HI = r'/HI'
        path_OCS = r'/OCS'
        path_H2CO = r'/H2CO'
        path_HOCl = r'/HOCl'
        path_HCN = r'/HCN'
        path_CH3Cl = r'/CH3Cl'
        path_H2O2 = r'/H2O2'
        path_C2H2 = r'/C2H2'
        path_C2H6 = r'/C2H6'
        path_PH3 = r'/PH3'
        path_H2S = r'/H2S'
        path_HCOOH = r'/HCOOH'
        path_C2H4 = r'/C2H4'
        path_CH3OH = r'/CH3OH'
        path_CH3Br = r'/CH3Br'
        path_CH3CN = r'/CH3CN'
        path_C4H2 = r'/C4H2'
        path_HC3N = r'/HC3N'
        path_SO3 = r'/SO3'
        path_COCl2 = r'/COCl2'
        
        
        t_start = datetime.datetime.now()

        
            
        

        if self.resolution == 0.01:
            pathMaster=pathMaster1
            n_discard_rows=0
            samplesize=360001-n_discard_rows
            if self.cv_type == 'pressure' or self.cv_type == 'concentration' :
                raise ValueError('Only resolution of 1 (1/cm) for cross-validation is available')

        elif self.resolution == 0.1:
            pathMaster=pathMaster2
            n_discard_rows=0
            samplesize=36001-n_discard_rows 
            if self.cv_type == 'pressure' or self.cv_type == 'concentration' :
                raise ValueError('Only resolution of 1 (1/cm) for cross-validation is available')

        elif self.resolution == 1:
            pathMaster=pathMaster3
            n_discard_rows=0
            samplesize=3601-n_discard_rows 
            
            #change path if cross-validation is required
            
            if self.cv_type == 'pressure':
                pathMaster=pathMaster_CV_P
            if self.cv_type == 'concentration':
                pathMaster=pathMaster_CV_C
                

        elif self.resolution == 2:
            raise ValueError('Resolution too coarse and is not implemented')
        elif self.resolution == 4:
            raise ValueError('Resolution too coarse and is not implemented')
        elif self.resolution == 8:
            raise ValueError('Resolution too coarse and is not implemented')
        
        else :

            raise ValueError('Resolution not available')


        self.samplesize = samplesize # get samplesize here
        
        
        
        if verbose == True :
            print('Number of Compounds:', self.n_compounds)
            print('Number of Spectrum:', self.n_spectrum)
            print('Total Number of Spectra:', self.n_spectra)
            print("Front trim :", self.front_trim_amount)
            print("End trim :", self.end_trim_amount)
            print('Data Start Input:',self.data_start)
            print('Data End Input:',self.data_end)           
            print('Sample Size of training data:', self.samplesize )
            print('Rows discarded:', n_discard_rows)
            print('Resolution (1/cm) = ', self.resolution)
            
        front_trim_amount = self.front_trim_amount
        end_trim_amount = self.end_trim_amount


        path=pathMaster+path_H2O
        if verbose == True :
            print('Loading H2O... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []
        idx=0
        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first n_discard_rows rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
        #     print(filename)
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]


            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     plt.plot(dfx,dfy)
        #     plt.show()

        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
        #     define temporary for first loop only
            if idx==0:
                Temporary = np.empty((self.n_spectra, 1, shoretend_samplesize))

            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('H2O Data in Memory ')    

        
        path=pathMaster+path_CO2
        if verbose == True :
            print('Loading CO2... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]


            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('CO2 Data in Memory ')    


        path=pathMaster+path_O3
        if verbose == True :
            print('Loading O3... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('O3 Data in Memory ')    

        #add another compound, don't change idx value


        path=pathMaster+path_N2O
        if verbose == True :
            print('Loading N2O... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('N2O Data in Memory ')    

        

        path=pathMaster+path_CO
        if verbose == True :
            print('Loading CO... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('CO Data in Memory ')    

        

        path=pathMaster+path_CH4
        if verbose == True :
            print('Loading CH4... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('CH4 Data in Memory ')    


        path=pathMaster+path_NO
        if verbose == True :
            print('Loading NO... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            ##convert string array to float
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('NO Data in Memory ')    

        
        path=pathMaster+path_SO2
        if verbose == True :
            print('Loading SO2... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            ##convert string array to float
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1


        if verbose == True :
            print('SO2 Data in Memory ')        

        path=pathMaster+path_NO2
        if verbose == True :
            print('Loading NO2... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1
            
        if verbose == True :
            print('NO2 Data in Memory ') 


        #path = r'C:\Users\Arshad Chowdhury\March ML Codes\HAPI_based_Codes\Uniform Training Data Small\C2H5OH' # vary compound folder name
        path=pathMaster+path_NH3
        if verbose == True :
            print('Loading NH3... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1
        
        if verbose == True :
            print('NH3 Data in Memory ') 


        #path = r'C:\Users\Arshad Chowdhury\March ML Codes\HAPI_based_Codes\Uniform Training Data Small\CH3CHO' # vary compound folder name

        path=pathMaster+path_HNO3
        if verbose == True :
            print('Loading HNO3... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('HNO3 Data in Memory ') 


        path=pathMaster+path_HF
        if verbose == True :
            print('Loading HF... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('HF Data in Memory ') 

        path=pathMaster+path_HCl
        if verbose == True :
            print('Loading HCl... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('HCl Data in Memory ') 

        path=pathMaster+path_HBr
        if verbose == True :
            print('Loading HF... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('HBr Data in Memory ') 

        path=pathMaster+path_HI
        if verbose == True :
            print('Loading HI... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('HI Data in Memory ') 

        path=pathMaster+path_OCS
        if verbose == True :
            print('Loading OCS... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('OCS Data in Memory ') 


        path=pathMaster+path_H2CO
        if verbose == True :
            print('Loading H2CO... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)

            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('H2CO Data in Memory ') 

        path=pathMaster+path_HOCl
        if verbose == True :
            print('Loading HOCl... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('HOCl Data in Memory ') 




        path=pathMaster+path_HCN
        if verbose == True :
            print('Loading HCN... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('HCN Data in Memory ') 

        path=pathMaster+path_CH3Cl
        if verbose == True :
            print('Loading CH3Cl... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('CH3Cl Data in Memory ') 

        path=pathMaster+path_H2O2
        if verbose == True :
            print('Loading H2O2... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('H2O2 Data in Memory ') 


        path=pathMaster+path_C2H2
        if verbose == True :
            print('Loading C2H2... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('C2H2 Data in Memory ') 

        path=pathMaster+path_C2H6
        if verbose == True :
            print('Loading C2H6... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('C2H6 Data in Memory ') 

        path=pathMaster+path_PH3
        if verbose == True :
            print('Loading PH3... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1
        
        if verbose == True :
            print('PH3 Data in Memory ') 

        path=pathMaster+path_H2S
        if verbose == True :
            print('Loading H2S... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('H2S Data in Memory ') 

        path=pathMaster+path_HCOOH
        if verbose == True :
            print('Loading HCOOH... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('HCOOH Data in Memory ') 

        path=pathMaster+path_C2H4
        if verbose == True :
            print('Loading C2H4... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('C2H4 Data in Memory ') 


        path=pathMaster+path_CH3OH
        if verbose == True :
            print('Loading CH3OH... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :    
            print('CH3OH Data in Memory ')

        path=pathMaster+path_CH3Br
        if verbose == True :
            print('Loading CH3Br... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1
            
        if verbose == True :
            print('CH3Br Data in Memory ') 

        path=pathMaster+path_CH3CN
        if verbose == True :
            print('Loading CH3CN... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1
            
        if verbose == True :
            print('CH3CN Data in Memory ') 

        path=pathMaster+path_C4H2
        if verbose == True :
            print('Loading C4H2... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1
        
        if verbose == True :
            print('C4H2 Data in Memory ') 

        path=pathMaster+path_HC3N
        if verbose == True :
            print('Loading HC3N... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1
            
        if verbose == True :
            print('HC3N Data in Memory ') 

        path=pathMaster+path_SO3
        if verbose == True :
            print('Loading SO3... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('SO3 Data in Memory ') 

        path=pathMaster+path_COCl2
        if verbose == True :
            print('Loading COCl2... ')    
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[n_discard_rows:]  #removes first 33 rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            #Trim from the end first
            dfx_store=dfx[:int(samplesize-end_trim_amount)]
            dfx=dfx_store[int(front_trim_amount):]
            dfy_store=dfy[:int(samplesize-end_trim_amount)]
            dfy=dfy_store[int(front_trim_amount):]

            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
        #     print('Original samplesize:',samplesize)
            shoretend_samplesize=len(dfx)
        #     print('Corrected samplesize:',shoretend_samplesize)
            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, shoretend_samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('COCl2 Data in Memory ') 

        t_end = datetime.datetime.now()
        delta = t_end - t_start
        Time_Loading=delta.total_seconds() * 1000
        
        if verbose == True :
            print('Data loading time (milliseconds):', Time_Loading)
        
          

        X=Temporary.reshape(self.n_spectra, shoretend_samplesize)

        
        store_y=np.array([])
        for ele_in_label_id in np.nditer(self.label_id): 
            #print(ele) #prints the array element itself
            for elem_images in range(self.n_spectrum):

                store_y=np.append(store_y, ele_in_label_id)


        y=store_y.astype(np.int)   # label data must be integer
        
        if verbose == True :
            print('Original samplesize:',samplesize)
            print('Corrected samplesize:',shoretend_samplesize)    
        
        if verbose == True :
            
            print('shape of spectra (features):', X.shape)
            print('shape of labels:', y.shape)
        
        self.spectra = X
        self.targets = y
        self.frequencies = dfx
        self.samplesize = samplesize
        self.shoretend_samplesize = shoretend_samplesize
        self.n_discard_rows = n_discard_rows
        
        
    def add_sinusoidal_noise(self):
        '''Add random sinusoidal noise.'''
        seed(1)
        
        Rand_multiplier = gauss(0, 1)/(gauss(0, 1)+1)
        Rand_shifter = gauss(0, 1)


        print(Rand_multiplier)
        print(Rand_shifter)

#         SINE_NOISE=np.pi*np.ones((1001,), dtype=int)*np.sin(Rand_shifter*np.pi)
#         SINE_NOISE.shape

#         # Noise=Validation_X-Validation_X
#         # Noise.shape

#         print(SINE_NOISE)
#         print(SINE_NOISE.shape)

        self.val_sim_spectra = Rand_multiplier*self.spectra + 0.01*Rand_shifter*np.sin(np.ones((len(self.frequencies)),)/(2*np.pi))
        
#         print((gauss(0, 1)/(gauss(0, 1)+1)))
#         print(gauss(0, 1))
#         self.val_sim_spectra=(gauss(0, 1)/(gauss(0, 1)+1))*self.spectra + 0.01*gauss(0, 1)*np.sin(np.ones((len(self.frequencies)),)/(2*np.pi))


    def drop_compounds(self, compounds):
        '''
        A function to drop compounds from the dataset.

        compounds:string, list of compounds to remove from the dataset.  
        ''' 
        SpectraFrame = pd.DataFrame(self.spectra)
        SpectraFrame['labels'] = [self.labels[i] for i in self.targets]
        SpectraFrame['targets'] =  self.targets

        #drop rows that contain any value in the list and then reset index
        df = SpectraFrame[SpectraFrame['labels'].isin(compounds) == False].reset_index()


        store_y=np.array([])
        for ele_in_label_id in np.nditer(range(len(np.unique(df['labels'].to_list()).tolist()))): 
            #print(ele) #prints the array element itself
            for elem_images in range(self.n_spectrum):

                store_y=np.append(store_y, ele_in_label_id)


        y=store_y.astype(int)   # label data must be integer
        df['targets'] = y

        self.n_compounds = self.n_compounds - len(compounds)
        self.n_spectra = self.n_compounds* self.n_spectrum


        self.spectra = df.drop(['labels','targets'], axis=1).to_numpy()[:,1:] # the first column in the np array is indices from pandas which are removed
        self.targets = df['targets'].to_numpy()
        self.labels = pd.unique(df['labels']).tolist()
        self.label_id = range(len(pd.unique(df['labels']).tolist()))

    def make_dataframe(self, spectra):
        '''
        Returns a dataframe.  
        ''' 
        SpectraFrame = pd.DataFrame(spectra)
        SpectraFrame['labels'] = [self.labels[i] for i in self.targets]
        SpectraFrame['targets'] =  self.targets
        
        self.spectraframe = SpectraFrame 
   

    def dataset_info(self):
        '''
        returns useful information about the dataset
        '''
        
        print('Number of Compounds:', self.n_compounds)
        print('Number of Spectrum:', self.n_spectrum)
        print('Total Number of Spectra:', self.n_spectra)
        print("Front trim :", self.front_trim_amount)
        print("End trim :", self.end_trim_amount)
        print('Data Start Input:',self.data_start)
        print('Data End Input:',self.data_end)           
        print('Sample Size of training data:', self.samplesize )
        print('Rows discarded:', self.n_discard_rows)
        print('Resolution (1/cm) = ', self.resolution)

        print('\n labels of molecules present \n', self.labels)
        print('\n target indices (integers) of molecules present', self.targets)
        print('\n frequencies present in the data \n ', self.frequencies)

def spectra_to_img(X, resamplesize,n_channel):
    X_grid=np.zeros((X.shape[0],resamplesize, resamplesize, n_channel))
    for idx in range(X.shape[0]):
        X_grid[idx] = signal.resample(X[idx], resamplesize*resamplesize*n_channel).reshape(resamplesize, resamplesize, n_channel)
        
    return X_grid        


         
if __name__ == "__main__":
    
    print('just some classes and functions!')