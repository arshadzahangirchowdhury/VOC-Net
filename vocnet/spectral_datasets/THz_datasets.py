#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: M Arshad Zahangir Chowdhury

Creates THz spectra dataset for use by various functions and subroutines
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

# if '../' not in sys.path:
#     sys.path.append('../')
    
if '../../' not in sys.path:
    sys.path.append('../../')


class THz_data:
    '''
    
    A class to handle THz dataset consisting of four different resolutions.
    
    resolution : float, 0.016, 0.001, 0.0001, 0.00004 1/cm are available resolutions
    verbosity : boolean, to get description
    
    '''
    
    def __init__(self, resolution=0.016, verbosity = False):
        self.resolution = resolution
        self.labels = ['CH3Cl', 'CH3OH', 'HCOOH', 'H2CO', 'H2S', 'SO2','OCS','HCN','CH3CN','HNO3','C2H5OH','CH3CHO']
        self.label_id = np.array([0,1,2,3,4,5,6,7,8,9,10,11]) 
        self.n_compounds=12 # total no. of compounds
        self.n_spectrum=164 # total no. of individual spectrum for a single compound
        self.n_spectra = self.n_spectrum*self.n_compounds
        #experimental spectra
        self.n_exp_compounds=6 # total no. of compounds
        self.n_exp_spectrum=6 # total no. of individual spectrum for a single compound
        self.n_exp_spectra=self.n_exp_compounds*self.n_exp_spectrum # Total number of spectra

        
        self.verbosity = verbosity
    

    def load_THz_data(self) :
        ''' 
        
        Loads the THz spectra and the targets based on the resolution
        
        '''


        verbose = self.verbosity
    
        pathMaster1 = r'../../data/Uniform Training Data Small' #Resoliution =0.016
        pathMaster2 = r'../../data/Uniform Training Data Medium' #Resoliution =0.001
        pathMaster3 = r'../../data/Uniform Training Data Less Fine' #Resoliution =0.0001
        pathMaster4 = r'../../data/Uniform Training Data Fine' #Resoliution =0.00004

        path_CH3Cl = r'/CH3Cl'
        path_CH3OH = r'/CH3OH'
        path_HCOOH = r'/HCOOH'
        path_H2CO = r'/H2CO'
        path_H2S = r'/H2S'
        path_SO2 = r'/SO2'
        path_OCS = r'/OCS'
        path_HCN = r'/HCN'
        path_CH3CN = r'/CH3CN'
        path_HNO3 = r'/HNO3'
        path_C2H5OH = r'/C2H5OH'
        path_CH3CHO = r'/CH3CHO'


        if self.resolution == 0.016:
            pathMaster=pathMaster1
            #pathmaster1
            n_discard_rows=22
            samplesize =251-n_discard_rows #discarding first 22 rows so we can get data start from nu=7.33

        elif self.resolution == 0.001:
            pathMaster=pathMaster2
            n_discard_rows=331
            samplesize =4001-n_discard_rows #discarding first 331 rows so we can get data start from nu=7.33

        elif self.resolution == 0.0001:
            pathMaster=pathMaster3
            n_discard_rows=3301
            samplesize =40001-n_discard_rows #discarding first 3301 rows so we can get data start from nu=7.33


        elif self.resolution == 0.00004:    
            pathMaster=pathMaster4
            n_discard_rows=8251
            samplesize =100001-n_discard_rows #discarding first 8251 rows so we can get data start from nu=7.33

        else :

            raise ValueError('Resolution not available')


        self.samplesize = samplesize # get samplesize here
        
        n_spectra=self.n_compounds*self.n_spectrum # Total number of spectra
        
        if verbose == True :
            print('Number of Compounds:', self.n_compounds)
            print('Number of Spectrum:', self.n_spectrum)
            print('Total Number of Spectra:', n_spectra)
            print('Sample Size of training data:', self.samplesize )
            print('Rows discarded:', n_discard_rows)
            print('Resolution (1/cm) = ', self.resolution)


        
        
        Temporary = np.empty((n_spectra, 1, samplesize))


        path=pathMaster+path_CH3Cl
        
        if verbose == True :
            print('Loading CH3Cl... ')    
            
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
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)

            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, samplesize ) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1
        
        if verbose == True :
            print('CH3Cl Data in Memory ')    



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
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)

            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1
        
        if verbose == True :
            print('CH3OH Data in Memory ')    


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
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)

            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, samplesize ) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1
        
        if verbose == True :
            print('HCOOH Data in Memory ')    

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
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)

            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, samplesize ) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('H2CO Data in Memory ')    

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
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)

            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, samplesize ) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1
        
        if verbose == True :
            print('H2S Data in Memory ')    

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
            ##convert string array to float
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)

            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, samplesize ) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1
        
        if verbose == True :
            print('SO2 Data in Memory ')    

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
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)

            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, samplesize ) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1
        
        if verbose == True :
            print('OCS Data in Memory ')    


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
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)

            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, samplesize ) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1
        
        if verbose == True :
            print('HCN Data in Memory ')    

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
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)

            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, samplesize ) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        if verbose == True :
            print('CH3CN Data in Memory ')        


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
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)

            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, samplesize ) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1
            
        if verbose == True :
            print('HNO3 Data in Memory ') 


        path=pathMaster+path_C2H5OH
        
        if verbose == True :
            print('Loading C2H5OH... ')    
        
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
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)

            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, samplesize ) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1
            
        if verbose == True :    
            print('C2H5OH Data in Memory ') 



        path=pathMaster+path_CH3CHO
        
        if verbose == True :
            print('Loading CH3CHO... ')    
        
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
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)

            #print(idx)  #for debug
            Temporary[idx]=dfy.reshape(1, samplesize ) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1
        
        if verbose == True :
            print('CH3CHO Data in Memory ')    

        X=Temporary.reshape(n_spectra, samplesize)


        store_y=np.array([])
        for ele_in_label_id in np.nditer(self.label_id): 
            #print(ele) #prints the array element itself
            for elem_images in range(self.n_spectrum):

                store_y=np.append(store_y, ele_in_label_id)


        y=store_y.astype(np.int)   # label data must be integer
        
        
        if verbose == True :
            
            print('shape of spectra (features):', X.shape)
            print('shape of labels:', y.shape)
        
        self.spectra = X
        self.targets = y
        self.frequencies = dfx
        self.samplesize = samplesize
        self.n_discard_rows = n_discard_rows
        
        self.CH3Cl_spectra = self.spectra[int(0*self.n_spectrum):int(1*self.n_spectrum)]
        self.CH3Cl_targets = self.targets[int(0*self.n_spectrum):int(1*self.n_spectrum)]

        self.CH3OH_spectra = self.spectra[int(1*self.n_spectrum):int(2*self.n_spectrum)]
        self.CH3OH_targets = self.targets[int(1*self.n_spectrum):int(2*self.n_spectrum)]

        self.HCOOH_spectra = self.spectra[int(2*self.n_spectrum):int(3*self.n_spectrum)]
        self.HCOOH_targets = self.targets[int(2*self.n_spectrum):int(3*self.n_spectrum)]

        self.H2CO_spectra = self.spectra[int(3*self.n_spectrum):int(4*self.n_spectrum)]
        self.H2CO_targets = self.targets[int(3*self.n_spectrum):int(4*self.n_spectrum)]

        self.H2S_spectra = self.spectra[int(4*self.n_spectrum):int(5*self.n_spectrum)]
        self.H2S_targets = self.targets[int(4*self.n_spectrum):int(5*self.n_spectrum)]

        self.SO2_spectra = self.spectra[int(5*self.n_spectrum):int(6*self.n_spectrum)]
        self.SO2_targets = self.targets[int(5*self.n_spectrum):int(6*self.n_spectrum)]

        self.OCS_spectra = self.spectra[int(6*self.n_spectrum):int(7*self.n_spectrum)]
        self.OCS_targets = self.targets[int(6*self.n_spectrum):int(7*self.n_spectrum)]

        self.HCN_spectra = self.spectra[int(7*self.n_spectrum):int(8*self.n_spectrum)]
        self.HCN_targets = self.targets[int(7*self.n_spectrum):int(8*self.n_spectrum)]

        self.CH3CN_spectra = self.spectra[int(8*self.n_spectrum):int(9*self.n_spectrum)]
        self.CH3CN_targets = self.targets[int(8*self.n_spectrum):int(9*self.n_spectrum)]

        self.HNO3_spectra = self.spectra[int(9*self.n_spectrum):int(10*self.n_spectrum)]
        self.HNO3_targets = self.targets[int(9*self.n_spectrum):int(10*self.n_spectrum)]

        self.C2H5OH_spectra = self.spectra[int(10*self.n_spectrum):int(11*self.n_spectrum)]
        self.C2H5OH_targets = self.targets[int(10*self.n_spectrum):int(11*self.n_spectrum)]

        self.CH3CHO_spectra = self.spectra[int(11*self.n_spectrum):int(12*self.n_spectrum)]
        self.CH3CHO_targets = self.targets[int(11*self.n_spectrum):int(12*self.n_spectrum)]


    

    def load_experiments(self, verbosity=False) :
        ''' 
        
        Loads the experimental dataset in the THz range. This function can only be executed once the simulated training data is loaded so that the experiments can be resampled to the same samplesize as the simulations.
        
        verbosity:bool, print dataset information
        
        '''

        n_exp_compounds=self.n_exp_compounds # total no. of compounds
        n_exp_spectrum=self.n_exp_spectrum # total no. of individual spectrum for a single compound
        n_exp_spectra=self.n_exp_spectra # Total number of spectra
        
        print('Number of Experimental Compounds:', self.n_exp_compounds)
        print('Number of Spectrum:', self.n_exp_spectrum)
        print('Total Number of Spectra:', self.n_exp_spectra)
        
        print('Sample Size of training data:', self.samplesize)
        print('Rows discarded:', self.n_discard_rows)
        
        

        Experimental = np.empty((n_exp_spectra, 1, self.samplesize))
        
        list_filenames = []

        idx=0
        path = r'../../data/AppliedPhysicsB_Matrix/SomeNoisySomeFilteredEthanol' # vary compound folder name

        all_files = glob.glob(path + "/*.csv")

        for filename in all_files:
            fileName_absolute = os.path.basename(filename)
            #print(fileName_absolute)
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[self.n_discard_rows:]  #removes first n_discard_rows rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
            #resample first then assemble
            if verbosity == True:
                print('Loaded Filtered Exp. Ethanol spectrum ',idx,' file:',fileName_absolute) 

            dfy_resampled= signal.resample(dfy, self.samplesize)
            dfx_resampled= signal.resample(dfx, self.samplesize)

            #print(idx)  #for debug
            list_filenames.append(fileName_absolute)
            Experimental[idx]=dfy_resampled.reshape(1, self.samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        path = r'../../data/AppliedPhysicsB_Matrix/SomeNoisySomeFilteredFormic' # vary compound folder name

        all_files = glob.glob(path + "/*.csv")

        for filename in all_files:
            fileName_absolute = os.path.basename(filename)
            #print(fileName_absolute)
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[self.n_discard_rows:]  #removes first n_discard_rows rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
            #resample first then assemble
            if verbosity == True:
                print('Loaded Filtered Exp. Formic Acid spectrum ',idx,' file:',fileName_absolute) 

            
            dfy_resampled= signal.resample(dfy, self.samplesize)
            dfx_resampled= signal.resample(dfx, self.samplesize)

            #print(idx)  #for debug
            list_filenames.append(fileName_absolute)
            Experimental[idx]=dfy_resampled.reshape(1, self.samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        path = r'../../data/AppliedPhysicsB_Matrix/SomeNoisySomeFilteredMethanol' # vary compound folder name

        all_files = glob.glob(path + "/*.csv")

        for filename in all_files:
            fileName_absolute = os.path.basename(filename)
            #print(fileName_absolute)
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[self.n_discard_rows:]  #removes first n_discard_rows rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
            #resample first then assemble
            if verbosity == True:
                print('Loaded Filtered Exp. Methanol spectrum ',idx,' file:',fileName_absolute) 

            
            dfy_resampled= signal.resample(dfy, self.samplesize)
            dfx_resampled= signal.resample(dfx, self.samplesize)

            #print(idx)  #for debug
            list_filenames.append(fileName_absolute)
            Experimental[idx]=dfy_resampled.reshape(1, self.samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1


        path = r'../../data/AppliedPhysicsB_Matrix/NoisyChloromethane' # vary compound folder name

        all_files = glob.glob(path + "/*.csv")

        for filename in all_files:
            fileName_absolute = os.path.basename(filename)
            #print(fileName_absolute)
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[self.n_discard_rows:]  #removes first n_discard_rows rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
            #resample first then assemble
            if verbosity == True:
                print('Loaded Noisy Exp. Chloromethane spectrum ',idx,' file:',fileName_absolute) 

            dfy_resampled= signal.resample(dfy, self.samplesize)
            dfx_resampled= signal.resample(dfx, self.samplesize)

            #print(idx)  #for debug
            list_filenames.append(fileName_absolute)
            Experimental[idx]=dfy_resampled.reshape(1, self.samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        path = r'../../data/AppliedPhysicsB_Matrix/NoisyAcetonitrile' # vary compound folder name

        all_files = glob.glob(path + "/*.csv")

        for filename in all_files:
            fileName_absolute = os.path.basename(filename)
            #print(fileName_absolute)
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[self.n_discard_rows:]  #removes first n_discard_rows rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
            #resample first then assemble
            if verbosity == True:
                print('Loaded Noisy Exp. Acetonitrile spectrum ',idx,' file:',fileName_absolute) 

            
            dfy_resampled= signal.resample(dfy, self.samplesize)
            dfx_resampled= signal.resample(dfx, self.samplesize)

            #print(idx)  #for debug
            list_filenames.append(fileName_absolute)
            Experimental[idx]=dfy_resampled.reshape(1, self.samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1

        path = r'../../data/AppliedPhysicsB_Matrix/SomeNoisySomeFilteredAcetaldehyde' # vary compound folder name

        all_files = glob.glob(path + "/*.csv")

        for filename in all_files:
            fileName_absolute = os.path.basename(filename)
            #print(fileName_absolute)
            df = pd.read_csv(filename, index_col=None, header=0)  
            #print(df) # Turn on to see all files
            #add your code here from above
            #df=df.values[1:]  #removes first 33 rows
            df=df.values[self.n_discard_rows:]  #removes first n_discard_rows rows
            dfx=df[:,0] #get first column
            dfy=df[:,1] #get second column
            ##convert string array to float
            dfx=dfx.astype(np.float)
            #print(dfx)
            dfy=dfy.astype(np.float)
            #resample first then assemble
            if verbosity == True:
                print('Loaded Noisy Exp. Acetaldehyde spectrum ',idx,' file:',fileName_absolute) 

            
            dfy_resampled= signal.resample(dfy, self.samplesize)
            dfx_resampled= signal.resample(dfx, self.samplesize)

            #print(idx)  #for debug
            list_filenames.append(fileName_absolute)
            Experimental[idx]=dfy_resampled.reshape(1, self.samplesize) #Arrange absorbance values as features and keep them in temporary
            idx=idx+1
            
        self.exp_spectra = Experimental.reshape(self.n_exp_spectra, self.samplesize)
        self.list_filenames=list_filenames
        self.exp_targets = np.array([10,10,10,10,10,10, 2,2,2,2,2,2,1,1,1,1,1,1,0,0,0,0,0,0,8,8,8,8,8,8,11,11,11,11,11,11])
        self.labels_exp=['C2H5OH','HCOOH', 'CH3OH','CH3Cl','CH3CN','CH3CHO']
        
        if verbosity == True:
        
            print('no. of experimental compounds', self.n_exp_compounds)
            print('no. of experimental spectrum per compound', self.n_exp_spectrum)
            print('no. of experimental spectra', self.n_exp_spectra)

            print('target indices for experimental spectra', self.exp_targets)
            print('molecule labels in experimental dataset', self.labels_exp)


    
    def filter_by_index(self, start_index,end_index):
        
        '''
        filters each molecule's spectra via indices.
        
        start_index : int, index to begin
        verbosity : int, index to end
        
        '''
    
        if start_index <0 or end_index<0 or start_index>self.n_spectrum or end_index>self.n_spectrum:
            raise ValueError('start or end index is out of range (0, n_spectrum)') 



        self.filtered_CH3Cl_spectra = self.CH3Cl_spectra[start_index:end_index]
        self.filtered_CH3Cl_targets = self.CH3Cl_targets[start_index:end_index]

        self.filtered_CH3OH_spectra = self.CH3OH_spectra[start_index:end_index]
        self.filtered_CH3OH_targets = self.CH3OH_targets[start_index:end_index]

        self.filtered_HCOOH_spectra = self.HCOOH_spectra[start_index:end_index]
        self.filtered_HCOOH_targets = self.HCOOH_targets[start_index:end_index]

        self.filtered_H2CO_spectra = self.H2CO_spectra[start_index:end_index]
        self.filtered_H2CO_targets = self.H2CO_targets[start_index:end_index]

        self.filtered_H2S_spectra = self.H2S_spectra[start_index:end_index]
        self.filtered_H2S_targets = self.H2S_targets[start_index:end_index]

        self.filtered_SO2_spectra = self.SO2_spectra[start_index:end_index]
        self.filtered_SO2_targets = self.SO2_targets[start_index:end_index]

        self.filtered_OCS_spectra = self.OCS_spectra[start_index:end_index]
        self.filtered_OCS_targets = self.OCS_targets[start_index:end_index]

        self.filtered_HCN_spectra = self.HCN_spectra[start_index:end_index]
        self.filtered_HCN_targets = self.HCN_targets[start_index:end_index]

        self.filtered_CH3CN_spectra = self.CH3CN_spectra[start_index:end_index]
        self.filtered_CH3CN_targets = self.CH3CN_targets[start_index:end_index]

        self.filtered_HNO3_spectra = self.HNO3_spectra[start_index:end_index]
        self.filtered_HNO3_targets = self.HNO3_targets[start_index:end_index]

        self.filtered_C2H5OH_spectra = self.C2H5OH_spectra[start_index:end_index]
        self.filtered_C2H5OH_targets = self.C2H5OH_targets[start_index:end_index]

        self.filtered_CH3CHO_spectra = self.CH3CHO_spectra[start_index:end_index]
        self.filtered_CH3CHO_targets = self.CH3CHO_targets[start_index:end_index]

        self.filtered_spectra = np.concatenate((self.filtered_CH3Cl_spectra,
                                               self.filtered_CH3OH_spectra,
                                              self.filtered_HCOOH_spectra,
                                              self.filtered_H2CO_spectra,
                                              self.filtered_H2S_spectra,
                                              self.filtered_SO2_spectra,
                                              self.filtered_OCS_spectra,
                                              self.filtered_HCN_spectra,
                                              self.filtered_CH3CN_spectra,
                                              self.filtered_HNO3_spectra,
                                              self.filtered_C2H5OH_spectra,
                                              self.filtered_CH3CHO_spectra))


        self.filtered_targets = np.concatenate((self.filtered_CH3Cl_targets,
                                               self.filtered_CH3OH_targets,
                                              self.filtered_HCOOH_targets,
                                              self.filtered_H2CO_targets,
                                              self.filtered_H2S_targets,
                                              self.filtered_SO2_targets,
                                              self.filtered_OCS_targets,
                                              self.filtered_HCN_targets,
                                              self.filtered_CH3CN_targets,
                                              self.filtered_HNO3_targets,
                                              self.filtered_C2H5OH_targets,
                                              self.filtered_CH3CHO_targets))
        
        self.filtered_n_spectra = self.filtered_spectra.shape[0]

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
        
        print('\n no. of discarded rows', self.n_discard_rows)
        print('\n samplesize (no. of sampling points)', self.samplesize)
        print('\n labels of molecules present \n', self.labels)
        print('\n target indices (integers) of molecules present', self.targets)
        print('\n frequencies present in the data \n ', self.frequencies)
        

         
if __name__ == "__main__":
    
    print('just some classes and functions!')