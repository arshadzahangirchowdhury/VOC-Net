import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.ticker import MaxNLocator
import numpy as np
from numpy import asarray
import pandas as pd
import math
import seaborn as sns  #heat map
import glob # batch processing of images


import matplotlib.font_manager as fm
import random
import sys
import os
import datetime
from datetime import date, datetime
from tempfile import TemporaryFile

from sklearn.datasets import make_regression
import tensorflow as tf
from sklearn.metrics import confusion_matrix    #confusion matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Collect all the font names available to matplotlib
font_names = [f.name for f in fm.fontManager.ttflist]
# print(font_names)

from scipy import signal
from scipy import interpolate

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF

#Sklearn model saving and loading
from joblib import dump, load

if '../../' not in sys.path:
    sys.path.append('../../')

from aimos.spectral_datasets.THz_datasets import THz_data
from aimos.misc.utils import simple_plotter


class THz_mixture_data:
    '''
    
    A class to create mixture data using the THz dataset at 0.016 wavenumber resolution.
    basis attribute contains the pure spectra.
    
    resolution : float, 0.016, 0.001, 0.0001, 0.00004 1/cm are available resolutions
    verbosity : boolean, to get description
    
    '''
    
    def __init__(self, resolution = 0.016, pressure = '1 Torr', verbosity = False):
        self.resolution = resolution
        self.pressure = pressure
        self.labels = ['CH3Cl', 'CH3OH', 'HCOOH', 'H2CO', 'H2S', 'SO2','OCS','HCN','CH3CN','HNO3','C2H5OH','CH3CHO']
        self.label_id = np.array([0,1,2,3,4,5,6,7,8,9,10,11]) 
        self.n_compounds=12 # total no. of compounds
        self.n_spectrum=164 # total no. of individual spectrum for a single compound
        self.n_spectra = self.n_spectrum*self.n_compounds
        
        self.n_mixture_component_max = 12
        self.components = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12]]) # extra 1 is for diluent
        

        
        self.verbosity = verbosity
    

    def initiate_THz_mixture_data(self) :
        print('Components : ',self.components)
        print('Components shape : ',self.components.shape)
        
        # load THz data and filter it at the index corresponding to 1 Torr pressure
        
        s = THz_data(resolution=self.resolution, verbosity=self.verbosity)
        s.load_THz_data()
        
        self.frequencies= s.frequencies
        
        
        if self.pressure == '1 Torr':
            s.filter_by_index(6,7)
            
        elif self.pressure == '2 Torr':
            s.filter_by_index(16,17)
        
        elif self.pressure == '3 Torr':
            s.filter_by_index(26,27)
        
        elif self.pressure == '4 Torr':
            s.filter_by_index(36,37)
        
        elif self.pressure == '5 Torr':
            s.filter_by_index(46,47)
        
        elif self.pressure == '6 Torr':
            s.filter_by_index(56,57)
        
        elif self.pressure == '7 Torr':
            s.filter_by_index(66,67)
        
        elif self.pressure == '8 Torr':
            s.filter_by_index(76,77)
        
        elif self.pressure == '9 Torr':
            s.filter_by_index(86,87)
        
        elif self.pressure == '10 Torr':
            s.filter_by_index(96,97)
            
        elif self.pressure == '11 Torr':
            s.filter_by_index(106,107)
            
        elif self.pressure == '12 Torr':
            s.filter_by_index(116,117)
        
        elif self.pressure == '13 Torr':
            s.filter_by_index(126,127)
        
        elif self.pressure == '14 Torr':
            s.filter_by_index(136,137)
        
        elif self.pressure == '15 Torr':
            s.filter_by_index(146,147)
            
        elif self.pressure == '16 Torr':
            s.filter_by_index(156,157)
        
        

        self.basis_C2H5OH=s.filtered_C2H5OH_spectra.reshape(s.samplesize)
        self.basis_CH3CHO=s.filtered_CH3CHO_spectra.reshape(s.samplesize)
        self.basis_CH3Cl=s.filtered_CH3Cl_spectra.reshape(s.samplesize)
        self.basis_CH3CN=s.filtered_CH3CN_spectra.reshape(s.samplesize)
        self.basis_CH3OH=s.filtered_CH3OH_spectra.reshape(s.samplesize)
        self.basis_H2CO=s.filtered_H2CO_spectra.reshape(s.samplesize)
        self.basis_H2S=s.filtered_H2S_spectra.reshape(s.samplesize)
        self.basis_HCN=s.filtered_HCN_spectra.reshape(s.samplesize)
        self.basis_HCOOH=s.filtered_HCOOH_spectra.reshape(s.samplesize)
        self.basis_HNO3=s.filtered_HNO3_spectra.reshape(s.samplesize)
        self.basis_OCS=s.filtered_H2S_spectra.reshape(s.samplesize)
        self.basis_SO2=s.filtered_SO2_spectra.reshape(s.samplesize)
        
        self.n_features = s.samplesize
        
        
        self.basis = np.array([[self.basis_C2H5OH], 
                  [self.basis_CH3CHO],
                  [self.basis_CH3Cl],
                  [self.basis_CH3CN],
                  [self.basis_CH3OH],
                  [self.basis_H2CO],
                  [self.basis_H2S],
                  [self.basis_HCN],
                  [self.basis_HCOOH],
                  [self.basis_HNO3],
                  [self.basis_OCS],
                  [self.basis_SO2]                  
                 ])
        
        self.basis=self.basis.reshape(self.n_compounds, self.n_features)
        
        self.labels = [' ', '', 'Diluent' ,r'$C_2H_5OH$', r'$CH_3CHO$', r'$CH_3Cl$', 
                       r'$CH_3CN$', r'$CH_3OH$', r'$H_2CO$', r'$H_2S$', r'$HCN$',
                       r'$HCOOH$', r'$HNO_3$', r'$OCS$', r'$SO_2$']
        
        
        if self.verbosity == True:
        
            print('labels : ', self.labels)
            print('Basis shape:',self.basis.shape)
        
        
    def make_pure_mixture(self, n_mixture_pure):
        
        '''
        adds a fixed number for each of the random pure component mixtures.
        
        n_mixture_pure: int, number of total pure mixtures 
        
        '''
        
        self.n_mixture_pure=n_mixture_pure

        targets_c1 = np.empty((self.n_mixture_pure,self.n_mixture_component_max),dtype=object)
        dilution_c1 = np.empty((self.n_mixture_pure),dtype=object)

        targets_c2 = np.empty((self.n_mixture_pure,self.n_mixture_component_max),dtype=object)
        dilution_c2 = np.empty((self.n_mixture_pure),dtype=object)

        targets_c3 = np.empty((self.n_mixture_pure,self.n_mixture_component_max),dtype=object)
        dilution_c3 = np.empty((self.n_mixture_pure),dtype=object)

        targets_c4 = np.empty((self.n_mixture_pure,self.n_mixture_component_max),dtype=object)
        dilution_c4 = np.empty((self.n_mixture_pure),dtype=object)

        targets_c5 = np.empty((self.n_mixture_pure,self.n_mixture_component_max),dtype=object)
        dilution_c5 = np.empty((self.n_mixture_pure),dtype=object)

        targets_c6 = np.empty((self.n_mixture_pure,self.n_mixture_component_max),dtype=object)
        dilution_c6 = np.empty((self.n_mixture_pure),dtype=object)

        targets_c7 = np.empty((self.n_mixture_pure,self.n_mixture_component_max),dtype=object)
        dilution_c7 = np.empty((self.n_mixture_pure),dtype=object)

        targets_c8 = np.empty((self.n_mixture_pure,self.n_mixture_component_max),dtype=object)
        dilution_c8 = np.empty((self.n_mixture_pure),dtype=object)

        targets_c9 = np.empty((self.n_mixture_pure,self.n_mixture_component_max),dtype=object)
        dilution_c9 = np.empty((self.n_mixture_pure),dtype=object)

        targets_c10 = np.empty((self.n_mixture_pure,self.n_mixture_component_max),dtype=object)
        dilution_c10 = np.empty((self.n_mixture_pure),dtype=object)

        targets_c11 = np.empty((self.n_mixture_pure,self.n_mixture_component_max),dtype=object)
        dilution_c11 = np.empty((self.n_mixture_pure),dtype=object)

        targets_c12 = np.empty((self.n_mixture_pure,self.n_mixture_component_max),dtype=object)
        dilution_c12 = np.empty((self.n_mixture_pure),dtype=object)

        mixtures_c1 = np.empty((self.n_mixture_pure,1, self.n_features),dtype=object)
        mixtures_c2 = np.empty((self.n_mixture_pure,1, self.n_features),dtype=object)
        mixtures_c3 = np.empty((self.n_mixture_pure,1, self.n_features),dtype=object)
        mixtures_c4 = np.empty((self.n_mixture_pure,1, self.n_features),dtype=object)

        mixtures_c5 = np.empty((self.n_mixture_pure,1, self.n_features),dtype=object)
        mixtures_c6 = np.empty((self.n_mixture_pure,1, self.n_features),dtype=object)
        mixtures_c7 = np.empty((self.n_mixture_pure,1, self.n_features),dtype=object)
        mixtures_c8 = np.empty((self.n_mixture_pure,1, self.n_features),dtype=object)

        mixtures_c9 = np.empty((self.n_mixture_pure,1, self.n_features),dtype=object)
        mixtures_c10 = np.empty((self.n_mixture_pure,1, self.n_features),dtype=object)
        mixtures_c11 = np.empty((self.n_mixture_pure,1, self.n_features),dtype=object)
        mixtures_c12 = np.empty((self.n_mixture_pure,1, self.n_features),dtype=object)


        c1=[1,0,0,0,0,0,0,0,0,0,0,0]
        c2=[0,1,0,0,0,0,0,0,0,0,0,0]
        c3=[0,0,1,0,0,0,0,0,0,0,0,0]
        c4=[0,0,0,1,0,0,0,0,0,0,0,0]

        c5=[0,0,0,0,1,0,0,0,0,0,0,0]
        c6=[0,0,0,0,0,1,0,0,0,0,0,0]
        c7=[0,0,0,0,0,0,1,0,0,0,0,0]
        c8=[0,0,0,0,0,0,0,1,0,0,0,0]

        c9=[0,0,0,0,0,0,0,0,1,0,0,0]
        c10=[0,0,0,0,0,0,0,0,0,1,0,0]
        c11=[0,0,0,0,0,0,0,0,0,0,1,0]
        c12=[0,0,0,0,0,0,0,0,0,0,0,1]

        for j in range(self.n_mixture_pure):
            dilution_c1[j]=np.random.random(1)

            targets_c1[j] = (1-dilution_c1[j])*c1
            targets_c1[j]=targets_c1[j].astype(np.float)
            mixtures_c1[j] = np.dot(self.basis.T,targets_c1[j])


            dilution_c2[j]=np.random.random(1)

            targets_c2[j] = (1-dilution_c2[j])*c2
            targets_c2[j]=targets_c2[j].astype(np.float)
            mixtures_c2[j] = np.dot(self.basis.T,targets_c2[j])

            dilution_c3[j]=np.random.random(1)

            targets_c3[j] = (1-dilution_c3[j])*c3
            targets_c3[j]=targets_c3[j].astype(np.float)
            mixtures_c3[j] = np.dot(self.basis.T,targets_c3[j])

            dilution_c4[j]=np.random.random(1)

            targets_c4[j] = (1-dilution_c4[j])*c4
            targets_c4[j]=targets_c4[j].astype(np.float)
            mixtures_c4[j] = np.dot(self.basis.T,targets_c4[j])

            dilution_c5[j]=np.random.random(1)

            targets_c5[j] = (1-dilution_c5[j])*c5
            targets_c5[j]=targets_c5[j].astype(np.float)
            mixtures_c5[j] = np.dot(self.basis.T,targets_c5[j])

            dilution_c6[j]=np.random.random(1)

            targets_c6[j] = (1-dilution_c6[j])*c6
            targets_c6[j]=targets_c6[j].astype(np.float)
            mixtures_c6[j] = np.dot(self.basis.T,targets_c6[j])

            dilution_c7[j]=np.random.random(1)

            targets_c7[j] = (1-dilution_c7[j])*c7
            targets_c7[j]=targets_c7[j].astype(np.float)
            mixtures_c7[j] = np.dot(self.basis.T,targets_c7[j])

            dilution_c8[j]=np.random.random(1)

            targets_c8[j] = (1-dilution_c8[j])*c8
            targets_c8[j]=targets_c8[j].astype(np.float)
            mixtures_c8[j] = np.dot(self.basis.T,targets_c8[j])

            dilution_c9[j]=np.random.random(1)

            targets_c9[j] = (1-dilution_c9[j])*c9
            targets_c9[j]=targets_c9[j].astype(np.float)
            mixtures_c9[j] = np.dot(self.basis.T,targets_c9[j])

            dilution_c10[j]=np.random.random(1)

            targets_c10[j] = (1-dilution_c10[j])*c10
            targets_c10[j]=targets_c10[j].astype(np.float)
            mixtures_c10[j] = np.dot(self.basis.T,targets_c10[j])

            dilution_c11[j]=np.random.random(1)

            targets_c11[j] = (1-dilution_c11[j])*c11
            targets_c11[j]=targets_c11[j].astype(np.float)
            mixtures_c11[j] = np.dot(self.basis.T,targets_c11[j])

            dilution_c12[j]=np.random.random(1)

            targets_c12[j] = (1-dilution_c12[j])*c12
            targets_c12[j]=targets_c12[j].astype(np.float)
            mixtures_c12[j] = np.dot(self.basis.T,targets_c12[j])


#             plt.plot(dfx,mixtures_c1.reshape(n_mixture_pure,n_features)[0])
#             print(mixtures_c1.shape)
    
            #Add all pure mixtures together

            targets_pure=np.concatenate((targets_c1,targets_c2,targets_c3,targets_c4,
                                          targets_c5,targets_c6,targets_c7,targets_c8,
                                          targets_c9,targets_c10,targets_c11,targets_c12),axis=0)

            mixtures_pure=np.concatenate((mixtures_c1,mixtures_c2,mixtures_c3,mixtures_c4,
                                          mixtures_c5,mixtures_c6,mixtures_c7,mixtures_c8,
                                          mixtures_c9,mixtures_c10,mixtures_c11,mixtures_c12),axis=0)

            self.targets_pure=targets_pure.astype(np.float)
            self.mixtures_pure=mixtures_pure.astype(np.float)
            
        if self.verbosity == True:

            print('targets_pure data type: ', self.targets_pure.dtype)
            print('mixtures_pure data type: ', self.mixtures_pure.dtype)


            print('targets_pure.shape :',self.targets_pure.shape)
            print('mixtures_pure.shape :',self.mixtures_pure.shape)
            
            
    def make_artificial_mixtures(self,n_mixtures):
        # Number of artificial randomized mixtures
        # This should equal to predetermined array size and loop counter

        # Have to set a random number seed here

        
        t_start = datetime.now()

        self.n_mixtures = n_mixtures

        targets = np.empty((self.n_mixtures,self.n_mixture_component_max),dtype=object)
        dilution = np.empty((self.n_mixtures),dtype=object)
        for i in range(self.n_mixtures):
        #      a[x]=np.array([x, x+1])
                c_rand=np.random.random(12)
                c_rand /=c_rand.sum()
                dilution[i]=np.random.random(1)
        #         print('Dilution : ', dilution[i])

                targets[i] = c_rand*(1-dilution[i])
                targets[i]=targets[i].astype(np.float)
        #         print('Sum of mixture components : ', targets[i].sum())



        t_end = datetime.now()
        delta = t_end - t_start
        Time_OVR=delta.total_seconds() * 1000

        if self.verbosity == True:
            
            print('Time elaspsed in generating targets: ', Time_OVR) # milliseconds
            print('Number of artificial mixtures : ', self.n_mixtures)
            print('Targets data type: ', targets.dtype)
            print('Targets shape: ', targets.shape)

        
        t_start = datetime.now()

        mixtures = np.empty((self.n_mixtures,1, self.n_features),dtype=object)

        for i in range(n_mixtures):
            mixtures[i] = np.dot(self.basis.T,targets[i])
            mixtures[i]=mixtures[i].astype(np.float)


        t_end = datetime.now()
        delta = t_end - t_start
        Time_OVR=delta.total_seconds() * 1000
        
        if self.verbosity == True:
            print('Loading time: ', Time_OVR) # milliseconds
            
            print('numpy state:',np.random.get_state()[1][0])

            print('mixtures data type: ', mixtures.dtype)
            print('mixtures shape: ', mixtures.shape)

            
        #Combine pure compounds and artificial mixtures together

        #If do not want pure spectra then do not run this cell

        self.n_mixtures = (self.n_mixtures + self.n_mixture_pure*self.n_mixture_component_max)
        

        self.targets=np.concatenate((self.targets_pure, targets),axis=0)

        self.mixtures=np.concatenate((self.mixtures_pure, mixtures),axis=0)

        
        if self.verbosity == True:
            print('Total number of linear simulated mixtures : ',self.n_mixtures)
            print('\nCombined (pure compounds and simulated mixtures)\n')
            print('targets data type: ', self.targets.dtype)
            print('mixtures_pure data type: ', self.mixtures.dtype)
            print('targets data shape: ', self.targets.shape)
            print('mixtures_pure data shape: ', self.mixtures.shape)

    def save_linear_sim_mixtures(self):
        #save the data in binary numpy file before removing indices
        #Get Current date and time to store the data


        today = date.today()
        self.now = datetime.now()
        
        

        print("now =", self.now)
        # dd/mm/YY H:M:S
        dt_string = self.now.strftime("%d-%m-%Y_time_%H-%M-%S")
        print("date and time =", dt_string)	





        data_identifier = 'RandomTrainingMixtures' + '/' + 'N_mix = ' + str(self.n_mixtures) + '_' + dt_string + '_FullSet_'


        np.save(data_identifier + 'Mixture_Net_data_mixtures', self.mixtures)
        np.save(data_identifier + 'Mixture_Net_data_targets', self.targets)
        
    def apply_abs_threshold(self, Abs_threshold = 0.001 ):
        # Check mixture threshold and eliminate
        # mixtures[0]
        # print the max absorbance values for all the mixtures below threshold and their indices
        self.Abs_threshold = Abs_threshold
        remove_indices = []
        for _ in range(self.n_mixtures):
            if np.amax(self.mixtures[_]) < self.Abs_threshold:

        #         print('Index : ',_)
                remove_indices=np.append(remove_indices, _)
                #save indices in array
        #         print(' Max Abs:',np.amax(mixtures[_]))

        # print(remove_indices)
#         print('Total indices removed : ',remove_indices)
        self.remove_indices=remove_indices.astype(int)

        #remove those mixtures and their corresponding concentrations from dataset
        self.mixtures=np.delete(self.mixtures, self.remove_indices, 0)
        self.targets=np.delete(self.targets, self.remove_indices, 0)

        print('mixtures data type: ', self.mixtures.dtype)
        print('mixtures shape: ', self.mixtures.shape)

        print('targets data type: ', self.targets.dtype)
        print('targets shape: ', self.targets.shape)
        
        self.n_mixtures = self.n_mixtures -self.remove_indices.shape[0]
        
        self.mixtures = self.mixtures.reshape(self.n_mixtures,self.n_features)
        
    def save_thresholded_mixtures(self):        
        #save the data in binary numpy file
        #Get Current date and time to store the data
        

        today = date.today()
        # now = datetime.now() # use same timestamp as before

        print("now =", self.now)
        # dd/mm/YY H:M:S
        dt_string = self.now.strftime("%d-%m-%Y_time_%H-%M-%S")
        print("date and time =", dt_string)	


        

        
        data_identifier = 'RandomTrainingMixtures' + '/' + 'N_mix = ' + str(self.n_mixtures -self.remove_indices.shape[0]) + '_' + dt_string + '_'
        

        np.save(data_identifier + 'Mixture_Net_data_mixtures', self.mixtures)
        np.save(data_identifier + 'Mixture_Net_data_targets', self.targets)
        
        
    def _target_generator(self,n_sample_mixtures,exclude_indices, verbose=False):
        '''Randomly generate concentration target vector with specific indices removed to come up with 2-,3-,4-

        etc. component mixture

        V18 added a check so very weak component contribution spectra are taken out.
        '''

        print(f'\n ...generating {12-exclude_indices}-component mixtures data...\n')
        test_targets = np.empty((n_sample_mixtures,self.n_mixture_component_max),dtype=object)
        test_dilution = np.empty((n_sample_mixtures),dtype=object)

        for i in range(n_sample_mixtures):
        #      a[x]=np.array([x, x+1])
                c_rand=np.random.random(12)
                exclude_set_indices=random.sample(range(0, 12), exclude_indices)
    #             print(exclude_set_indices)
                for _ in exclude_set_indices:
        ##             print(_)
                    c_rand[_]=0

        ##         print('before normalization c_rand = ', c_rand)
                c_rand /=c_rand.sum()
        ##         print('after normalization c_rand = ', c_rand)
                test_dilution[i]=np.random.random(1)
                test_dilution[i]=test_dilution[i].astype(np.float)
    #             print('Dilution : ', test_dilution[i])

                test_targets[i] = c_rand*(1-test_dilution[i])
    #             print('\n')
                if verbose==True:
                    print('test_targets = ', test_targets[i])
    #             print('\n')
                test_targets[i]=test_targets[i].astype(np.float)
    #             print('Sum of mixture components : ', test_targets[i].sum())

        return test_targets, test_dilution

    
    def make_controlled_test_mixtures(self, equal_amount = 10000, tweleve_component_amount=140000, abs_threshold = 0.001, test_Species_Abs_Threshold=0.0001, save_to_file = False):
        
        '''
        Creates controlled 1-, 2-, 3-, 4- , 5- 6-, 7-, 8-, 9-, 10-, 11- and 12- component test mixtures, applies absorbance threshold to remove mixtures below a certain absorbance value. The function saves 6 files. Fullset refers to mixtures created completely randomly. Absorbance thresholded mixtures do not have any other comments in the filename. "weak_rm" refers to the removal of spectra containing very weak contribution from at least one component species.
        
        
        equal amount: int, the amount of 1-, 2-, 3-, 4- , 5- 6-, 7-, 8-, 9-, 10-, 11- component mixtures.
        tweleve_component_amount: int, the amount of 12- component mixtures.
        abs_threshold: float, an absorbance threshold on the spectra. Any mixture with maximum absorbance below this value is excluded.
        test_Species_Abs_Threshold: float, an absorbance threshold on the absorbance contribution from a species present in the spectra. Any mixture containing at least one such species is excluded.
        save_to_file, boolean, saves mixture data into .npy files.
        
        
        '''
        
        t_start = datetime.now()


        n_test_pure_mixtures = equal_amount
        n_test_two_component_mixtures = equal_amount
        n_test_three_component_mixtures = equal_amount
        n_test_four_component_mixtures = equal_amount
        n_test_five_component_mixtures = equal_amount
        n_test_six_component_mixtures = equal_amount
        n_test_seven_component_mixtures = equal_amount
        n_test_eight_component_mixtures = equal_amount
        n_test_nine_component_mixtures = equal_amount
        n_test_ten_component_mixtures = equal_amount
        n_test_eleven_component_mixtures = equal_amount
        n_test_twelve_component_mixtures = tweleve_component_amount

        test_targets_pure,test_targets_pure_dilution = self._target_generator(n_test_pure_mixtures,11)
        test_targets_two_components,test_targets_two_components_dilution = self._target_generator(n_test_two_component_mixtures,10)
        test_targets_three_components,test_targets_three_components_dilution = self._target_generator(n_test_three_component_mixtures,9)
        test_targets_four_components,test_targets_four_components_dilution = self._target_generator(n_test_four_component_mixtures,8)
        test_targets_five_components,test_targets_five_components_dilution = self._target_generator(n_test_five_component_mixtures,7)
        test_targets_six_components,test_targets_six_components_dilution = self._target_generator(n_test_six_component_mixtures,6)
        test_targets_seven_components,test_targets_seven_components_dilution = self._target_generator(n_test_seven_component_mixtures,5)
        test_targets_eight_components,test_targets_eight_components_dilution = self._target_generator(n_test_eight_component_mixtures,4)
        test_targets_nine_components,test_targets_nine_components_dilution = self._target_generator(n_test_nine_component_mixtures,3)
        test_targets_ten_components,test_targets_ten_components_dilution = self._target_generator(n_test_ten_component_mixtures,2)
        test_targets_eleven_components,test_targets_eleven_components_dilution = self._target_generator(n_test_eleven_component_mixtures,1)
        test_targets_twelve_components,test_targets_twelve_components_dilution = self._target_generator(n_test_twelve_component_mixtures,0)


        t_end = datetime.now()
        delta = t_end - t_start
        Time_OVR=delta.total_seconds() * 1000
        print('Time elasped:',Time_OVR)


        #Combine all the test mixtures together

        #If do not want pure spectra then do not run this cell

        n_test_mixtures = (n_test_pure_mixtures + 
                           n_test_two_component_mixtures +
                           n_test_three_component_mixtures +
                           n_test_four_component_mixtures +
                           n_test_five_component_mixtures +
                           n_test_six_component_mixtures +
                           n_test_seven_component_mixtures +
                           n_test_eight_component_mixtures +
                           n_test_nine_component_mixtures +
                           n_test_ten_component_mixtures +
                           n_test_eleven_component_mixtures +
                           n_test_twelve_component_mixtures                  
                          )
        print('Total number of test mixtures : ',n_test_mixtures)

        test_targets=np.concatenate((test_targets_pure, 
                                     test_targets_two_components,
                                     test_targets_three_components,
                                     test_targets_four_components,
                                     test_targets_five_components,
                                     test_targets_six_components,
                                     test_targets_seven_components,
                                     test_targets_eight_components,
                                     test_targets_nine_components,
                                     test_targets_ten_components,
                                     test_targets_eleven_components,
                                     test_targets_twelve_components 
                                    ),axis=0)

        test_dilution=np.concatenate((test_targets_pure_dilution, 
                                     test_targets_two_components_dilution,
                                     test_targets_three_components_dilution,
                                     test_targets_four_components_dilution,
                                     test_targets_five_components_dilution,
                                     test_targets_six_components_dilution,
                                     test_targets_seven_components_dilution,
                                     test_targets_eight_components_dilution,
                                     test_targets_nine_components_dilution,
                                     test_targets_ten_components_dilution,
                                     test_targets_eleven_components_dilution,
                                     test_targets_twelve_components_dilution
                                     ),axis=0)




        print('\nCombined test simulated mixtures\n')

        print('No. of test mixtures: ', n_test_mixtures)

        print('test_targets data type: ', test_targets.dtype)
        print('test_targets data shape: ', test_targets.shape)



        t_start = datetime.now()

        test_mixtures = np.empty((n_test_mixtures,1, self.n_features),dtype=object)


        for j in range(n_test_mixtures):
            test_mixtures[j] = np.dot(self.basis.T.reshape(self.n_features,self.n_mixture_component_max),test_targets[j])
            test_mixtures[j]=test_mixtures[j].astype(np.float)
        #     print(test_mixtures[j])


        t_end = datetime.now()
        delta = t_end - t_start
        Time_OVR=delta.total_seconds() * 1000

        print('Time elaspsed: ', Time_OVR) # milliseconds

        # Use predetermined mixtures to understand the model better
        #Print the random number seed
        print('numpy random state: ', np.random.get_state()[1][0])

                
        self.n_test_mixtures = n_test_mixtures
        self.test_targets = test_targets
        self.test_dilution = test_dilution
        self.test_mixtures = test_mixtures
        
        
        
        
        
        if save_to_file == True:
        
            print('test mixtures data type: ', self.test_mixtures.dtype)
            print('test mixtures shape: ', self.test_mixtures.shape)

            today = date.today()
            now = datetime.now()

            print("now =", now)
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d-%m-%Y_time_%H-%M-%S")
            print("date and time =", dt_string)	

            data_identifier = 'RandomTrainingMixtures' + '/' + 'N_mix = ' + str(self.n_test_mixtures) + '_' + dt_string + '_FullSet_'

            np.save(data_identifier + 'Mixture_Net_data_test_mixtures', self.test_mixtures)
            np.save(data_identifier + 'Mixture_Net_data_test_targets', self.test_targets)
            
        # start removing indices based on absorbance threshold on mixture spectra
        
        remove_indices = []
        for _ in range(self.n_test_mixtures):
            if np.amax(self.test_mixtures[_]) < abs_threshold:

        #         print('Index : ',_)
                remove_indices=np.append(remove_indices, _)
                #save indices in array
        #         print(' Max Abs:',np.amax(mixtures[_]))

        # print(Remove_indices)
#         print('Total indices removed : ',remove_indices)
        remove_indices=remove_indices.astype(int)
        
        
        #remove those mixtures and their corresponding concentrations from dataset
        self.test_mixtures=np.delete(self.test_mixtures, remove_indices, 0)
        self.test_targets=np.delete(self.test_targets, remove_indices, 0)

        print('test_mixtures data type: ', self.test_mixtures.dtype)
        print('test_mixtures shape: ', self.test_mixtures.shape)

        print('test_targets data type: ', self.test_targets.dtype)
        print('test_targets shape: ', self.test_targets.shape)
        
        
        #Adjust the number of test mixtures
        
        self.n_test_mixtures = self.n_test_mixtures - remove_indices.shape[0]
        print('Adjusted n_test_mixtures: ', self.n_test_mixtures)
        
        
        if save_to_file == True:
        
            today = date.today()
            now = datetime.now()

            print("now =", now)
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d-%m-%Y_time_%H-%M-%S")
            print("date and time =", dt_string)	

            data_identifier = 'RandomTrainingMixtures' + '/' + 'N_mix = ' + str(self.n_test_mixtures) + '_' + dt_string + '_'

            np.save(data_identifier + 'Mixture_Net_data_test_mixtures', self.test_mixtures)
            np.save(data_identifier + 'Mixture_Net_data_test_targets', self.test_targets)
        
        
        #integrate the stricter species wise detection threshold
        
        

        remove_indices_weak_test_spectra = []
        # for i in range(y_test.shape[0]):
        for i in range(self.n_test_mixtures):
        #     print(i)
            for _ in range(self.n_mixture_component_max):
                if np.amax(np.dot(self.basis[_],self.test_targets[i][_])) < test_Species_Abs_Threshold and np.amax(np.dot(self.basis[_],self.test_targets[i][_])) > 0:
                    remove_indices_weak_test_spectra=np.append(remove_indices_weak_test_spectra, i)
        #             print( np.amax(np.dot(Basis[_],y_test[i][_])) )
        #             print(i)
        # Remove this indice spectra from y_test_BK and X_test_BK         

        #         print( np.dot(Basis[_],y_test[i][_]).shape )

        remove_indices_weak_test_spectra=np.unique(remove_indices_weak_test_spectra) #Remove duplicates of same indices
        remove_indices_weak_test_spectra=remove_indices_weak_test_spectra.astype(int)
        print('Total spectra with weak components in testing dataset:',np.unique(remove_indices_weak_test_spectra).shape)    

        
        self.test_mixtures=np.delete(self.test_mixtures, remove_indices_weak_test_spectra, 0)
        self.test_targets=np.delete(self.test_targets, remove_indices_weak_test_spectra, 0)

        print('test_mixtures data type: ', self.test_mixtures.dtype)
        print('test_mixtures shape: ', self.test_targets.shape)

        print('test_targets data type: ', self.test_mixtures.dtype)
        print('test_targets shape: ', self.test_targets.shape)
        #Adjust the number of test mixtures
        self.n_test_mixtures = self.n_test_mixtures - remove_indices_weak_test_spectra.shape[0]
        print('Adjusted n_test_mixtures after removing spectra with weak components: ', self.n_test_mixtures)
        
        
        if save_to_file == True:
        
            today = date.today()
            now = datetime.now()

            print("now =", now)
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d-%m-%Y_time_%H-%M-%S")
            print("date and time =", dt_string)	

            data_identifier = 'RandomTrainingMixtures' + '/' + 'N_mix = ' + str(self.n_test_mixtures) + '_' + dt_string + '_'

            np.save(data_identifier + 'Mixture_Net_data_test_mixtures_weak_rm', self.test_mixtures)
            np.save(data_identifier + 'Mixture_Net_data_test_targets_weak_rm', self.test_targets)
        
        
        
        self.test_mixtures = self.test_mixtures.reshape(self.n_test_mixtures,self.n_features)
      

        

