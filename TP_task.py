#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 20:38:32 2019

@author: zxc
"""

from TP_EM import TP_EM

import numpy as np

from TP_GMM1 import TP_GMM_frame1
from TP_GMM2 import TP_GMM_frame2

def TP_task(Mu11, Mu12, Mu13, Mu21, Mu22, Mu23,
            Sigma11, Sigma12, Sigma13, Sigma21, Sigma22, Sigma23, nbStates):
    #Mu1 (8, 3)
    #Sigma1 (8, 8, 3)
    #print(Mu2)
    #print(np.shape(Mu1), np.shape(Sigma1))
    nbVar = 8
    Mu = np.ndarray(shape = (nbVar, nbStates))
    Sigma = np.ndarray(shape = (nbVar, nbVar, nbStates))
    
    _, _, _, b1 = TP_GMM_frame1()
    _, _, _, b2 = TP_GMM_frame2()
    
    #对Mu1和Mu2进行处理
    for m in range(0, nbStates):        
        Mu11[1:8, m] = Mu11[1:8, m] + np.transpose(b1[m])
        Mu12[1:8, m] = Mu12[1:8, m] + np.transpose(b1[m])
        Mu13[1:8, m] = Mu13[1:8, m] + np.transpose(b1[m])
        
    for m in range(0, nbStates):      
        Mu21[1:8, m] = Mu21[1:8, m] + np.transpose(b2[m])
        Mu22[1:8, m] = Mu22[1:8, m] + np.transpose(b2[m])
        Mu23[1:8, m] = Mu23[1:8, m] + np.transpose(b2[m])
   
    for i in range(0, nbStates):
        
        Sigma11_ni = np.linalg.inv(Sigma11[:,:,i])
        Sigma12_ni = np.linalg.inv(Sigma12[:,:,i])
        Sigma13_ni = np.linalg.inv(Sigma11[:,:,i])
        #print(np.shape(Sigma1_ni))
        #print(Sigma1[:,:,i])
        Sigma21_ni = np.linalg.inv(Sigma21[:,:,i])
        Sigma22_ni = np.linalg.inv(Sigma22[:,:,i])
        Sigma23_ni = np.linalg.inv(Sigma23[:,:,i])
        
        Sigma12_sum = np.add(Sigma11_ni, Sigma12_ni)
        Sigma12_sum = np.add(Sigma12_sum, Sigma13_ni)
        Sigma12_sum = np.add(Sigma12_sum, Sigma21_ni)
        Sigma12_sum = np.add(Sigma12_sum, Sigma22_ni)
        Sigma12_sum = np.add(Sigma12_sum, Sigma23_ni)
        
        Sigma_gu = np.linalg.inv(Sigma12_sum)
        Sigma[:,:,i] = Sigma_gu
        
        #Mu1[:,i] = np.array(Mu1[:,i])
        #print(Mu1[:,i])
        #Mu1_re = np.reshape(Mu1[:,i], (7, 1))
        Mu11_ni = np.dot(Sigma11_ni, Mu11[:,i])
        Mu12_ni = np.dot(Sigma12_ni, Mu12[:,i])
        Mu13_ni = np.dot(Sigma13_ni, Mu13[:,i])
        #Mu2_re = np.reshape(Mu2[:,i], (7, 1))
        Mu21_ni = np.dot(Sigma21_ni, Mu21[:,i])
        Mu22_ni = np.dot(Sigma22_ni, Mu22[:,i])
        Mu23_ni = np.dot(Sigma23_ni, Mu23[:,i])
        
        Mu12_sum = np.add(Mu11_ni, Mu12_ni)
        Mu12_sum = np.add(Mu12_sum, Mu13_ni)
        Mu12_sum = np.add(Mu12_sum, Mu21_ni)
        Mu12_sum = np.add(Mu12_sum, Mu22_ni)
        Mu12_sum = np.add(Mu12_sum, Mu23_ni)
        
        Mu_gu = np.dot(Sigma_gu, Mu12_sum)
        #print(Mu_gu)
        Mu[:,i] = Mu_gu
        
    Mu = np.array(Mu)
    Sigma = np.array(Sigma)
    
    return (Mu, Sigma)