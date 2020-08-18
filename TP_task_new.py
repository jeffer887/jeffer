#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 20:52:58 2019

@author: zxc
"""

from TP_EM import TP_EM

import numpy as np
from TP_GMM1 import TP_GMM_frame1
from TP_GMM2 import TP_GMM_frame2

def TP_task_new(Mu1, Mu2, Sigma1, Sigma2, nbStates):
    
    nbVar = 8
    Mu = np.ndarray(shape = (nbVar, nbStates))   #Mu (8*3)
    Sigma = np.ndarray(shape = (nbVar, nbVar, nbStates))  #Sigma (8*8*3)
     
    _, _, _, b1 = TP_GMM_frame1()
    _, _, _, b2 = TP_GMM_frame2()
    
    #对Mu1, Mu2进行处理
    for m in range(0, nbStates):        
        Mu1[1:8, m] = Mu1[1:8, m] + np.transpose(b1[m])
        
    for m in range(0, nbStates):        
        Mu2[1:8, m] = Mu2[1:8, m] + np.transpose(b2[m])
   
    for i in range(0, nbStates):
        
        Sigma1_ni = np.linalg.inv(Sigma1[:,:,i])

        Sigma2_ni = np.linalg.inv(Sigma2[:,:,i])
        Sigma12_sum = np.add(Sigma1_ni, Sigma2_ni)
        Sigma_gu = np.linalg.inv(Sigma12_sum)
        Sigma[:,:,i] = Sigma_gu
        
        #Mu1[:,i] = np.array(Mu1[:,i])

        #Mu1_re = np.reshape(Mu1[:,i], (7, 1))
        Mu1_ni = np.dot(Sigma1_ni, Mu1[:,i])
        #Mu2_re = np.reshape(Mu2[:,i], (7, 1))
        Mu2_ni = np.dot(Sigma2_ni, Mu2[:,i])
        Mu12_sum = np.add(Mu1_ni, Mu2_ni)
        Mu_gu = np.dot(Sigma_gu, Mu12_sum)

        Mu[:,i] = Mu_gu
        
    Mu = np.array(Mu)
    Sigma = np.array(Sigma)


    return (Mu, Sigma)

if __name__ == "__main__":
    TP_task_new()
