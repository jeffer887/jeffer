#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:19:19 2019

@author: zxc
"""


from gaussPDF import gaussPDF
import numpy as np
import sys
def TP_EM(Priors, Data1, Data2, Mu1, Mu2, Sigma1, Sigma2): #参数待定!!
    
    realmax = sys.float_info[0]  # =max
    realmin = sys.float_info[3]  # =min
    #确定上下限
    #loglik_threshold = 1e-10
    #1e-10太小，扩大100倍
    loglik_threshold = 1e-10
    #确定迭代精度
    
    nbVar, nbData = np.shape(Data1) #nbVar行数(=8),nbData列数
    nbStates = np.size(Priors)
    
    loglik_old = -realmax
    nbStep = 0

    Data1 = Data1
    Data2 = Data2    
    Mu1 = Mu1
    Mu2 = Mu2
    Sigma1 = Sigma1
    Sigma2 = Sigma2
    
    Priors = Priors  #待定!!与初始化的一致
    
    Pix = np.ndarray(shape = (nbStates, nbData))
    Pxi = np.ndarray(shape = (nbData, nbStates))
    Pxi1 = np.ndarray(shape = (nbData, nbStates))
    Pxi2 = np.ndarray(shape = (nbData, nbStates))
    
    while 1:
        
        #E-step
        for i in range (0, nbStates):
            Pxi1[:,i] = gaussPDF(Data1, Mu1[:,i], Sigma1[:,:,i])  
            #Pxi  nbData*nbStates
            Pxi2[:,i] = gaussPDF(Data2, Mu2[:,i], Sigma2[:,:,i])
            Pxi[:,i] = np.multiply(Pxi1[:,i], Pxi2[:,i])
            
        #防止Pix_tmp出现奇异矩阵
        #Pix_tmp = Pix_tmp + 0.00001 * np.ones((nbData, nbStates))
        
        #防止Pix出现奇异矩阵
        Pxi = Pxi + 0.00001 * np.ones((nbData, nbStates))
               
        Pix_tmp = np.multiply(np.tile(Priors, (nbData, 1)),Pxi)       

        Pix = np.divide(Pix_tmp, np.tile(np.reshape(np.sum(Pix_tmp,1), (nbData, 1)), (1, nbStates)))
        #Pix=p(k|xi)
        
        E = np.sum(Pix, 0)  #ek
        Priors = np.reshape(Priors, (nbStates))
        
        #M-step
        #计算坐标系1下的均值与方差
        for i in range (0, nbStates):
            Priors[i] = E[i] / nbData   #pi(k)=ek/N
            Mu1[:,i] = np.dot(Data1, Pix[:,i]) / E[i]  #计算uk_1
            
            Data_tmp1 = Data1 - np.tile(np.reshape(Mu1[:,i], (nbVar, 1)), (1, nbData))
            #Data_tmp1  xi-uk  nbVar*nbData
            a = np.transpose(Pix[:, i])  #a  1*nbData
            b = np.reshape(a, (1, nbData))
            c = np.tile(b, (nbVar, 1))  #nbVar*nbData
            d = c * Data_tmp1  #此处*号=np.multiply()
            e = np.transpose(Data_tmp1)  #xi-uk转置
            f = np.dot(d,e)
            #计算p(k|xi)*(xi-uk)*(xi-uk)'
            Sigma1[:,:,i] = f / E[i] # Sigma(k)
            Sigma1[:,:,i] = Sigma1[:,:,i] + 0.00001 * np.diag(np.diag(np.ones((nbVar, nbVar))))
            #Sigma1[:,:,i] = Sigma1[:,:,i] + 0.1 * np.eye(nbVar)
        
        #计算坐标系2下的均值与方差
        for i in range (0, nbStates):
            Priors[i] = E[i] / nbData   #pi(k)=ek/N
            Mu2[:,i] = np.dot(Data2, Pix[:,i]) / E[i]  #计算uk_2
            
            Data_tmp1 = Data2 - np.tile(np.reshape(Mu2[:,i], (nbVar, 1)), (1, nbData))
            #Data_tmp1  xi-uk  nbVar*nbData
            a = np.transpose(Pix[:, i])  #a  1*nbData
            b = np.reshape(a, (1, nbData))
            c = np.tile(b, (nbVar, 1))  #nbVar*nbData
            d = c * Data_tmp1  #此处*号=np.multiply()
            e = np.transpose(Data_tmp1)  #xi-uk转置
            f = np.dot(d,e)
            #计算p(k|xi)*(xi-uk)*(xi-uk)'
            Sigma2[:,:,i] = f / E[i] # Sigma(k)
            Sigma2[:,:,i] = Sigma2[:,:,i] + 0.00001 * np.diag(np.diag(np.ones((nbVar, nbVar))))
            #Sigma2[:,:,i] = Sigma2[:,:,i] + 0.1 * np.eye(nbVar)
        
	#print(Mu1[0])
        #迭代
        for i in range (0, nbStates):
            Pxi1[:,i] = gaussPDF(Data1, Mu1[:,i],Sigma1[:,:,i])  
            #Pxi  nbData*nbStates
            Pxi2[:,i] = gaussPDF(Data2, Mu2[:,i],Sigma2[:,:,i])
            Pxi[:,i] = np.multiply(Pxi1[:,i], Pxi2[:,i])
            #将新计算得到的均值u和方差Sigma带入进行迭代
        
        F = np.dot(Pxi, np.transpose(Priors))
        indexes = np.nonzero(F<realmin)
        indexes = list(indexes)
        indexes = np.reshape(indexes, np.size(indexes))
        F[indexes] = realmin
        F = np.reshape(F, (nbData, 1))
        loglik = np.mean(np.log10(F), 0)
        if np.absolute((loglik/loglik_old)-1)<loglik_threshold:
            break
        loglik_old = loglik
        nbStep = nbStep+1
     
    #return(Priors, Mu1, Mu2, Sigma1, Sigma2, Pix, nbStates)
    return(Priors, Mu1, Mu2, Sigma1, Sigma2)

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
            
