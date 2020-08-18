#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gaussPDF import *
import numpy as np
import sys
def EM(Data, Priors0, Mu0, Sigma0):
    realmax = sys.float_info[0]  # =max
    #sys.float_info(max=1.7976931348623157e+308, max_exp=1024, max_10_exp=308, 
    #min=2.2250738585072014e-308, min_exp=-1021, min_10_exp=-307, dig=15, 
    #mant_dig=53, epsilon=2.220446049250313e-16, radix=2, rounds=1)
    realmin = sys.float_info[3]  # =min
    loglik_threshold = 1e-10
    nbVar, nbData = np.shape(Data) #nbVar行数(=8),nbData列数

    nbStates = np.size(Priors0)
    #size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数。
    loglik_old = -realmax
    nbStep = 0
    Mu = Mu0
    Sigma = Sigma0
    Priors = Priors0 #1*nbData

    Pix = np.ndarray(shape = (nbStates, nbData)) 
    Pxi = np.ndarray(shape = (nbData, nbStates))
    while 1:
        for i in range (0,nbStates):
            Pxi[:,i] = gaussPDF(Data,Mu[:,i],Sigma[:,:,i])  #Pxi  nbData*nbStates

        Pix_tmp = np.multiply(np.tile(Priors, (nbData, 1)),Pxi)
        Pix = np.divide(Pix_tmp,np.tile(np.reshape(np.sum(Pix_tmp,1), (nbData, 1)), (1, nbStates)))
        #Pix  nbData*nbStates
        #Pix=p(k|xi)
        E = np.sum(Pix, 0)  #ek
        Priors = np.reshape(Priors, (nbStates))
        for i in range (0,nbStates):
            Priors[i] = E[i]/nbData   #pi(k)=ek/N
            Mu[:,i] = np.dot(Data,Pix[:,i])/E[i]  
            #计算uk  (8, 1)
            Data_tmp1 = Data - np.tile(np.reshape(Mu[:,i], (nbVar, 1)), (1,nbData))
            #Data_tmp1  xi-uk  nbVar*nbData
            a = np.transpose(Pix[:, i])  #a  1*nbData
            b = np.reshape(a, (1, nbData))
            c = np.tile(b, (nbVar, 1))  #nbVar*nbData
            d = c*Data_tmp1 
            e = np.transpose(Data_tmp1)  #xi-uk转置
            f = np.dot(d,e)
            #计算p(k|xi)*(xi-uk)*(xi-uk)'
            Sigma[:,:,i] = f/E[i] # Sigma(k)
            Sigma[:,:,i] = Sigma[:,:,i] + 0.00001 * np.diag(np.diag(np.ones((nbVar,nbVar))))

        for i in range (0,nbStates):
            Pxi[:,i] = gaussPDF(Data,Mu[:,i],Sigma[:,:,i])
            #将新计算得到的均值u和方差Sigma带入进行迭代
        F = np.dot(Pxi,np.transpose(Priors))
        indexes = np.nonzero(F<realmin)
        #np.nonzero(a)返回数组a中非零元素的索引值数组。indexes为2维tuple数组
        indexes = list(indexes)#转换为列表
        indexes = np.reshape(indexes,np.size(indexes))
        F[indexes] = realmin #将小于realmin的值改成realmin
        F = np.reshape(F, (nbData, 1))
        loglik = np.mean(np.log10(F), 0)
        if np.absolute((loglik/loglik_old)-1)<loglik_threshold:
            break
        loglik_old = loglik
        nbStep = nbStep+1
    return(Priors, Mu, Sigma, Pix)
