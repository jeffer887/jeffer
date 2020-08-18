#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
from gaussPDF import gaussPDF
def GMR(Priors, Mu, Sigma, x, input, output):  #此处x=expData[0, :]，即第一行的长度

    #Priors维度为(3, 1)
    #Mu维度为(数据的行数+1)×nbStates (8, 3)
    #Sigma维度为(8, 8, 3) 
    lo = np.size(output)  #lo=7

    nbData = np.size(x) #nbData=N
    nbVar = np.size(Mu, 0) #nbVar=8

    nbStates = np.size(Sigma, 2) #nbStates=3
    #numpy.size(a, axis=None) 指定返回哪一维的元素个数，当没有指定时，返回整个矩阵的元素个数
    realmin = sys.float_info[3]
    Pxi = np.ndarray(shape=(nbData, nbStates))  #Pxi  nbData*nbStates
    x = np.reshape(x,(1,nbData))  #x  1*nbData
    y_tmp = np.ndarray(shape = (nbVar-1, nbData, nbStates))  
    #y_tmp  (nbVar-1)*nbData*nbStates (7, nbData, 3)
    Sigma_y_tmp = np.ndarray(shape = (lo, lo, 1, nbStates))
    #Sigma_y_tmp = (nbVar-1)*(nbVar-1)*1*nbStates (7, 7, 1, 3)

    for i in range (0,nbStates):
        m = Mu[input,i] #m=均值Mu的第一行数据
        m = np.reshape(m,(np.size(input),1)) #np.size(input)=1

        #将m变成1*1的矩阵
        s = Sigma[input, input, i] #s = 方差Sigma第一维的数据
        s = np.reshape(s, (np.size(input),np.size(input)))
        #将s变成1*1的矩阵
        Pxi[:,i] = np.multiply(Priors[i],gaussPDF(x,m,s))
    beta = np.divide(Pxi,np.tile(np.reshape(np.sum(Pxi,1),(nbData, 1))+realmin,(1,nbStates)))
    #beta为高斯分量的后验概率,维度nbData*nbStates
    for j in range (0,nbStates):
        #删除均值Mu第一行
        #求mu(s,k)
        a = np.delete(Mu, np.s_[input], axis = 0)
        #np.s_[0]就是切片的意思,np.s_[2::2]=slice(2, None, 2)
        #此处沿0轴(竖轴)删除Mu第0行数据
        a = a[:,j]
        a = np.reshape(a,(nbVar-np.size(input),1)) #维度(7, 1)
        a = np.tile(a, (1, nbData)) #平铺a，使其维度变为(7, nbData)
        
        #求sigma(st,k)
        b = np.delete(Sigma[:,:,j], 0, axis = 0)
        #删除第一个高斯分量方差第一行,b维度(7, 8)
        b = np.delete(b, np.s_[1:nbVar], axis = 1)
        #print(np.shape(b))
        #删除第一个高斯分量方差第一列(1)到最后一列(nbVar),b维度(7, 1)
        
        c = Sigma[input, input, j]
        #c为第一个高斯分量方差第一个点Sigma[0][0]=sigma(tt,k)
        c = np.reshape(c, (1,1))
        c = np.linalg.inv(c)#求逆
        c = np.dot(b, c) #c维度(7, 1)
        #c=sigma(st,k)*sigma(tt,k)^(-1)
        
        d = np.reshape(Mu[input, j], (1,1))
        #d为第一个高斯分量均值第一个点Mu[0][0]=mu(t,k)
        d = np.tile(d, (1,nbData))
        #平铺d,使其维度变为(1,nbData)
        d = x - d
        #x=range(1, nbData)
        #d=x(t)-mu(t,k)
        d = np.dot(c, d)  #d维度(7, nbData)
        #d=sigma(st,k)*sigma(tt,k)^(-1)*[x(t)-mu(t,k)]
        #d维度(7, nbData)

        y_tmp[:,:,j] = a + d
        #求得x(s,k)=mu(s,k)+sigma(st,k)*sigma(tt,k)^(-1)*[x(t)-mu(t,k)]
        
    # pravilno
    a, b = np.shape(beta)
    #print(a,b) a=86,b=3
    beta_tmp = np.reshape(beta, (1,a,b))
    #print(beta_tmp)
    #beta_tmp (1, 86, 3)
    a = np.tile(beta_tmp,(lo,1,1))
    #print(np.shape(a))
    #a (7, 86, 3))
    y_tmp2 = a*y_tmp
    #求得beta*x(s,k)
    #print(y_tmp2)
    #print(np.shape(y_tmp2))
    #y_tmp2 (7, 86, 3)
    
    
    # print('1')
    # print(y_tmp2[:,:,0])
    # print('2')
    # print(y_tmp2[:, :, 1])
    # print('3')
    # print(y_tmp2[:, :, 2])
    # print('4')
    # print(y_tmp2[:, :, 3])
    
    y = np.sum(y_tmp2,2)
    #求得x(s)
    
    #print(y)
    #print(np.shape(y))
    #y (7, 86)

    for j in range(0, nbStates):
        
        a = np.delete(Sigma[:,:,j], 0, axis = 0)
        a = np.delete(a, 0, axis = 1)
        #a=sigma(ss,k) 维度(7, 7)
        
        b = np.delete(Sigma[:, :, j], 0, axis=0)
        b = np.delete(b, np.s_[1:nbVar], axis=1)
        #b=sigma(st,k) 维度(7, 1)
        
        c = Sigma[input, input, j]
        c = np.reshape(c, (1, 1))
        c = np.linalg.inv(c)
        #c=[sigma(tt,k)]^(-1)
        c = np.dot(b, c)
        #c=sigma(st,k)*[sigma(tt,k)]^(-1) 维度(7, 1)
        
        d = np.delete(Sigma[:,:,j], 0,axis = 1)
        d = np.delete(d, np.s_[1:nbVar], axis = 0)
        #d=sigma(ts,k) 维度(1, 7)
        d = np.dot(c, d)
        #d=sigma(st,k)*[sigma(tt,k)]^(-1)*sigma(ts,k) 维度(7, 7)
        
        Sigma_y_tmp[:,:,0,j] = a - d 
        #求得sigma(s,k)=sigma(ss,k)-sigma(st,k)*[sigma(tt,k)]^(-1)*sigma(ts,k)
        
    #print(np.shape(Sigma_y_tmp))  sigma_y_tmp (7, 7, 1, 3)
    #print(Sigma_y_tmp)
    a, b = np.shape(beta)
    beta_tmp = np.reshape(beta,(1,1,a,b))
    a = beta_tmp*beta_tmp
    a = np.tile(a, (lo,lo,1,1))
    b = np.tile(Sigma_y_tmp,(1,1,nbData,1))
    Sigma_y_tmp2 = a*b
    #求得beta^2 * sigma(s,k)
    
    Sigma_y = np.sum(Sigma_y_tmp2, 3)
    #求得sigma(s)
    
    return (y, Sigma_y)
