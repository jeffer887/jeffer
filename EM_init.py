#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

#已改进，前文迭代次数不达标
def kMeans(X, K):
    centroids = X[np.random.choice(np.arange(len(X)), K), :]  #选取的中心随机，不利于后续计算，现将均值按表头顺序有小到大排序
    
    centroids = np.array(np.transpose(centroids)) #将centroids转置

    #冒泡法调整centroids各元素位置
    for i in range(0, len(np.transpose(centroids))):    # 这个循环负责设置冒泡排序进行(大步骤)的次数
        for index in range(0, len(np.transpose(centroids)) - i - 1):  # 负责每个小步骤的进行，并提供索引
            if centroids[0, index] > centroids[0, index + 1]:
                centroids[:,(index, index + 1)] = centroids[:,(index + 1, index)] 
                #将前一列与后一列交换位置，注意是在数组中，在列表中写法不同
    centroids = np.array(np.transpose(centroids))  #转置回来

    loglik_threshold = 1e-10
    step = 0
    while 1:
        ex_center = centroids
        C = np.array([np.argmin([np.dot(x_i-y_k, x_i-y_k) for y_k in centroids]) for x_i in X])        
        centroids = [X[C == k].mean(axis = 0) for k in range(K)]

        delta = np.array(centroids) - np.array(ex_center)

        Q = np.linalg.norm(delta, axis=1)

        if np.linalg.norm(Q) < loglik_threshold:
            break
        step = step + 1

    return np.array(centroids) , C


def EM_init(Data, nbStates):
    nbVar, nbData = np.shape(Data) 

    Priors = np.ndarray(shape = (1, nbStates)) 
    
    Sigma = np.ndarray(shape = (nbVar, nbVar, nbStates)) #Data的行数×Data的行数×nbStates
    #Sigma = np.ndarray(shape = (nbStates, nbVar, nbVar)) 
    
    Centers, Data_id = kMeans(np.transpose(Data), nbStates) 
    #Centers是簇的中心，维度为nbStates×(数据的行数+1),Data_id是每个点所属的类型，总共有nbStates种

    Mu = np.transpose(Centers)

    #Mu是簇中心Centers的转置，维度为(数据的行数+1)×nbStates

    for i in range (0, nbStates):
        idtmp = np.nonzero(Data_id==i)
        #np.nonzero()函数是numpy中用于得到数组array中非零元素的位置（数组索引）的函数，若b=np.nonzero(a)中a为一维数组，则输出b为一维，从横轴axis=0一个维度来描述。若a为二维,则输出b为二维，描述a中非零元素所在的位置，从横轴axis=0和纵轴axis=1两个维度来描述。三维同理。

        idtmp = list(idtmp)
        idtmp = np.reshape(idtmp,(np.size(idtmp)))

        #size()函数主要是用来统计矩阵元素个数(默认)，或矩阵某一维(0为行，1为列)上的元素个数的函数

        Priors[0,i] = np.size(idtmp) #Priors得到idtmp中每一种数据的个数

        a = np.concatenate((Data[:, idtmp],Data[:, idtmp]), axis = 1)
        #np.concatenate()数组拼接函数，默认是 axis = 0，
        
        Sigma[:,:,i] = np.cov(a)
        #np.cov()协方差函数,计算各个维度之间的协方差
        #rowvar:默认为True,此时每一行代表一个变量（属性，维度！！），每一列代表一个观测；为False时，则反之
        #此处应为rowvar = False
        
        Sigma[:,:,i] = Sigma[:,:,i] + 0.00001 * np.diag(np.diag(np.ones((nbVar,nbVar))))
    
    Priors = Priors / nbData

    return (Priors, Mu, Sigma)
