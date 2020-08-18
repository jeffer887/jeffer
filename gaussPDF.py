#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math
import sys
def gaussPDF(Data, Mu, Sigma): #高斯分布概率密度函数

    realmin = sys.float_info[3]
    nbVar, nbData = np.shape(Data) #nbVar行数,nbData列数
    Data = np.transpose(Data) - np.tile(np.transpose(Mu), (nbData, 1))
    #np.tile将原矩阵横向、纵向地复制
    prob = np.sum(np.dot(Data, np.linalg.inv(Sigma))*Data, 1) 
    #(x-u)^T*sigma^(-1)*(x-u)
    # np.linalg.inv(Sigma)求Sigma的逆矩阵
    prob = np.exp(-0.5*prob)/np.sqrt((np.power((2*math.pi), nbVar))*np.absolute(np.linalg.det(Sigma))+realmin)
    #np.absolute计算绝对值
    #np.power(x1, x2)求x1的x2次方，x2可以是数字，也可以是数组，但是x1和x2的列数要相同
    return prob
