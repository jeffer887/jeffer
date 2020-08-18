#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:59:27 2019

@author: zxc
"""
#from GMM_GMR import GMM_GMR
#from matplotlib import pyplot as plt 
import numpy as np

from get_time_axis import Get_time_axis
from get_TP_data import Get_TP_data

#得到第一个坐标系（初始坐标系）的数据分布
def TP_GMM_frame1():   
    
    datas, datas_number, datas_D, datas_N = Get_TP_data()
    
    inde_number = datas_number / 3
    
    b1 = [datas[0][0][0], datas[0][1][0], datas[0][2][0], datas[0][3][0], 
          datas[0][4][0], datas[0][5][0], datas[0][6][0]]
    b2 = [datas[10][0][0], datas[10][1][0], datas[10][2][0], datas[10][3][0], 
          datas[10][4][0], datas[10][5][0], datas[10][6][0]]
    b3 = [datas[20][0][0], datas[20][1][0], datas[20][2][0], datas[20][3][0], 
          datas[20][4][0], datas[20][5][0], datas[20][6][0]]
    
    b = [b1, b2, b3]
    
    for i in range(0, inde_number): 
        for j in range(0, datas_D):
            #head_data = datas[i][j][0]
            for k in range(0, datas_N):
                datas[i][j][k] = datas[i][j][k] - b1[j]
                
    for i in range(inde_number, 2 * inde_number): 
        for j in range(0, datas_D):
            #head_data = datas[i][j][0]
            for k in range(0, datas_N):
                datas[i][j][k] = datas[i][j][k] - b2[j]
                
    for i in range(2 * inde_number, 3 * inde_number): 
        for j in range(0, datas_D):
            #head_data = datas[i][j][0]
            for k in range(0, datas_N):
                datas[i][j][k] = datas[i][j][k] - b3[j]
    
    """
    new_data1 = np.zeros((datas_D, datas_N))
    for i in range(datas_D):
        for j in range(datas_N):
            new_data1[i][j] = data1[i][j] - data1[i][0]
    #print(len(new_data1[0]))
    """
    
    datas_trans = np.array(datas[0])
    
    for i in range(1, datas_number):
        #print(np.shape(datas[i]))
        datas_trans = np.hstack((datas_trans, datas[i]))
        
    #print(np.shape(datas_trans))

    head = Get_time_axis()
    
    datas_frame1 = np.vstack((head, datas_trans))
    #print(datas_frame1[:,0])
    #print(np.shape(datas_frame1))


    
    """    
    gmr = GMM_GMR(3)
    gmr.fit(data)
    #get_mu, get_sigma = gmr.fitMatrix()
    
    #for j in range (0, 3):
        #print(get_sigma[:,:,j])
    
    #for i in range(3):
    #    print(get_sigma[:,:,i])
        
    timeInput = np.linspace(1, 98, 98)
    gmr.predict(timeInput)
    
    get_mu, get_sigma = gmr.getPredictedMatrix()
    get_mu = get_mu[1:8, :]
    

    fig = plt.figure()
    
    ax1 = fig.add_subplot(121)
    plt.title("Data")
    gmr.plot(ax=ax1, plotType="Data", xAxis=1, yAxis=2)

    ax2 = fig.add_subplot(122)
    plt.title("Gaussian States")
    gmr.plot(ax=ax2, plotType="Clusters", xAxis=1, yAxis=2)
    
    #predictedMatrix = gmr.getPredictedMatrix()
    #print(predictedMatrix.shape[0], predictedMatrix.shape[1], predictedMatrix)
    
    #while i < 98:
        #print(predictedMatrix[:,:,i])
        #i += 1
    """
    #plt.show()
    #np.savetxt('/home/zxc/桌面/Pdtw_head_forward_average.csv', datas_frame1, delimiter = ',')
    return (datas_frame1, datas_N, inde_number, b)   
    #return (data, get_mu, get_sigma)
        
    
if __name__ == "__main__":
    TP_GMM_frame1()
