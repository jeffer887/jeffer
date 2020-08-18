#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 22:42:11 2019

@author: zxc
"""

import numpy as np

from matplotlib import pyplot as plt

from EM import EM
from GMM_GMR import GMM_GMR
from TP_EM import TP_EM
from TP_task import TP_task
from TP_task_new import TP_task_new

from get_TP_data import Get_TP_data
from TP_GMM1 import TP_GMM_frame1
from TP_GMM2 import TP_GMM_frame2

if __name__ == "__main__":
    
    data1, datas_N, inde_number, b1 = TP_GMM_frame1()
    data2, _, _, _ = TP_GMM_frame2()
    
    #获得时间轴
    time_axis = np.linspace(1, datas_N, num = datas_N)
    all_time_axis = time_axis
    for i in range(1, inde_number):
        all_time_axis = np.hstack((all_time_axis, time_axis))    
    
    #分别对三组数据进行处理，得到均值和方差
    data11 = np.vstack((all_time_axis, data1[1:8, 0:inde_number * datas_N]))
    data12 = np.vstack((all_time_axis, data1[1:8, inde_number * datas_N:2 * inde_number * datas_N]))
    data13 = np.vstack((all_time_axis, data1[1:8, 2 * inde_number * datas_N:]))    

    data21 = np.vstack((all_time_axis, data2[1:8, 0:inde_number * datas_N]))
    data22 = np.vstack((all_time_axis, data2[1:8, inde_number * datas_N:2 * inde_number * datas_N]))
    data23 = np.vstack((all_time_axis, data2[1:8, 2 * inde_number * datas_N:]))    
    
    #确定高斯分量的个数，后期BIC改进
    nbStates = 3
    
    #建立模型
    gmr = GMM_GMR(nbStates)
    
    #gmr1 = gmr.fit_TP(data1)
    #gmr2 = gmr.fit_TP(data2)
    #print(gmr1)
    #print(gmr2)
    
    #获得两坐标系下的数据--->可能出现问题
    Priors11, Mu11, Sigma11 = gmr.fit_TP(data11)
    Priors12, Mu12, Sigma12 = gmr.fit_TP(data12)
    Priors13, Mu13, Sigma13 = gmr.fit_TP(data13)
    #Priors2, Mu2, Sigma2 = gmr.fit_TP(data2)
    Priors21, Mu21, Sigma21 = gmr.fit_TP(data21)
    Priors22, Mu22, Sigma22 = gmr.fit_TP(data22)
    Priors23, Mu23, Sigma23 = gmr.fit_TP(data23)
    #print(np.shape(Priors1), np.shape(Mu1), np.shape(Sigma1))
    #print(np.shape(Priors2), np.shape(Mu2), np.shape(Sigma2))
    #print(Priors1, Priors2) #两概率并不相同，后期改进
    
    #EM迭代求最优解
    #Priors, Mu1, Mu2, Sigma1, Sigma2 = TP_EM(Priors1, data1, data2, Mu1, Mu2, Sigma1, Sigma2)
    Priors1, Mu11, Mu21, Sigma11, Sigma21 = TP_EM(Priors11, data11, data21, Mu11, Mu21, Sigma11, Sigma21)
    Priors2, Mu12, Mu22, Sigma12, Sigma22 = TP_EM(Priors12, data12, data22, Mu12, Mu22, Sigma12, Sigma22)
    Priors3, Mu13, Mu23, Sigma13, Sigma23 = TP_EM(Priors13, data13, data23, Mu13, Mu23, Sigma13, Sigma23)
    #print(np.shape(Priors), np.shape(Mu1), np.shape(Sigma1))
    #print(np.shape(Mu2), np.shape(Sigma2))
    
    #得到每次示教下的均值和方差
    Mu1, Sigma1 = TP_task_new(Mu11, Mu21, Sigma11, Sigma21, nbStates)
    Mu2, Sigma2 = TP_task_new(Mu12, Mu22, Sigma12, Sigma22, nbStates)
    Mu3, Sigma3 = TP_task_new(Mu13, Mu23, Sigma13, Sigma23, nbStates)
    #Mu, Sigma = TP_task_new(Mu1, Mu2, Sigma1, Sigma2, nbStates)
    #print(np.shape(Mu), np.shape(Sigma))
    
    #给定一个终点，得到该终点所在坐标系下的数据,终点=b1
    _, _, datas_D, _ = Get_TP_data()
    for i in range(1, datas_D + 1):
        for j in range(0, datas_N):
            data1[i][j] = data1[i][j] + b1[0][i - 1]
            
    #得到GMM, 共七次迭代  fit_TP 含有 EM_init
    Priors_TP, Mu_TP, Sigma_TP = gmr.fit_TP(data1)
    Priors_TP, Mu_TP, Sigma_TP, _ = EM(data1, Priors_TP, Mu_TP, Sigma_TP)
    
    """
    for i in range(1, 8):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        gmr.plot_gmm(ax = ax1, data = data1, Priors_tp = Priors_TP, Mu_tp = Mu_TP, Sigma_tp = Sigma_TP,
                     plotType = "Clusters", xAxis = 0, yAxis = i)
        #fig.savefig('/home/zxc/桌面/TP_fig_new/GMM/theta' + str(i) + '.jpg', dpi = 100)
    """
    

    #回归
    timeInput = np.linspace(1, datas_N, datas_N)
    Priors = (Priors1 + Priors2 + Priors3) / 3
    expData, expSigma = gmr.predict_TP(data1, timeInput, Priors, Mu1, Sigma1)
    #print(np.shape(expData), np.shape(expSigma))
    #print(expData, expSigma)
    
    #数据导出
    """
    i = 0
    while i < max_number:
        np.savetxt('/home/zxc/catkin_ws_qt/src/均值-方差数据/TP_cov_datas_/Pdtw_mouse_forward_cov/cov'+str(i + 1)+'.csv', expSigma[:,:,i], delimiter = ',')
        i += 1
    """
    #np.savetxt('/home/zxc/catkin_ws_qt/src/均值-方差数据/TP_average_datas_/Pdtw_mouse_forward_average.csv', expData, delimiter = ',')
    
    #画出图形
    """
    for i in range(1, 8):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #plt.title('expData-theta' + str(i))
        gmr.plot_expData(ax = ax, xAxis = 0, yAxis = i)
        #fig.savefig('/home/zxc/桌面/TP_fig/GMR/GMR-theta' + str(i) + '.jpg', dpi = 100)
    """
    for i in range(1, 8):
        fig = plt.figure()
        ax2 = fig.add_subplot(111)
        gmr.plot_expData(ax = ax2, plotType="Regression", xAxis = 0, yAxis = i)
        #fig.savefig('/home/zxc/桌面/TP_fig_new/GMR/GMR-theta' + str(i) + '.jpg', dpi = 100)

    plt.show()

    
    
    
