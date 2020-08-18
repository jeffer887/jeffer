#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 21:04:57 2019

@author: zxc
"""

import numpy as np

from get_time_axis import Get_time_axis
from get_TP_data import Get_TP_data

#得到第二个坐标系（终点坐标系）下的数据分布
def TP_GMM_frame2():
    
    datas, datas_number, datas_D, datas_N = Get_TP_data()
    
    inde_number = datas_number / 3
    
    b1 = [datas[0][0][datas_N - 1], datas[0][1][datas_N - 1], datas[0][2][datas_N - 1], datas[0][3][datas_N - 1],
          datas[0][4][datas_N - 1], datas[0][5][datas_N - 1], datas[0][6][datas_N - 1]]
    b2 = [datas[10][0][datas_N - 1], datas[10][1][datas_N - 1], datas[10][2][datas_N - 1], datas[10][3][datas_N - 1],
          datas[10][4][datas_N - 1], datas[10][5][datas_N - 1], datas[10][6][datas_N - 1]]
    b3 = [datas[20][0][datas_N - 1], datas[20][1][datas_N - 1], datas[20][2][datas_N - 1], datas[20][3][datas_N - 1],
          datas[20][4][datas_N - 1], datas[20][5][datas_N - 1], datas[20][6][datas_N - 1]]
    
    b = [b1, b2, b3]
    
    #tail_data = datas[0][0][datas_N - 1]
    for i in range(0, inde_number): 
        for j in range(0, datas_D):
            #tail_data = datas[i][j][datas_N - 1]
            for k in range(0, datas_N):
                datas[i][j][k] = datas[i][j][k] - b1[j]
                
    for i in range(inde_number, 2 * inde_number): 
        for j in range(0, datas_D):
            #tail_data = datas[i][j][datas_N - 1]
            for k in range(0, datas_N):
                datas[i][j][k] = datas[i][j][k] - b2[j]
                
    for i in range(2 * inde_number, 3 * inde_number): 
        for j in range(0, datas_D):
            #tail_data = datas[i][j][datas_N - 1]
            for k in range(0, datas_N):
                datas[i][j][k] = datas[i][j][k] - b3[j]
    
    datas_trans = np.array(datas[0])
    
    for i in range(1, datas_number):
        #print(np.shape(datas[i]))
        datas_trans = np.hstack((datas_trans, datas[i]))
        
    #print(np.shape(datas_trans))

    head = Get_time_axis()
    
    datas_frame2 = np.vstack((head, datas_trans))
    #print(np.shape(datas_frame2))
    #print(datas_frame2[1])
            
    #np.savetxt('/home/zxc/桌面/Pdtw_head_forward_average_back.csv', datas_frame2, delimiter = ',')
    return (datas_frame2, datas_N, inde_number, b)
    
if __name__ == "__main__":
    TP_GMM_frame2()
