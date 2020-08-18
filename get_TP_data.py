#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:22:07 2019

@author: zxc
"""

import csv
import numpy as np

import pandas as pd

def Get_TP_data():
    filenames = [
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward1.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward2.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward3.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward4.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward5.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward6.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward7.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward8.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward9.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward10.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward11.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward12.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward13.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward14.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward15.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward16.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward17.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward18.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward19.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward20.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward21.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward22.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward23.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward24.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward25.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward26.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward27.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward28.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward29.csv',
                 '/home/zxc/catkin_ws_qt/src/2016坐标系数据已对齐/Pdtw_head_forward30.csv',]
    data_file = pd.read_csv(filenames[0])

    
    datas_D = 7  #每个数据文件行数
    datas_N = len(data_file.columns)  #每个数据文件列数
    #print(datas_N)
    datas_number = len(filenames)  #数据文件个数
    datas_TP = np.ndarray(shape = (datas_number, datas_D, datas_N))
    
    file_count = 0
    while file_count < datas_number:
        with open(filenames[file_count]) as f:
            reader = csv.reader(f)
            datas = []
            for line in reader:
                float_datas = []
                data = line #读取该文件里面每一行的数据
                for data_one in data:
                    data_one_new = float(data_one)
                    float_datas.append(data_one_new) #将行数据添加到float_datas里面
                datas.append(float_datas) #将列表float_datas数据添加到大列表datas里面
            datas_TP[file_count,:,:] = np.array(datas)

        file_count += 1

    return (datas_TP, datas_number, datas_D, datas_N)


if __name__ == "__main__":
    Get_TP_data()



