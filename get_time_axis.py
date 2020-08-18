#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 22:13:48 2019

@author: zxc
"""

import numpy as np
from get_TP_data import Get_TP_data

def Get_time_axis():
    _, datas_number, _, time_len = Get_TP_data()
    time_axis = np.linspace(1, time_len, num = time_len)  #np.linspace, num可选，=50为默认值
    #print(time_axis)
    
    all_time_axis = time_axis
    for i in range(1, datas_number):
        all_time_axis = np.hstack((all_time_axis, time_axis))
    
    #print(all_time_axis, np.shape(all_time_axis))    
    return (all_time_axis)
    
if __name__ == "__main__":
    Get_time_axis()