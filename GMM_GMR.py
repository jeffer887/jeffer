#!/usr/bin/env python
# -*- coding: utf-8 -*-
from EM_init import *
from EM import *
from plotGMM import *
from GMR import *
from matplotlib import pyplot as plt
from scipy import interpolate
import numpy as np

class GMM_GMR(object):

    def __init__(self, numberOfStates):
        self.numbefOfStates = numberOfStates

    def fit(self, data):
        self.data = data
        Priors, Mu, Sigma = EM_init(data, self.numbefOfStates)

        self.Priors, self.Mu, self.Sigma, self.Pix = EM(data, Priors, Mu, Sigma)

    
    def fit_TP(self, data_TP):
        Priors_TP, Mu_TP, Sigma_TP = EM_init(data_TP, self.numbefOfStates)
	#print(Mu_TP)--->排序错乱
        return (Priors_TP, Mu_TP, Sigma_TP)

    
    def fitMatrix(self):
        return self.Mu, self.Sigma

    def predict(self, inputMat):
        nbVar, nbData = np.shape(self.data)
        self.expData = np.ndarray(shape=(nbVar, np.size(inputMat)))

        #inputMat:时间维度的长度N
        self.expData[0, :] = inputMat
        self.expData[1:nbVar, :], self.expSigma = GMR(self.Priors, self.Mu, self.Sigma, self.expData[0, :], 0,
                                                      np.arange(1, nbVar))

        
    def predict_TP(self, input_data, input_time, Priors, Mu, Sigma):
        nbVar, nbData = np.shape(input_data)
        self.expData_TP = np.ndarray(shape=(nbVar, np.size(input_time)))
        self.expData_TP[0, :] = input_time
        self.expData_TP[1:nbVar, :], self.expSigma_TP = GMR(Priors, Mu, Sigma, self.expData_TP[0, :], 0,
                                                      np.arange(1, nbVar))
        return(self.expData_TP, self.expSigma_TP)
        

    def getPredictedMatrix(self):
        #return self.expData
        #return self.expSigma
        return self.expData, self.expSigma
    
    def plot_gmm(self, data, Priors_tp, Mu_tp, Sigma_tp, xAxis = 0, yAxis = 1, plotType = "Clusters", ax = plt):
        xlim = [data[xAxis,:].min() - (data[xAxis,:].max() - data[xAxis,:].min())*0.1, data[xAxis,:].max() + (data[xAxis,:].max() - data[xAxis,:].min())*0.1]
        ylim = [data[yAxis,:].min() - (data[yAxis,:].max() - data[yAxis,:].min())*0.1, data[yAxis,:].max() + (data[yAxis,:].max() - data[yAxis,:].min())*0.1]   
        if plotType == "Clusters":
            rows = np.array([xAxis, yAxis])
            cols = np.arange(0, self.numbefOfStates, 1)
            plotGMM(Mu_tp[np.ix_(rows, cols)], Sigma_tp[np.ix_(rows, rows, cols)], [0, 0.8, 0], 1, ax)
        plt.xlim(xlim)
        plt.ylim(ylim)        
    
    def plot_expData(self, xAxis = 0, yAxis = 1, plotType = "Clusters", ax = plt, dataColor = [0, 0.8, 0.7], regressionColor = [0,0,0.8]):
        xlim = [self.expData_TP[xAxis,:].min() - (self.expData_TP[xAxis,:].max() - self.expData_TP[xAxis,:].min())*0.1, self.expData_TP[xAxis,:].max() + (self.expData_TP[xAxis,:].max() - self.expData_TP[xAxis,:].min())*0.1]
        ylim = [self.expData_TP[yAxis,:].min() - (self.expData_TP[yAxis,:].max() - self.expData_TP[yAxis,:].min())*0.1, self.expData_TP[yAxis,:].max() + (self.expData_TP[yAxis,:].max() - self.expData_TP[yAxis,:].min())*0.1]   
        #画均值
        ax.plot(self.expData_TP[xAxis, :], self.expData_TP[yAxis, :], color=dataColor)
        #画方差
        if plotType == "Regression":
            rows = np.array([xAxis, yAxis])
            rows2 = np.array([yAxis - 1, yAxis - 1])
            cols = np.arange(0, self.expData_TP.shape[1], 1)
            cols = cols.astype(int)
            plotGMM(self.expData_TP[np.ix_(rows, cols)], self.expSigma_TP[np.ix_(rows2, rows2, cols)], regressionColor, 2, ax)
        plt.xlim(xlim)
        plt.ylim(ylim)
        
        
    def plot_data(self, datas_N = 100, datas_number = 30, xAxis = 0, yAxis = 1, inter_n = 11, plotType = "Data", ax = plt, 
            dataColor = [0, 0.8, 0.7], clusterColor = [0, 0.8, 0], regressionColor = [0,0,0.8]):
        xlim = [self.data[xAxis,:].min() - (self.data[xAxis,:].max() - self.data[xAxis,:].min())*0.1, self.data[xAxis,:].max() + (self.data[xAxis,:].max() - self.data[xAxis,:].min())*0.1]
        ylim = [self.data[yAxis,:].min() - (self.data[yAxis,:].max() - self.data[yAxis,:].min())*0.1, self.data[yAxis,:].max() + (self.data[yAxis,:].max() - self.data[yAxis,:].min())*0.1]

        weights=np.hanning(inter_n)
        for i in range(0, 30):
            if i < datas_number - 1:
                x = self.data[xAxis, i * datas_N : (i+1) * datas_N]
                y = np.convolve(weights / weights.sum(), self.data[yAxis, i * datas_N : (i+1) * datas_N])[inter_n-1 : -inter_n+1]
                y_new = np.hstack((self.data[yAxis, i * datas_N : i * datas_N + (inter_n - 1) / 2], y))
                y_new = np.hstack((y_new, self.data[yAxis, (i+1) * datas_N - ((inter_n - 1) / 2): (i+1) * datas_N]))
                #y_new = np.hstack((y, self.data[yAxis, (i+1) * datas_N - (inter_n - 1) : (i+1) * datas_N]))
                ax.plot(x, y_new, color=dataColor)
            else:
                x = self.data[xAxis, i * datas_N :]
                y = np.convolve(weights / weights.sum(), self.data[yAxis, i * datas_N :])[inter_n-1 : -inter_n+1]
                y_new = np.hstack((self.data[yAxis, i * datas_N : i * datas_N + (inter_n - 1) / 2], y))
                y_new = np.hstack((y_new, self.data[yAxis, (i+1) * datas_N - ((inter_n - 1) / 2):]))
                #y_new = np.hstack((y, self.data[yAxis, (i+1) * datas_N - (inter_n - 1) :]))
                ax.plot(x, y_new, color=dataColor)
                #ax.plot(self.data[xAxis, i * datas_N :], self.data[yAxis, i * datas_N :], color=dataColor)
        plt.xlim(xlim)
        plt.ylim(ylim)
        
    def plot(self, xAxis = 0, yAxis = 1, plotType = "Clusters", ax = plt, dataColor = [0, 0.8, 0.7],
             clusterColor = [0, 0.8, 0], regressionColor = [0,0,0.8]):
        xlim = [self.data[xAxis,:].min() - (self.data[xAxis,:].max() - self.data[xAxis,:].min())*0.1, self.data[xAxis,:].max() + (self.data[xAxis,:].max() - self.data[xAxis,:].min())*0.1]
        ylim = [self.data[yAxis,:].min() - (self.data[yAxis,:].max() - self.data[yAxis,:].min())*0.1, self.data[yAxis,:].max() + (self.data[yAxis,:].max() - self.data[yAxis,:].min())*0.1]
        if plotType == "Data":
            #散点图
            #ax.plot(self.data[xAxis,:], self.data[yAxis,:],'.', color=dataColor) 
            #曲线图
            ax.plot(self.data[xAxis,:], self.data[yAxis,:], color=dataColor)
            plt.xlim(xlim)
            plt.ylim(ylim)
        elif plotType == "Clusters":
            rows = np.array([xAxis, yAxis])
            cols = np.arange(0, self.numbefOfStates, 1)
            plotGMM(self.Mu[np.ix_(rows, cols)], self.Sigma[np.ix_(rows, rows, cols)], [0, 0.8, 0], 1, ax)

            plt.xlim(xlim)
            plt.ylim(ylim)
        elif plotType == "Regression":
            rows = np.array([xAxis, yAxis])
            rows2 = np.array([yAxis - 1, yAxis - 1])
            cols = np.arange(0, self.expData.shape[1], 1)
            cols = cols.astype(int)
            plotGMM(self.expData[np.ix_(rows, cols)], self.expSigma[np.ix_(rows2, rows2, cols)], regressionColor, 2, ax)

            plt.xlim(xlim)
            plt.ylim(ylim)
        else:
            print "Invalid plot type.\nPossible choices are: Data, Clusters, Regression."
