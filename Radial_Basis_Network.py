# -*- coding: utf-8 -*-
"""
@author: yiping
"""

import numpy as np
from scipy.linalg import norm

class RBF:
    def __init__(self,NumofNeurons,_lambda = 0,_covar = 0.1):
        self.NumofNeurons = NumofNeurons
        self._lambda = _lambda
        self._covar = _covar
        
    #Gaussian function
    def BasisFunc(self,x,c):    
        return np.exp(-np.dot((x-c).T,(x-c))/self._covar**2)   
        
    def Train(self,x,y):
        N_data,dim = x.shape
        #rnd_indx = np.random.permutation(N_data)[:self.NumofNeurons]
        rnd_indx = np.random.randint(0,N_data,self.NumofNeurons)        

        self.centers = np.zeros((self.NumofNeurons,dim))
        H = np.zeros((N_data,self.NumofNeurons))
        for i in range(self.NumofNeurons):
            self.centers[i,:] = x[rnd_indx[i],:]
            for j in range(N_data):
                H[j,i] = self.BasisFunc(x[j,:],self.centers[i,:])
        
        I = np.identity(H.shape[1])
      
        self.weight = np.linalg.lstsq((np.dot(H.T,H) + \
        self._lambda*I), np.dot(H.T,y))[0]
     
        
    def Test(self,x):
        N_data,dim = x.shape
        H = np.zeros((N_data,self.NumofNeurons))
        for i in range(self.NumofNeurons):
            for j in range(N_data):
                H[j,i] = self.BasisFunc(x[j,:],self.centers[i,:])
        return np.dot(H,self.weight)
        