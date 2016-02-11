# -*- coding: utf-8 -*-
"""
@author: yiping
"""
import numpy as np
from scipy.linalg import pinv,inv

class ELM:
    def __init__(self,NumofHiddenNeurons,_lambda):  
        self.NumHiddenNeurons = NumofHiddenNeurons
        self._lambda = _lambda
            
    def ActivationFun(self,x):
        x = 1.0 / (1+np.exp(-x))
        return x
    
    def Train(self,x,y):
        N_data, dim = x.shape
        self.inputweight = np.random.rand(self.NumHiddenNeurons,dim)
        H = np.dot(x,self.inputweight.T)
        H = self.ActivationFun(H)
        I = np.identity(H.shape[1])
        self.outputweight = np.linalg.lstsq((np.dot(H.T,H) + self._lambda*I), np.dot(H.T,y))[0]
        
    def Test(self,x):
        H = np.dot(x,self.inputweight.T)
        H = self.ActivationFun(H)
        output = np.dot(H,self.outputweight)
        return self.ActivationFun(output)