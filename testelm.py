# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 17:58:45 2016

@author: yiping
"""

import numpy as np
from Extreme_Learning_Machine import ELM
from Radial_Basis_Network import RBF

def LoadData(filename):
    data=np.loadtxt(filename,delimiter=',')
    return data

inputs = LoadData("breastcancer_inputs.txt")
target = LoadData("breastcancer_target.txt")


inputs_train = inputs[0:500,:]
target_train = target[0:500,:]


inputs_test = inputs[500:699,:]
target_test = target[500:699,:]


elm = ELM(30,0)

elm.Train(inputs_train,target_train)
result = elm.Test(inputs_train)

label = np.zeros(result.shape)

for r in range(result.shape[0]):
    label[r,np.argmax(result[r,:])] = 1
    label[r,np.argmin(result[r,:])] = 0
    
print sum(target_train[:,0] == label[:,0])
print sum(target_train[:,0] == label[:,0]) / float(target_train.shape[0])



rbf =RBF(30,0,0.5)
rbf.Train(inputs_train,target_train)
result = rbf.Test(inputs_train)

label = np.zeros(result.shape)

for r in range(result.shape[0]):
    label[r,np.argmax(result[r,:])] = 1
    label[r,np.argmin(result[r,:])] = 0
    
print sum(target_train[:,0] == label[:,0])
print sum(target_train[:,0] == label[:,0]) / float(target_train.shape[0])




'''
result = elm.Test(inputs_test)

label = np.zeros(result.shape)

for r in range(result.shape[0]):
    label[r,np.argmax(result[r,:])] = 1
    label[r,np.argmin(result[r,:])] = 0
    
print sum(target_test[:,0] == label[:,0]) / float(target_test.shape[0])
'''