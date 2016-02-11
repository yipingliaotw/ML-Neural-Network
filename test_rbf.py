import numpy as np
from Radial_Basis_Network import RBF
from Extreme_Learning_Machine import ELM

elm= ELM(NumofHiddenNeurons = 10,_lambda = 0)

x = [[1,2,3,8,9],[3,2,1,7,8]]
y = [[0],[1]]


x = np.array(x)
y = np.array(y)


elm.Train(x,y)
