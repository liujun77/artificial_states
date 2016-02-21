# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 15:02:38 2016

@author: liujun
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
#%%
V = -11/27.211
data = np.loadtxt('energy.txt')
data.view('i8,f8,f8').sort(order=['f2'], axis=0)
(n_, m_) = data.shape
index = np.arange(n_).astype(np.float32) / n_ /100000
energy = data[:,2]
energy = energy - V 
dtype = [('index', np.float32), ('energy', np.float32)]
train_data = np.column_stack( (index ,energy)) 
labels = data[:,0].astype(np.int32)
#%%
cf = KNeighborsClassifier(n_neighbors = 1)
cf.fit(train_data,labels)##train_data
#%%
cf.score(train_data,labels) ##train_data

#%%
de = -V / (n_+1)
test_energy = (np.arange(n_).astype(np.float32) +1)* de

test_data = np.column_stack((index, test_energy))

test_labels = cf.predict(test_data) ##test_data
#%%
def convolute(energy, labels, sig):
    out_ene = (np.arange(labels.shape[0]).astype(np.float32) +1)* de
    dos = labels *2 +1    
    out_dos = range(labels.shape[0])
    for i in range(n_):    
        out_dos += dos[i] * np.exp(- ((out_ene - energy[i])**2) / sig / sig)
    return out_ene, out_dos

#%%
conv_energy , conv_dos = convolute(energy,labels,-V/1000)
conv_energy , conv_dos_test = convolute(test_energy,test_labels,-V/1000)
#%%
np.savetxt('dos.txt', np.column_stack((energy, labels)))
np.savetxt('dos_test.txt', np.column_stack((test_energy, test_labels)))