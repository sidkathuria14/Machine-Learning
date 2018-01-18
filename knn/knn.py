#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 23:58:41 2018

@author: sidkathuria14
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
%matplotlib inline
from sklearn.neighbors import KNeighborsClassifier
sns.set_style('whitegrid')

df = sns.load_dataset('iris')
df.head()

x_train = df[['petal_length','petal_width']]
species_to_num = {'setosa':0,
                  'versicolor':1,
                  'virginica':2}
df['species'] = df['species'].map(species_to_num)
y_train = df['species']

knn = KNeighborsClassifier(n_neighbors = 20)
knn.fit(x_train,y_train)

xv = x_train.values.reshape(-1,1)
h = 0.02
x_min,x_max = xv.min(),xv.max()+1
y_min,y_max = y_train.min(),y_train.max() + 1
xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))

z = knn.predict(np.c_[xx.ravel(),yy.ravel()])
z = z.reshape(xx.shape)

fig = plt.figure(figsize = (8,5))

ax = plt.contourf(xx,yy,z,cmap = 'afmhot',alpha = 0.3)
plt.scatter(x_train.values[:,0],x_train.values[:,1],c = y_train,s = 40,alpha = 0.9,
            edgecolors = 'k')




