#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 02:01:25 2017

@author: sidkathuria14
"""

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LinearRegression
regresso = LinearRegression()
regresso.fit(x_train,y_train)

y_pred = regresso.predict(x_test)

plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_test,regresso.predict(x_test),color = 'blue')
plt.title('Salary vs. Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
