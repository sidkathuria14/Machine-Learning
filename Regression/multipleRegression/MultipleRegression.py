
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv
('Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split
(x,y,test_size = 0.2,random_state = 0