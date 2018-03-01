
# coding: utf-8

# In[2]:


import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


dataset = pd.read_csv('train.csv')


# In[4]:


dataset.head()


# In[5]:


print dataset.shape
x = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values


# In[6]:


from sklearn.cross_validation import train_test_split


# In[7]:


x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.8,random_state = 0)


# In[8]:


print x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[9]:


plt.figure(0)
plt.imshow(x_train[84].reshape(28,28),cmap = 'gray',interpolation = 'nearest')
print y_train[84]
plt.show()


# In[17]:


knn = KNeighborsClassifier(17)


# In[18]:


knn.fit(x_train.reshape(-1,784),y_train)


# In[19]:


pred = knn.predict(x_test)


# In[13]:


print pred.shape


# In[14]:


print pred


# In[15]:


print y_test

