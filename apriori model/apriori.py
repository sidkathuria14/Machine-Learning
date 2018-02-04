
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[4]:


dataset = pd.read_csv('/home/sidkathuria14/Desktop/desktop/ML Code/Market_Basket_Optimisation.csv',header = None)


# In[20]:


transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])


# In[18]:


from apyori import apriori
rules = apriori(transactions,min_support= 0.003,min_confidence = 0.2 ,min_lift = 3,min_length = 2)


# In[19]:


results = list(rules)
print results

