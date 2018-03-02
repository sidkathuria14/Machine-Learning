
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd
from nltk.corpus import names
import nltk
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords


# In[34]:


# nltk.download()
names = ( [(name,'male') for name in names.words('male.txt')] +
           [(name,'female') for name in names.words('female.txt')])
# names = ([(name, 'male') for name in names.words('male.txt')] + 
# 	 [(name, 'female') for name in names.words('female.txt')])


# In[35]:


def gender_features(word): 
    return {'last_letter': word[-1]}


# In[36]:


featuresets = [(gender_features(n), g) for (n,g) in names] 
train_set = featuresets
classifier = nltk.NaiveBayesClassifier.train(train_set) 


# In[44]:


print(classifier.classify(gender_features('thomas')))

