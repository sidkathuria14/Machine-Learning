
# coding: utf-8

# In[31]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from sklearn.cluster import KMeans
get_ipython().magic(u'matplotlib inline')


# In[32]:


im = cv2.imread('/home/sidkathuria14/Desktop/img2.jpg')
im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
r,c=im.shape[:2]
out_r= 120
im=cv2.resize(im,(int(out_r*float(c)/r), out_r))
print (im.shape)
pixel=im.reshape(-1,3)
print (pixel.shape)
plt.imshow(im)

km = KMeans(n_clusters = 8)
km.fit(pixel)


# In[33]:


colors=np.asarray(km.cluster_centers_,dtype='uint8')
per=np.asarray(np.unique(km.labels_,return_counts=True)[1],dtype='float32')
per = per/pixel.shape[0]
print per


# In[34]:


plt.figure(0)
for ix in range(colors.shape[0]):
    patch=np.ones((20,20,3))
    patch[:,:,:]=255- colors[ix]
    plt.subplot(1,colors.shape[0],ix+1)
    plt.axis('off')
    plt.imshow(patch)


# In[35]:



dom=[[per[ix],colors[ix]] for ix in range(colors.shape[0])]
Dom=sorted(dom, key =lambda z:z[0],reverse=True)
print (dom)
print (Dom)


# In[36]:


plt.figure(0)

patch= np.zeros((50,500,3))
start=0
for ix in range(km.n_clusters):
    width = int(Dom[ix][0] * patch.shape[1])
    end = start + width
    patch[:,start:end,:]=255-Dom[ix][1]
    start=end
plt.axis('off')
plt.imshow(patch)

