#!/usr/bin/env python
# coding: utf-8

# In[104]:


import numpy as np
import time
import idx2numpy
import os
from os.path import expanduser
import sklearn
from sklearn import linear_model


# In[105]:


home = expanduser("~")
mnist_path = os.path.join(home, "Datasets","MNIST")
[x for x in os.listdir(mnist_path) if x[0] !="." ]


# In[106]:


train_path = os.path.join(mnist_path,'train-images-idx3-ubyte')
test_path  = os.path.join(mnist_path,'t10k-images-idx3-ubyte')
train_y_path = os.path.join(mnist_path,'train-labels-idx1-ubyte')
test_y_path = os.path.join(mnist_path,'t10k-labels-idx1-ubyte')


# In[107]:


X_tr = idx2numpy.convert_from_file(train_path)
X_te = idx2numpy.convert_from_file(test_path)
y_tr = idx2numpy.convert_from_file(train_y_path)
y_te = idx2numpy.convert_from_file(test_y_path)

X_tr = X_tr.reshape((60000, 28*28))
X_te = X_te.reshape((10000, 28*28))

print(X_tr.shape)
print(X_te.shape)
print(y_tr.shape)
print(y_te.shape)


# In[108]:


model = sklearn.linear_model.Perceptron(max_iter=100,
                                        n_iter_no_change=100)


# In[110]:


print("training started")
t0 = time.time()
model.fit(X_tr,y_tr)
t = abs(time.time() - t0)


# In[112]:


print(f"train time {t} seconds")


# In[113]:


print("Accuracy train", np.mean(model.predict(X_tr) == y_tr))


# In[114]:


print("Accuracy test", np.mean(model.predict(X_te) == y_te))


# In[ ]:




