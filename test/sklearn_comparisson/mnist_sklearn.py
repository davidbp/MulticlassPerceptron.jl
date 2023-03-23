#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
import idx2numpy
import os
from os.path import expanduser
import sklearn
from sklearn import linear_model

home = expanduser("~")
mnist_path = os.path.join(home, "Datasets","MNIST")
[x for x in os.listdir(mnist_path) if x[0] !="." ]

# Data can be downloaded from  http://yann.lecun.com/exdb/mnist/
train_path = os.path.join(mnist_path,'train-images-idx3-ubyte')
test_path  = os.path.join(mnist_path,'t10k-images-idx3-ubyte')
train_y_path = os.path.join(mnist_path,'train-labels-idx1-ubyte')
test_y_path = os.path.join(mnist_path,'t10k-labels-idx1-ubyte')

X_tr = idx2numpy.convert_from_file(train_path)
X_te = idx2numpy.convert_from_file(test_path)
y_tr = idx2numpy.convert_from_file(train_y_path)
y_te = idx2numpy.convert_from_file(test_y_path)

X_tr = X_tr.reshape((60000, 28*28))
X_te = X_te.reshape((10000, 28*28))

X_tr = np.array(X_tr, dtype=np.float32)
X_te = np.array(X_te, dtype=np.float32)

print(X_tr.shape)
print(X_te.shape)
print(y_tr.shape)
print(y_te.shape)

model = sklearn.linear_model.Perceptron(max_iter=100,
                                        n_iter_no_change=100)

print("training started")
t0 = time.time()
model.fit(X_tr,y_tr)
t = abs(time.time() - t0)

print(f"train time {t} seconds")
print("Accuracy train", np.mean(model.predict(X_tr) == y_tr))
print("Accuracy test", np.mean(model.predict(X_te) == y_te))




