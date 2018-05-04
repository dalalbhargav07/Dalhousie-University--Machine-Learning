# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow
from tensorflow.examples.tutorials.mnist import input_data 

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_X,batch_y = mnist.train.next_batch(1)
import matplotlib.pyplot as plt

#1st way
plt.imshow(batch_X.reshape(28,28))
plt.show()

#2nd way
mnist.train.images[2].shape
img = mnist.train.images[2].reshape(28,28)
plt.imshow(img, cmap='gray')
plt.show()

img2 = mnist.train.images[2043].reshape(28,28)
plt.imshow(img2, cmap='gray')
plt.show()