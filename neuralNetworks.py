# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 16:39:40 2023

@author: MON PC
"""

import tensorflow as tf 
from tensorflow.keras.datasets import mnist
import numpy as np 
import matplotlib.pyplot as plt
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam

(X_train, y_train), (X_test, y_test) = mnist.load_data()


idx = random.randint(0,X_train.shape[0]-1)
img2show, label = X_train[idx],y_train[idx]
print(f'The image wee see is {label}')
plt.imshow(img2show, cmap='gray')


X_train, X_test = X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)
y_train, y_test=tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)
X_train, X_test = X_train/255.0, X_test/255.0


model = Sequential()
model.add(Dense(784, activation='relu', input_shape = (784,)))
model.add(Dense(36, activation = 'relu'))
model.add(Dense(10, activation='softmax'))
model.summary()


model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=20, batch_size=36, validation_data=(X_test, y_test))


idx_test = random.randint(0, X_test.shape[0]-1)
img_test = X_test[idx_test]
img_test = np.expand_dims(img_test, axis=0)
prediction = model.predict(img_test)
predict_label = np.argmax(prediction)
true_label = y_test[idx_test]
print(f'The predicted label is {predict_label} and the true label is {np.argmax(true_label)}')


