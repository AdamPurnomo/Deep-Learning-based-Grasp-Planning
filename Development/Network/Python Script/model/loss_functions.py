# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 13:00:49 2020

@author: Adam Syammas Zaki P
"""
# %%
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss

import tensorflow as tf 
def weighted_binary_crossentropy( y_true, y_pred, weight=2. ) :
    y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
    y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
    logloss = -(y_true * K.log(y_pred) * weight + (1 - y_true) * K.log(1 - y_pred))
    return K.mean(logloss, axis=-1)

class WeightedBinaryCrossentropy(Loss):
    def __init__(self, weight = 2., **kwargs):
        self.weight = weight
        super(WeightedBinaryCrossentropy, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        y_true = K.clip(y_true, K.epsilon(), 1-K.epsilon())
        y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())
        
        logloss = -(y_true * tf.math.log(y_pred) * self.weight + (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.math.reduce_mean(logloss)

class BinaryFocalLoss(Loss):
    def __init__(self, alpha=0.75, gamma=2., scale = 20., **kwargs):
        self.alpha = alpha
        self.gamma = gamma
        self.scale = scale
        super(BinaryFocalLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        pt = (y_true*y_pred) + (1-y_true)*(1-y_pred)
        pt = K.clip(pt, K.epsilon(), 1-K.epsilon())

        at = (y_true*self.alpha) + (1 - y_true)*(1 - self.alpha)
        at = K.clip(at, K.epsilon(), 1-K.epsilon())

        loss = -self.scale*at*(1 - pt)**self.gamma * tf.math.log(pt)
        return tf.math.reduce_mean(loss)

