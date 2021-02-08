# %%
import sys 
sys.path.append(r'../../../utility')
sys.path.append(r'../model')
import os
import numpy as np 
import random as r 
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import loss_functions as lf
import metrics as m  

from model import Discriminator

# loading data into RAM
data_path = r'..\..\Training Data\hv18'

file_path = []
for (root, dirs, files) in os.walk(data_path):
    for file_name in files:
        path = os.path.join(root,file_name)
        file_path.append(path)

r.shuffle(file_path)

data = []
labels = []
for im_path in file_path:
    image = cv2.imread(im_path)

    list_form = im_path.split(os.path.sep)
    label = list_form[-2]

    data.append(image)
    labels.append(label)

#casting labels into float32
labels = np.array(labels)
labels[np.where(labels == 'Positive')] = 1.
labels[np.where(labels == 'Negative')] = 0.
labels = labels.astype(np.float32)

data = np.array(data, dtype = np.float32) / 255.

#splitting data to train and validation data
split_indice = 40000
train_x = data[0:split_indice]
train_y = labels[0:split_indice]

val_x = data[split_indice:]
val_y = labels[split_indice:]
#%%
#instantiating model
model = Discriminator()
Epochs = 30
opt = tf.keras.optimizers.SGD(learning_rate = 0.001)
bs = 1

#%%
#defining metrics to monitor
train_prec = tf.keras.metrics.Precision(name = 'train_precision', thresholds = 0.75)
val_prec = tf.keras.metrics.Precision(name = 'validation_precision', thresholds = 0.75)

train_cost = m.BinaryFocalLoss(scale = 20)
val_cost = m.BinaryFocalLoss(scale = 20)

# Creating list for logging
tl = [] #train loss
tp = [] #train precision
vl = [] #validation loss
vp = [] #validation precision

#instatiating loss function
lossfunc = lf.BinaryFocalLoss(scale = 20)
# %%
#defining train step and validation step
@tf.function
def train_step(model, batch_im, batch_label):
    with tf.GradientTape() as tape:
        prediction = model(batch_im)
        cost = lossfunc(batch_label, prediction)
    grads = tape.gradient(cost, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return prediction, cost, grads[-1]

@tf.function
def validation_step(model, batch_im, batch_label):
    prediction = model(batch_im)
    cost = lossfunc(batch_label, prediction)
    return prediction, cost

# %%
#training network
for epoch in range(Epochs):
    train_prec.reset_states()
    val_prec.reset_states()
    train_cost.reset_states()
    val_cost.reset_states()

    tf.print('Epoch ', epoch+1, end = '\n')
    tf.print('Training ', end = '')

    for full_im, label in zip(train_x, train_y):
        full_im = np.expand_dims(full_im, axis = 0)
        label = np.reshape(label, (1,1))

        prediction, cost, grad = train_step(model, full_im, label)
        train_cost(label, prediction)
        train_prec(label, prediction)

    tf.print('\n')
    tf.print('Validating ', end = '')

    for full_im, label in zip(val_x, val_y):
        full_im = np.expand_dims(full_im, axis = 0)
        label = np.reshape(label, (1,1))

        prediction, cost = validation_step(model, full_im, label)
        val_cost(label, prediction)
        val_prec(label, prediction)

    tf.print('\n')

    template = '\nTraining Loss: {}\nValidation Loss: {}\nTrain Accuracy: {}\nValidation Accuracy: {}\nTrain Precision: {}\nValidation Precision: {}\n'
    print(template.format(  train_cost.result(), 
                            val_cost.result(), 
                            train_prec.result() * 100, 
                            val_prec.result() * 100))
    tl.append(train_cost.result())
    tp.append(train_prec.result())
    vl.append(val_cost.result())
    vp.append(val_prec.result())

    base = r'..\..\Output\tmp\weight check point'
    path = base + '\Epoch_' + str(epoch)
    model.save_weights(path)

model.save_weights(r'..\..\Output\Weights\Discriminator_hv18.h5')
N = np.arange(0, Epochs)
History = np.array([N, tl, ta, tp, vl, va, vp])
np.save(r'..\..\Output\History\Discriminator_hv18.npy', History)

# %%
