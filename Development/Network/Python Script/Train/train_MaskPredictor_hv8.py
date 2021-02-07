# %%
import os
import numpy as np 
import random as r 
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import loss_functions as lf
import metrics as m  

from model import MaskPredictor, Discriminator

# loading data into RAM
data_path = r'..\..\Training Data\Mask\hv8'

file_path = []
for (root, dirs, files) in os.walk(data_path):
    for file_name in files:
        path = os.path.join(root,file_name)
        list_form = path.split(os.path.sep)
        if(list_form[-2] == 'full'):
            file_path.append(path)

r.shuffle(file_path)

data = []
labels = []
for im_path in file_path:
    image = cv2.imread(im_path)

    list_form = im_path.split(os.path.sep)
    file_name = list_form[-1]
    ext = file_name.split(sep='.')[1]
    name = file_name.split(sep='.')[0]
    
    label_name = name + '.npy'
    label_path = list_form
    label_path[-1] = label_name
    label_path[-2] = 'sparse matrix'
    label_path = os.path.sep.join(label_path)
    label = np.load(label_path)

    data.append(image)
    labels.append(label)

#casting labels into float32
labels = np.array(labels, dtype = 'float32')
data = np.array(data, dtype = np.float32) / 255.

#splitting data to train and validation data
split_indice = 23000
train_x = data[0:split_indice]
train_y = labels[0:split_indice]

val_x = data[split_indice:]
val_y = labels[split_indice:]

#%%
#defining metrics to monitor
train_prec = tf.keras.metrics.Precision(name = 'train_precision', thresholds = 0.5)
val_prec = tf.keras.metrics.Precision(name = 'validation_precision', thresholds = 0.5)
train_mse = tf.keras.metrics.MeanSquaredError()
val_mse = tf.keras.metrics.MeanSquaredError()

train_cost = m.BinaryFocalLoss(scale= 20000)
val_cost = m.BinaryFocalLoss(scale= 20000)

# Creating list for logging
tl = [] #train loss
tp = [] #train precision
vl = [] #validation loss
vp = [] #validation precision
tm = [] #train mse
vm = [] #validation mse

#instatiating loss function
lossfunc = lf.BinaryFocalLoss(scale= 20000)

#instantiating model
d_model = Discriminator()
d_model.build((1,220,220,3))
d_model.load_weights(r'..\..\Training Output\Weights\Discriminator_hv8')
model = MaskPredictor()
model.build((1,220,220,3))

#transferring weight
for i in range(7):
    weight = d_model.layers[i].get_weights()
    model.layers[i].set_weights(weight)
    model.layers[i].trainable = False

Epochs = 20
opt = tf.keras.optimizers.SGD(learning_rate = 0.001)
bs = 1

#continue previous training
model.load_weights(r'..\..\Training Output\tmp\weight check point\Epoch_5')

#defining train step and validation step
# %%
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

#training network
for epoch in range(6,Epochs):
    train_prec.reset_states()
    val_prec.reset_states()
    train_cost.reset_states()
    val_cost.reset_states()
    train_mse.reset_states()
    val_mse.reset_states()

    tf.print('Epoch ', epoch, end = '\n')
    tf.print('Training ', end = '')

    for full_im, label in zip(train_x, train_y):
        full_im = np.expand_dims(full_im, axis = 0)
        label = np.expand_dims(label, axis = [-1,0])

        prediction, cost, grad = train_step(model, full_im, label)
    
        index_true = np.where(label[0,:,:,0] == 1)
        index_true = np.array([[index_true[0][0], index_true[1][0]]], dtype = 'float32')

        index_pred = np.unravel_index(np.argmax(prediction[0,:,:,0]), prediction[0,:,:,0].shape)
        index_pred = np.array(index_pred, dtype = 'float32').reshape((1,2))
      
        train_cost(label, prediction)
        train_prec(label, prediction)
        train_mse(index_true, index_pred)

    tf.print('\n')
    tf.print('Validating ', end = '')

    for full_im, label in zip(val_x, val_y):
        full_im = np.expand_dims(full_im, axis = 0)
        label = np.expand_dims(label, axis = [-1,0])

        prediction, cost = validation_step(model, full_im, label)

   
        index_true = np.where(label[0,:,:,0] == 1)
        index_true = np.array([[index_true[0][0], index_true[1][0]]], dtype = 'float32')

        index_pred = np.unravel_index(np.argmax(prediction[0,:,:,0]), prediction[0,:,:,0].shape)
        index_pred = np.array(index_pred, dtype = 'float32').reshape((1,2))
      

        val_cost(label, prediction)
        val_prec(label, prediction)
        val_mse(index_true, index_pred)
    tf.print('\n')

    template = '\nTraining Loss: {}\nValidation Loss: {}\nTrain Precision: {}\nValidation Precision: {}\nTrain Mean Pixel Distance: {}\nValidation Mean Pixel Distance : {}\n'
    print(template.format(  train_cost.result(), 
                            val_cost.result(), 
                            train_prec.result() * 100, 
                            val_prec.result() * 100,
                            tf.math.sqrt(2*train_mse.result()),
                            tf.math.sqrt(2*val_mse.result())
                            ))
    print(cost, '\n')
    tl.append(train_cost.result())
    tp.append(train_prec.result())
    vl.append(val_cost.result())
    vp.append(val_prec.result())
    tm.append(tf.math.sqrt(2*train_mse.result()))
    vm.append(tf.math.sqrt(2*train_mse.result()))
    
    #saving for check point
    base = r'..\..\Training Output\tmp\weight check point'
    dir = base + r'\Epoch_' + str(epoch)
    model.save_weights(dir)

model.save_weights(r'..\..\Training Output\Weights\MaskPredictor_hv8.h5')
model.save_weights(r'..\..\Training Output\Weights\MaskPredictor_hv8')
N = np.arange(0, Epochs)
History = np.array([N, tl, tp, tm, vl, vp, vm])
np.save(r'..\..\Training Output\History\MaskPredictor_hv8.npy', History)




# %%
