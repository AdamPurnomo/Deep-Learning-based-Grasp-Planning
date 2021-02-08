from tensorflow.keras.metrics import Metric
import tensorflow as tf
from tensorflow.keras import backend as K

class WeightedBinaryCrossEntropy(Metric):
    def __init__(self, name="Weighted Binary Crossenropy", weight = 1.0, **kwargs):
        super(WeightedBinaryCrossEntropy, self).__init__(name=name, **kwargs)
        self.wbce = self.add_weight(name="wbce", initializer="zeros")
        self.i = self.add_weight(name='iterator', initializer='zeros')
        self.weight = weight

    def update_state(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        logloss = -(y_true * tf.math.log(y_pred) * self.weight + (1 - y_true) * tf.math.log(1 - y_pred))
        self.wbce.assign_add(tf.reduce_mean(logloss))
        self.i.assign_add(tf.constant(1.))

    def result(self):
        return self.wbce / self.i

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.wbce.assign(0.0)
        self.i.assign(0.0)

class BinaryFocalLoss(Metric):
    def __init__(self, name = 'Binary Focal Loss', alpha=0.25, gamma=2.0, scale = 1, **kwargs):
        super(BinaryFocalLoss, self).__init__(**kwargs)
        self.bfl = self.add_weight(name="bfl", initializer="zeros")
        self.i = self.add_weight(name='iterator', initializer='zeros')
        self.alpha = alpha
        self.gamma = gamma
        self.scale = scale

        
    def update_state(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        pt = (y_true*y_pred) + (1-y_true)*(1-y_pred)
        pt = K.clip(pt, K.epsilon(), 1-K.epsilon())

        at = (y_true)*self.alpha + (1-y_true)*(1-self.alpha)
        at = K.clip(at, K.epsilon(), 1-K.epsilon())
        loss = self.scale*-at*(1 - pt)**self.gamma * tf.math.log(pt)
        self.bfl.assign_add(tf.reduce_mean(loss))
        self.i.assign_add(tf.constant(1.))

    def result(self):
        return self.bfl / self.i

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.bfl.assign(0.0)
        self.i.assign(0.0)
