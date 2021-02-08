# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Activation

# %%
class conv_module(Layer):
    
    '''
    A series of convolution and batch normalization layer.
    x: input tensor
    K: num of filters/kernels
    kx: row kernel size
    ky: column kernel size
    '''

    def __init__(self, K, kx, ky, stride, chanDim=-1, padding="same"):
        super(conv_module, self).__init__()
        self.conv = Conv2D(K, (kx, ky), strides=stride, padding=padding)
        self.activation =  Activation("relu")
        
    def call(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
class ResNet_module (Layer):
    '''
    '''
    def __init__(self, num_filt):
        super(ResNet_module, self).__init__()
        self.conv1 = conv_module(num_filt, 3, 3, 1)
        self.conv2 = conv_module(num_filt, 3, 3, 1)
        self.conv3 = conv_module(num_filt, 3, 3, 1)
        self.conv4 = conv_module(num_filt, 3, 3, 1)
    
    def call(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2 = tf.concat([x2, x], axis = -1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x4 = tf.concat([x4, x2], axis = -1)
        return x4

# %%        
class dense_module(Layer):
    '''
    A series of dense, batch normalization and dropout layer.
    x: input tensor
    node: num of nodes
    f: activation function 
    '''
    def __init__(self, node, f):
        super(dense_module, self).__init__()
        self.dense = Dense(node)
        self.activation = Activation(f)
        self.dropout = Dropout(0.5)

    def call(self, x):
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)   
        return x

# %%
class SpatialPyramidPooling(Layer):
    '''
    #input shape
    4D tensor with shape:
        (batch, rows, cols, channels) since it is tf ordering
    #Output shape
    2D tensor with shape:
        (batch, channels*sum([i*i for i in pool list]))
    '''
    def __init__(self, pool_list, **kwargs):
        
        
        self.pool_list = pool_list
        self.num_output_per_channel = sum([i*i for i in pool_list])
        
        super(SpatialPyramidPooling, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.nb_channels = input_shape[3]
        
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nb_channels * self.num_output_per_channel)
    
    def get_config(self):
        config = {'pool_list' : self.pool_list}
        base_config = super(SpatialPyramidPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def call(self, x, mask = None):
        
        input_shape = tf.shape(x)
        
        num_rows = tf.cast(input_shape[1], 'float32')
        num_cols = tf.cast(input_shape[2], 'float32')
        
        row_length = [tf.cast(tf.floor(num_rows/tf.cast(i, 'float32')) + num_rows % tf.cast(i, 'float32'), 'int32') for i in self.pool_list]
        col_length = [tf.cast(tf.floor(num_cols/tf.cast(i, 'float32')) + num_cols % tf.cast(i, 'float32'), 'int32') for i in self.pool_list]
        
        row_strides = [tf.cast(tf.floor(num_rows/i), 'int32') for i in self.pool_list]
        col_strides = [tf.cast(tf.floor(num_cols/i), 'int32') for i in self.pool_list]      
        
        outputs = []
        
        for pool_num, num_pool_regions in enumerate(self.pool_list):
            for jy in range(num_pool_regions):
                for ix in range(num_pool_regions):
                    x1 = ix*col_strides[pool_num]
                    x2 = x1 + col_length[pool_num]
                    
                    y1 = jy*row_strides[pool_num]
                    y2 = y1 + row_length[pool_num]
                    
                    new_shape = [input_shape[0], y2-y1, x2-x1, input_shape[3]]
                    
                    x_crop = x[:, y1:y2, x1:x2, :]
                    xm = tf.reshape(x_crop, new_shape)
                    pooled_val = tf.keras.backend.max(xm, axis = (1,2))
                    outputs.append(pooled_val)
                    
        outputs = tf.keras.backend.concatenate(outputs)
        
        return outputs

class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.stage1 = ResNet_module(64)
        self.maxpool1 =  MaxPooling2D(pool_size = (2,2))
        self.stage2 = ResNet_module(128)
        self.maxpool2 =  MaxPooling2D(pool_size = (2,2))
        self.stage3 = ResNet_module(256)
        self.maxpool3 = MaxPooling2D(pool_size = (2,2))
        self.stage4 = ResNet_module(512)

        self.spp = SpatialPyramidPooling([1])
        self.dense1 = dense_module(512, 'relu')
        self.dense2 = dense_module(256, 'relu')
        self.pred = Dense(units = 1, activation = 'sigmoid')
    
    def call(self, u):
        u1 = self.stage1(u)
        u1 = self.maxpool1(u1)
        u2 = self.stage2(u1)
        u2 = self.maxpool2(u2)
        u3 = self.stage3(u2)
        u3 = self.maxpool3(u3)
        u4 = self.stage4(u3)
        
        u5 = self.spp(u4)
        w = self.dense1(u5)
        w = self.dense2(w)
        y = self.pred(w)
        return y

class MaskPredictor(Model):
    def __init__(self):
        super(MaskPredictor, self).__init__()
        self.stage1 = ResNet_module(64)
        self.maxpool1 =  MaxPooling2D(pool_size = (2,2))
        self.stage2 = ResNet_module(128)
        self.maxpool2 =  MaxPooling2D(pool_size = (2,2))
        self.stage3 = ResNet_module(256)
        self.maxpool3 = MaxPooling2D(pool_size = (2,2))
        self.stage4 = ResNet_module(512)

        self.padding = tf.keras.layers.ZeroPadding2D(padding = ((0,1), (0,1)))
        self.upsample1 = UpSampling2D(size = (2,2))
        self.stage5 = ResNet_module(256)
        self.upsample2 = UpSampling2D(size=(2,2))
        self.stage6 = ResNet_module(128)
        self.upsample3 = UpSampling2D(size=(2,2))
        self.stage7 = ResNet_module(64)
        self.kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0, stddev = 0.001)
        self.bias_initializer = tf.keras.initializers.Constant(-4)
        self.last = Conv2D(1, (1, 1), strides=1, padding='same', 
                            kernel_initializer= self.kernel_initializer, 
                            bias_initializer=self.bias_initializer)
        self.activation =  Activation("sigmoid")
    
    def call(self, u):
        u1 = self.stage1(u)

        u2 = self.maxpool1(u1)
        u2 = self.stage2(u2)
        
        u3 = self.maxpool2(u2)
        u3 = self.stage3(u3)

        u4 = self.maxpool3(u3)
        u4 = self.stage4(u4)

        u5 = self.upsample1(u4)
        u5 = self.padding(u5)
        u5 = tf.concat([u5, u3], axis = -1)
        u5 = self.stage5(u5)

        u6 = self.upsample2(u5)
        u6 = tf.concat([u6,u2], axis = -1)
        u6 = self.stage6(u6)

        u7 = self.upsample3(u6)
        u7 = tf.concat([u7, u1], axis = -1)
        y = self.last(u7)
        y = self.activation(y)

        return y

class GraspNet(Model):
    def __init__(self):
        super(GraspNet, self).__init__()
        self.stage1 = ResNet_module(64)
        self.maxpool1 =  MaxPooling2D(pool_size = (2,2))
        self.stage2 = ResNet_module(128)
        self.maxpool2 =  MaxPooling2D(pool_size = (2,2))
        self.stage3 = ResNet_module(256)
        self.maxpool3 = MaxPooling2D(pool_size = (2,2))
        self.stage4 = ResNet_module(512)

        self.spp = SpatialPyramidPooling([1])
        self.dense1 = dense_module(512, 'relu')
        self.dense2 = dense_module(256, 'relu')
        self.pred = Dense(units = 1, activation = 'sigmoid')

        self.padding = tf.keras.layers.ZeroPadding2D(padding = ((0,1), (0,1)))
        self.upsample1 = UpSampling2D(size = (2,2))
        self.stage5 = ResNet_module(256)
        self.upsample2 = UpSampling2D(size=(2,2))
        self.stage6 = ResNet_module(128)
        self.upsample3 = UpSampling2D(size=(2,2))
        self.stage7 = ResNet_module(64)
        self.kernel_initializer = tf.keras.initializers.RandomNormal(mean = 0, stddev = 0.001)
        self.bias_initializer = tf.keras.initializers.Constant(-4)
        self.last = Conv2D(1, (1, 1), strides=1, padding='same', 
                            kernel_initializer= self.kernel_initializer, 
                            bias_initializer=self.bias_initializer)
        self.activation =  Activation("sigmoid")
    
    def call(self, u):
        u1 = self.stage1(u)
        u2 = self.maxpool1(u1)
        u2 = self.stage2(u2)
        u3 = self.maxpool2(u2)
        u3 = self.stage3(u3)
        u4 = self.maxpool3(u3)
        u4 = self.stage4(u4)

        u5 = self.spp(u4)
        w = self.dense1(u5)
        w = self.dense2(w)
        score = self.pred(w)

        u5 = self.upsample1(u4)
        u5 = self.padding(u5)
        u5 = tf.concat([u5, u3], axis = -1)
        u5 = self.stage5(u5)
        u6 = self.upsample2(u5)
        u6 = tf.concat([u6,u2], axis = -1)
        u6 = self.stage6(u6)
        u7 = self.upsample3(u6)
        u7 = tf.concat([u7, u1], axis = -1)
        y = self.last(u7)
        mask = self.activation(y)

        return score, mask



        
