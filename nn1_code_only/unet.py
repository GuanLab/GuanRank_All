from __future__ import print_function

import os
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose,Lambda,BatchNormalization, Dense, Flatten, Reshape
from keras import losses
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
import tensorflow as tf

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

ss=10

def crossentropy_cut(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f= tf.clip_by_value(y_pred_f, 1e-7, (1. - 1e-7))
    mask=K.greater_equal(y_true_f,-0.5)
    losses = -(y_true_f * K.log(y_pred_f) + (1.0 - y_true_f) * K.log(1.0 - y_pred_f))
    losses = tf.boolean_mask(losses, mask)
    masked_loss = tf.reduce_mean(losses)
    return masked_loss

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    mask=K.cast(K.greater_equal(y_true_f,-0.5),dtype='float32')
    intersection = K.sum(y_true_f * y_pred_f * mask)
    return (2. * intersection + ss) / (K.sum(y_true_f * mask) + K.sum(y_pred_f * mask) + ss)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def mse_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
#    y_pred_f= tf.clip_by_value(y_pred_f, 1e-7, (1. - 1e-7))
    loss=K.mean(K.square(y_true_f - y_pred_f))
    return loss

def abse(y_true, y_pred):
   return K.mean(K.abs(y_pred - y_true))

def pcc(layer_in, num_filter, size_kernel, activation='relu', padding='same'):
    x = MaxPooling2D(pool_size=2)(layer_in)
    x = BatchNormalization()(Conv2D(num_filter,size_kernel,activation=activation,padding=padding)(x))
    x = BatchNormalization()(Conv2D(num_filter,size_kernel,activation=activation,padding=padding)(x))
    return x

def ucc(layer_in1,layer_in2, num_filter, size_kernel, activation='relu', padding='same'):
    x = concatenate([Conv2DTranspose(num_filter,2,strides=2,padding=padding)(layer_in1), layer_in2], axis=3)
    x = BatchNormalization()(Conv2D(num_filter,size_kernel,activation=activation,padding=padding)(x))
    x = BatchNormalization()(Conv2D(num_filter,size_kernel,activation=activation,padding=padding)(x))
    return x

# with concatenation
def pcc2(layer_in1,layer_in2, num_filter, size_kernel, activation='relu', padding='same'):
    x = concatenate([MaxPooling2D(pool_size=2)(layer_in1), layer_in2], axis=3)
    x = BatchNormalization()(Conv2D(num_filter,size_kernel,activation=activation,padding=padding)(x))
    x = BatchNormalization()(Conv2D(num_filter,size_kernel,activation=activation,padding=padding)(x))
    return x

def get_unet(the_lr=1e-3, num_class=15, num_channel=1, size=(1024,1024)):
#    inputs = Input((None, None, num_channel))
    input_image = Input((size[0], size[1], num_channel)) 
#    input_feature = Input((31,1))

    num_blocks=4 # 
    initial_filter=num_channel * 3 #15
    scale_filter=1.5
    size_kernel=3
    activation='relu'
    padding='same'    

    layer_set1=[]
    layer_set2=[]
    layer_set3=[]

    conv0 = BatchNormalization()(Conv2D(initial_filter, size_kernel, \
        activation=activation, padding=padding)(input_image))
    conv0 = BatchNormalization()(Conv2D(initial_filter, size_kernel, \
        activation=activation, padding=padding)(conv0))

    # 0.0-0.5 unet
    layer_set1.append(conv0)
    num=initial_filter
    for i in range(num_blocks):
        num=int(num * scale_filter)
        the_layer=pcc(layer_set1[i], num, size_kernel, activation=activation, padding=padding)
        layer_set1.append(the_layer)

#    # 0.5-1.0 unet
#    layer_set2.append(the_layer)
#    for i in range(num_blocks):
#        num=int(num / scale_filter)
#        the_layer=ucc(layer_set2[i],layer_set1[-(i+2)],num, size_kernel, activation=activation, padding=padding)
#        layer_set2.append(the_layer)
#
#
#    the_layer = BatchNormalization()(Conv2D(num_channel, size_kernel, \
#        activation=activation, padding=padding)(the_layer))
#
#    # if conv_seg be part of the final e2e score
#    conv_seg = Conv2D(2, 1, activation='sigmoid')(the_layer)
#    #conv_seg = Conv2D(2, size_kernel, activation='sigmoid', padding=padding)(the_layer)
#
#    the_layer = BatchNormalization()(Conv2D(num_channel, size_kernel, \
#        activation=activation, padding=padding)(conv_seg))
#
#    # 1.0-1.5 unet
#    layer_set3.append(the_layer)
#    num=initial_filter
#    for i in range(num_blocks):
#        num=int(num * scale_filter)
#        the_layer=pcc2(layer_set3[i], layer_set2[-(i+2)], num, size_kernel, activation=activation, padding=padding)
#        layer_set3.append(the_layer)

    convn=the_layer
    convn = BatchNormalization()(Conv2D(initial_filter * 4, size_kernel, \
        activation=activation, padding=padding)(convn))
    convn = BatchNormalization()(Conv2D(initial_filter * 2, size_kernel, \
        activation=activation, padding=padding)(convn))
    convn = BatchNormalization()(Conv2D(num_channel, size_kernel, \
        activation=activation, padding=padding)(convn))
    convn = BatchNormalization()(Conv2D(1, size_kernel, \
        activation=activation, padding=padding)(convn))

    densen = Dense(31,activation='relu')(Flatten()(convn))
#    densen = Reshape((31,1))(densen)
#    densen = concatenate([densen, Flatten()(input_feature)])
    densen = Dense(1, activation='sigmoid')(densen)

#    model = Model(inputs=[input_image, input_feature], outputs=[densen])
    model = Model(inputs=[input_image], outputs=[densen])

    model.compile(optimizer=Adam(lr=the_lr,beta_1=0.9, beta_2=0.999,decay=1e-5), \
        loss=[losses.mean_squared_error], \
        metrics=[abse])

    return model


