#!/usr/bin/env python
import os
import sys
import logging
import numpy as np
import re
import time
import glob
import unet
import tensorflow as tf
import keras
from keras import backend as K
import cv2
from datetime import datetime
import argparse
print('tf-' + tf.__version__, 'keras-' + keras.__version__)

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.15 
set_session(tf.Session(config=config))

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2.).mean()

def rmse(y_true, y_pred):
    return ((y_true - y_pred) ** 2.).mean() ** 0.5

###### PARAMETER ###############

size=(1024,1024)
batch_size=1

def get_args():
    parser = argparse.ArgumentParser(description="run unet-patch prediction")
    parser.add_argument('-c', '--cancer', default='coad', type=str, help='cancer name')
    parser.add_argument('-f', '--fold', default='0', type=str, help='cross validtion fold')
    parser.add_argument('-s', '--seed', default='0', type=int, help='seed for train-vali partition')
    parser.add_argument('-e', '--epoch', default='10', type=str, help='number of epochs')
    args = parser.parse_args()
    return args

args = get_args()

cancer = args.cancer
fold = args.fold
seed_partition = args.seed
num_epoch = args.epoch

## model
num_class = 1
num_channel = 3
name_model='./epoch' + num_epoch + '/weights_fold' + fold +'_seed' + str(seed_partition) + '.h5'
model = unet.get_unet(the_lr=1e-3,num_class=num_class,num_channel=num_channel,size=size)
model.load_weights(name_model)
#model.summary()

path0='../../data/png_' + cancer + '/'

## an individual has multiple samples
id_all=np.loadtxt('../../data/id_' + cancer + '.txt', dtype='str')
individual_test=np.loadtxt('./partition/individual_test' + fold + '.txt', dtype='str')

print('pre-loading images ...')
dict_image={}
for the_id in id_all:
    print(the_id)
    the_individual = '-'.join(the_id.split('-')[:3])
    if os.path.isfile(path0 + the_id + '.png') and (the_individual in individual_test):
        if the_individual not in dict_image.keys():
            dict_image[the_individual]=[]
        image = cv2.imread(path0 + the_id + '.png')
        height,width = image.shape[:2]
        # gaussian normalization
        the_avg = np.mean(image,axis=(0,1))
        the_std = np.std(image,axis=(0,1))
        image = (image - the_avg.reshape(1,1,3)) / the_std.reshape(1,1,3)
        image = (image * 255).astype('float32')
        if 'DX' in the_id.split('-')[-1]:
            # rescale with the original ratio
            tmp = np.max((height/size[0],width/size[1]))
            height = int(np.floor(height/tmp))
            width = int(np.floor(width/tmp))
            image_new=cv2.resize(image,(width,height))
            dict_image[the_individual].append(image_new)
        else: # cut into two subimages
            width1 = int(width/2)
            width2 = width - width1
            image1 = image[:,:width1,:]
            image2 = image[:,width1:,:]
            # left-subimage
            tmp = np.max((height/size[0],width1/size[1]))
            height = int(np.floor(height/tmp))
            width1 = int(np.floor(width1/tmp))
            image_new=cv2.resize(image1,(width1,height))
            dict_image[the_individual].append(image_new)
            # right-subimage
            tmp = np.max((height/size[0],width2/size[1]))
            height = int(np.floor(height/tmp))
            width2 = int(np.floor(width2/tmp))
            image_new=cv2.resize(image2,(width2,height))
            dict_image[the_individual].append(image_new)

## label and clinical features
mat=np.loadtxt('../../data/feature_tcga_' + cancer + '.tsv',delimiter='\t',dtype='str')

dict_label={}
dict_feature={}
dict_date={}
dict_status={}
for i in np.arange(mat.shape[0]):
    the_individual = mat[i,0]
    if the_individual not in dict_label.keys():
        dict_label[the_individual] = float(mat[i,3])
        dict_feature[the_individual] = np.array(mat[i,4:],dtype='float').reshape(-1,1)
        dict_status[the_individual] = float(mat[i,2])
        dict_date[the_individual] = float(mat[i,1])

####################################

date_all=[]
status_all=[]
pred_all=[]
label_all=[]

print('predicting ...')
for the_individual in individual_test:
    print(the_individual)
    # 0. date & status & label
    date_all.append(dict_date[the_individual])
    status_all.append(dict_status[the_individual])
    label_all.append(dict_label[the_individual])
    # 1. image & feature
    #feature = dict_feature[the_individual]
    image_all = dict_image[the_individual]
    pred = []
    for image in image_all:
        ## padding white background
        height,width = image.shape[:2]
        image_new = np.zeros((size[0], size[1], num_channel)) + 255
        shift1 = int((size[0] - height)/2)
        shift2 = int((size[1] - width)/2)
        image_new[shift1:(height+shift1), shift2:(width+shift2), :] = image
        image = image_new
        ## batch = 1 here
        image_batch = image.reshape(batch_size,size[0],size[1],num_channel)
        #feature_batch = feature.reshape(batch_size, -1, 1)
        #output = model.predict([image_batch, feature_batch])
        output = model.predict(image_batch)
        pred.append(output[0,0])
    pred = np.array(pred)
    pred = np.max(pred)
    pred_all.append(pred)

file_pred=open('./epoch' + num_epoch + '/pred_fold' + fold + '_seed' + str(seed_partition) + '.tsv','w')
file_pred.write('id\ttime\tstatus\tlabel\tpred\n')
for i in np.arange(len(date_all)):
    file_pred.write('%s\t%d\t%d\t%.3f\t%.3f\n' % 
        (individual_test[i],date_all[i],status_all[i],label_all[i],pred_all[i]))

file_pred.close()

## correlation
status_all=np.array(status_all)
label_all=np.array(label_all)
pred_all=np.array(pred_all)

print(np.max(pred_all), np.min(pred_all), np.mean(pred_all))
print('cor-status=%.3f' % np.corrcoef(status_all,pred_all)[0,1])
print('cor-label=%.3f' % np.corrcoef(label_all,pred_all)[0,1])


