import os
import sys
import numpy as np
import re
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose,Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
print('tf-' + tf.__version__, 'keras-' + keras.__version__)
from datetime import datetime
import unet
import cv2
import argparse

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.29 
set_session(tf.Session(config=config))

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

size=(1024,1024)
size_shift = 10
batch_size=10

def get_args():
    parser = argparse.ArgumentParser(description="run unet-patch prediction")
    parser.add_argument('-c', '--cancer', default='coad', type=str, help='cancer name')
    parser.add_argument('-f', '--fold', default='0', type=str, help='cross validtion fold')
    parser.add_argument('-s', '--seed', default='0', type=int, help='seed for train-vali partition')
    args = parser.parse_args()
    return args

args = get_args()

cancer = args.cancer
fold = args.fold
seed_partition = args.seed

## model
num_class = 1
num_channel = 3
name_model='weights_fold' + fold +'_seed' + str(seed_partition) + '.h5'
model = unet.get_unet(the_lr=1e-3,num_class=num_class,num_channel=num_channel,size=size)
#model.load_weights(name_model)
model.summary()

path0='../../data/png_' + cancer + '/'

#individual_all=np.loadtxt('../../data/individual_' + cancer + '.txt', dtype='str')
#individual_all.sort()
#print(individual_all)
#np.savetxt('./partition/individual_all.txt', individual_all, fmt='%s')
#np.random.seed(449) # 5-fold
#np.random.shuffle(individual_all)
#num = int(np.ceil(len(individual_all)*0.2))
#for i in np.arange(5):
#    start = i * num
#    end = (i+1) * num
#    if end > len(individual_all):
#        end = len(individual_all)
#    individual_test = individual_all[start:end]
#    individual_test.sort()
#    np.savetxt('./partition/individual_test' + str(i) + '.txt', individual_test, fmt='%s')

## an individual has multiple samples
id_all=np.loadtxt('../../data/id_' + cancer + '.txt', dtype='str')

individual_all=np.loadtxt('./partition/individual_all.txt', dtype='str')
individual_test=np.loadtxt('./partition/individual_test' + fold + '.txt', dtype='str')
individual_tv = []
for the_individual in individual_all:
    if the_individual not in individual_test:
        individual_tv.append(the_individual)

print('pre-loading images ...')
dict_image={}
for the_id in id_all:
#    print(the_id)
    the_individual = '-'.join(the_id.split('-')[:3])
    if os.path.isfile(path0 + the_id + '.png') and (the_individual in individual_tv):
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

#################################

## label and clinical features
mat=np.loadtxt('../../data/feature_tcga_' + cancer + '.tsv',delimiter='\t',dtype='str')

dict_label={}
dict_feature={}
dict_status={}
for i in np.arange(mat.shape[0]):
    the_individual = mat[i,0]
    if the_individual not in dict_label.keys():
        dict_label[the_individual] = float(mat[i,3])
        dict_feature[the_individual] = np.array(mat[i,4:],dtype='float').reshape(-1,1)
        dict_status[the_individual] = float(mat[i,2])

## id partition ##################
np.random.seed(seed_partition) # HERE
np.random.shuffle(individual_tv)
ratio=[0.75,0.25]
num = int(len(individual_tv)*ratio[0])
individual_train = individual_tv[:num]
individual_vali = individual_tv[num:]

individual_train.sort()
individual_vali.sort()
print('individual_train: ', len(individual_train), individual_train)
print('individual_vali : ', len(individual_vali), individual_vali)

np.random.seed(datetime.now().microsecond)
np.random.shuffle(individual_train)
np.random.shuffle(individual_vali)

####################################

## oversampling ###################
train_pos=[]
train_neg=[]
for the_individual in individual_train:
    if dict_status[the_individual] == 1:
        train_pos.append(the_individual)
    else:
        train_neg.append(the_individual)
 
train_pos = np.array(train_pos)
train_neg = np.array(train_neg)

if len(train_neg) > len(train_pos):
    num_diff = len(train_neg) - len(train_pos)
    index = np.random.randint(0, len(train_pos), num_diff)
    train_all = np.concatenate((train_neg, train_pos, train_pos[index]))
else:
    num_diff = len(train_pos) - len(train_neg)
    index = np.random.randint(0, len(train_neg), num_diff)
    train_all = np.concatenate((train_neg, train_pos, train_neg[index]))

vali_pos=[]
vali_neg=[]
for the_individual in individual_train:
    if dict_status[the_individual] == 1:
        vali_pos.append(the_individual)
    else:
        vali_neg.append(the_individual)

vali_pos = np.array(vali_pos)
vali_neg = np.array(vali_neg)

if len(vali_neg) > len(vali_pos):
    num_diff = len(vali_neg) - len(vali_pos)
    index = np.random.randint(0, len(vali_pos), num_diff)
    vali_all = np.concatenate((vali_neg, vali_pos, vali_pos[index]))
else:
    num_diff = len(vali_pos) - len(vali_neg)
    index = np.random.randint(0, len(vali_neg), num_diff)
    vali_all = np.concatenate((vali_neg, vali_pos, vali_neg[index]))

vali_all = np.concatenate((vali_neg, vali_pos))

# shuffle
np.random.seed(datetime.now().microsecond)
np.random.shuffle(train_all)
np.random.shuffle(vali_all)

# number of pos/neg
print('number of pos: ', len(train_pos))
print('number of neg: ', len(train_neg))
print('number of train: ', len(train_all))

## augmentation parameters ######
if_time=False
max_scale=1.15
min_scale=1
if_mag=True
#if_mag=False
max_mag=1.15
min_mag=0.9
if_flip=True
####################################

def generate_data(ids, batch_size, if_train):

    i=0
    while True:
        image_batch = []
        #feature_batch = []
        label_batch = []
        for b in np.arange(batch_size):
            if i == len(ids):
                i=0
                np.random.shuffle(ids)

            the_individual = ids[i]
            i += 1

            # 0. label
            label = dict_label[the_individual]

            # 1. image 
            image_all=dict_image[the_individual]
            # randomly use one slide
            image=image_all[np.random.choice(len(image_all))]

            ## padding white background
            height,width = image.shape[:2]
            image_new = np.zeros((size[0], size[1], num_channel)) + 255 
            shift1 = int((size[0] - height)/2)
            shift2 = int((size[1] - width)/2)
            image_new[shift1:(height+shift1), shift2:(width+shift2), :] = image
            image = image_new

            if (if_train==1):
                if if_mag:
                    the_mag = min_mag + (max_mag - min_mag) * np.random.uniform(0,1,1)[0]
                    image = image * the_mag
                if if_flip & np.random.randint(2) == 1:
                    # 0 vertically; 1 horizontally; -1 both
                    the_seed = np.random.randint(-1,2,1)[0]
                    image = cv2.flip(image, the_seed)

            # 2. feature
            #feature=dict_feature[the_individual]

            #print(image.shape,feature.shape,label)
            image_batch.append(image)
            #feature_batch.append(feature)
            label_batch.append(label)

        image_batch=np.array(image_batch)
        #feature_batch=np.array(feature_batch)
        label_batch=np.array(label_batch)
        #print(image_batch.shape, feature_batch.shape, label_batch.shape)
        #yield [image_batch,feature_batch], label_batch
        yield image_batch, label_batch

callbacks = [
    keras.callbacks.ModelCheckpoint(os.path.join('./', name_model),
    save_weights_only=False,monitor='val_loss')
    ]

model.fit_generator(
    generate_data(train_all, batch_size,True),
    steps_per_epoch=int(len(train_all) // (batch_size)), nb_epoch=20,
#    steps_per_epoch=10, nb_epoch=1,
    validation_data=generate_data(vali_all, batch_size,False),
    validation_steps=int(len(vali_all) // (batch_size)),
#    validation_steps=10,
    callbacks=callbacks,verbose=1)


