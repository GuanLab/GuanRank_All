#!/usr/bin/env python
import os
import sys
import numpy as np
import pandas as pd
import re
import time
import glob
import cv2
from datetime import datetime
import argparse

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2.).mean()

def rmse(y_true, y_pred):
    return ((y_true - y_pred) ** 2.).mean() ** 0.5

###### PARAMETER ###############

def get_args():
    parser = argparse.ArgumentParser(description="run unet-patch prediction")
    parser.add_argument('-c', '--cancer', default='coad', type=str, help='cancer name')
    parser.add_argument('-f', '--fold', default='0', type=str, help='cross validtion fold')
#    parser.add_argument('-s', '--seed', default='0', type=int, help='seed for train-vali partition')
    parser.add_argument('-e', '--epoch', default='10', type=str, help='number of epochs')
    args = parser.parse_args()
    return args

args = get_args()

cancer = args.cancer
fold = args.fold
#seed_partition = args.seed
num_epoch = args.epoch

num_seed = 6

for i in np.arange(num_seed):
    pred = pd.read_csv('./epoch' + num_epoch + '/pred_fold' + fold + '_seed' + str(i) + '.tsv', sep='\t')
    if i == 0:
        pred_consensus = pred.copy()
    else:
        pred_consensus.pred += pred.pred

pred_consensus.pred = pred_consensus.pred / float(num_seed)

pred_consensus.to_csv('./epoch' + num_epoch + '/pred_fold' + fold + '_consensus.tsv', sep='\t', float_format='%.3f')

print(np.max(pred_consensus.pred), np.min(pred_consensus.pred), np.mean(pred_consensus.pred))
print('cor-status=%.3f' % np.corrcoef(pred_consensus.status,pred_consensus.pred)[0,1])
print('cor-label=%.3f' % np.corrcoef(pred_consensus.label,pred_consensus.pred)[0,1])


