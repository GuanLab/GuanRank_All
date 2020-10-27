import os
import sys
import numpy as np
import lightgbm as lgb
from datetime import datetime
import pickle
import argparse

def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2.).mean()

def rmse(y_true, y_pred):
    return ((y_true - y_pred) ** 2.).mean() ** 0.5

###### PARAMETER ###############
num_boost=500
num_early_stop=20
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'num_leaves': 5,
    'min_data_in_leaf': 3,
    #'learning_rate': 0.05,
    'verbose': 0,
    'lambda_l2': 2.0,
    'bagging_freq': 1,
    'bagging_fraction': 0.7,
}
################################


def get_args():
    parser = argparse.ArgumentParser(description="run unet-patch prediction")
    parser.add_argument('-c', '--cancer', default='coad', type=str, help='cancer name')
    parser.add_argument('-f', '--fold', default='0', type=str, help='cross validtion fold')
#    parser.add_argument('-s', '--seed', default='1', type=int, help='seed for lgbm')
    args = parser.parse_args()
    return args

args = get_args()

cancer = args.cancer
fold = args.fold
#seed_partition = args.seed
num_seed = 10

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

individual_all=np.loadtxt('./partition/individual_all.txt', dtype='str')
individual_test=np.loadtxt('./partition/individual_test' + fold + '.txt', dtype='str')

individual_tv = []
for the_individual in individual_all:
    if the_individual not in individual_test:
        individual_tv.append(the_individual)

#################################

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
        #dict_feature[the_individual] = np.array(mat[i,4:],dtype='float').reshape(-1,1)
        dict_feature[the_individual] = np.array(mat[i,4:],dtype='float')
        dict_date[the_individual] = float(mat[i,1])
        dict_status[the_individual] = float(mat[i,2])

feature_test=[]
label_test=[]
date_test=[]
status_test=[]
for the_individual in individual_test:
    feature_test.append(dict_feature[the_individual])
    label_test.append(dict_label[the_individual])
    date_test.append(dict_date[the_individual])
    status_test.append(dict_status[the_individual])

feature_test=np.array(feature_test) # row sample; column feature
label_test=np.array(label_test)

for seed_partition in np.arange(num_seed):
    np.random.seed(seed_partition) # HERE
    np.random.shuffle(individual_tv)
    ratio=[0.75,0.25]
    num = int(len(individual_tv)*ratio[0])
    individual_train = individual_tv[:num]
    individual_vali = individual_tv[num:]
    
    print('number of train: ', len(individual_train))
    print('number of vali: ', len(individual_vali))
    ####################################
    
    feature_train=[]
    label_train=[]
    for the_individual in individual_train:
        feature_train.append(dict_feature[the_individual])
        label_train.append(dict_label[the_individual])
    
    feature_train=np.array(feature_train) # row sample; column feature
    label_train=np.array(label_train)
    
    feature_vali=[]
    label_vali=[]
    for the_individual in individual_vali:
        feature_vali.append(dict_feature[the_individual])
        label_vali.append(dict_label[the_individual])
    
    feature_vali=np.array(feature_vali) # row sample; column feature
    label_vali=np.array(label_vali)
    
    path_model = './model_lgbm/'
    os.system('mkdir -p ' + path_model)
    data_train=lgb.Dataset(feature_train, label=label_train, free_raw_data=False)
    data_vali=lgb.Dataset(feature_vali, label=label_vali, free_raw_data=False)
    
    gbm = lgb.train(params, data_train, num_boost_round=num_boost, \
            valid_sets=data_vali, early_stopping_rounds=num_early_stop)
    pickle.dump(gbm, open(path_model + 'lgbm_fold' + fold + '_seed' + str(seed_partition) + '.model', 'wb'))
    
    the_pred = gbm.predict(feature_test)
    path_pred = './pred_lgbm/'
    os.system('mkdir -p ' + path_pred)
    np.save(path_pred + 'pred_fold' + fold + '_seed' + str(seed_partition) + '.model', the_pred)
    
    if seed_partition == 0 :
        pred_consensus = the_pred.copy()
    else:
        pred_consensus += the_pred

pred_consensus = pred_consensus / float(num_seed)

file_pred=open(path_pred + '/pred_fold' + fold + '_consensus.tsv','w')
file_pred.write('id\ttime\tstatus\tlabel\tpred\n')
for i in np.arange(len(individual_test)):
    file_pred.write('%s\t%d\t%d\t%.3f\t%.3f\n' %
        (individual_test[i],date_test[i],status_test[i],label_test[i],pred_consensus[i]))

file_pred.close()    
    
status_test=np.array(status_test)
print(np.max(pred_consensus), np.min(pred_consensus), np.mean(pred_consensus))
print('cor-status=%.3f' % np.corrcoef(status_test,pred_consensus)[0,1])
print('cor-label=%.3f' % np.corrcoef(label_test,pred_consensus)[0,1])



 
