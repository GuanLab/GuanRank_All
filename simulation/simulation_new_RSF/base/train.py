import numpy as np
import sklearn
import sklearn.ensemble
import os
import pickle
import lightgbm as lgb
import sys



train_gs=np.loadtxt('train.dat')[:,0]
train_feature=np.loadtxt('train.dat')[:,1:]

lgb_train = lgb.Dataset(np.asarray(train_feature), np.asarray(train_gs))

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'num_leaves': 50,
    'learning_rate': 0.05,
    'verbose': 0,
    'n_estimators': 200,
    'reg_alpha': 2.0,
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=200
                )


test_feature=np.loadtxt('test.dat')[:,1:]
value=gbm.predict(test_feature)

np.savetxt('prediction.dat',value)






