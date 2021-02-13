import numpy as np
from sklearn.preprocessing import OneHotEncoder


x=np.loadtxt('clinical_tcga_brca.tsv', dtype='str', delimiter='\t')

for i in np.arange(x.shape[1]):
    print(i)
    print(np.unique(x[:,i],return_counts=True))

## potentially useful columns:
## 'days_to_last_follow_up', 'days_to_death', 'vital_status'
## 'age_at_index', 'ethnicity', 'gender', 'race', 
## 'ajcc_pathologic_m', 'ajcc_pathologic_n', 'ajcc_pathologic_stage', 'ajcc_pathologic_t'
## 'primary_diagnosis', 'tumor_stage', 'treatment_or_therapy'

cancer_all = ['coad','kirc','lihc']


for cancer in cancer_all:
    x=np.loadtxt('clinical_tcga_' + cancer + '.tsv', dtype='str', delimiter='\t')
#    for i in np.arange(x.shape[1]):
#        print(i)
#        print(np.unique(x[:,i],return_counts=True))
    # double check
    dat1 = x[:, np.array([1,9,47,15,3,11,14,127,107])]
    dat1 = dat1[np.arange(1,x.shape[0],2), :]
    dat2 = x[:, np.array([1,9,47,15,3,11,14,127,107])]
    dat2 = dat2[np.arange(2,x.shape[0],2), :]
    np.sum(dat1 != dat2) #0 means identical
    # subset
    dat = x[:, np.array([1,9,47,15,3,11,14,127,107])]
    dat = dat[np.arange(0,x.shape[0],2), :]
    np.savetxt('clinical_tcga_' + cancer + '_short.tsv', dat, fmt='%s', delimiter='\t')
    # time & status
    dat = dat[1:,:]
    # exclude label-missing cases
    index = (dat[:,3] == 'Alive') | (dat[:,3] == 'Dead')
    dat = dat[index,:]
    dat_new = dat[:,[0,1,3,4]]
    index = dat[:,3] == 'Alive'
    dat_new[index,1] = dat[index,2]
    dat_new[~index,1] = dat[~index,1]
    dat_new[index,2] = '0' 
    dat_new[~index,2] = '1' 
    # one-hot
    enc = OneHotEncoder(handle_unknown='error')
    enc.fit(dat[:,5:])
    tmp = enc.transform(dat[:,5:]).toarray()
    dat_new = np.hstack((dat_new, tmp))
    # save
    np.savetxt('feature_tcga_' + cancer + '_short.tsv', dat_new, fmt='%s', delimiter='\t')






