import os
import sys
import numpy as np


name_all = ['coad','kirc','lihc']

for the_name in name_all:
    path0='svs_' + the_name + '/'
    os.system('mkdir -p ' + path0)
    
    x=np.loadtxt('gdc_' + the_name + '.txt', dtype='str')
    for i in np.arange(1,len(x)):
        the_id = x[i,0]
        os.system("curl --remote-name --remote-header-name 'https://api.gdc.cancer.gov/data/" + the_id + "'")
    
    os.system('mv TCGA*svs ' + path0)


