import random
import os
random.seed(1)
REF=open('../../preprocess/status_cut.txt','r')
TRAIN=open('train_gs.dat','w')
TEST=open('test_gs.dat','w')
for line in REF:
    rrr=random.random()
    if (rrr<0.8):
        TRAIN.write(line)
        table=line.split('\t')
        os.system('cp ../../preprocess/individual_cut/'+table[0]+' tmp_data/')

    else:
        TEST.write(line)
REF.close()
TRAIN.close()
TEST.close()






