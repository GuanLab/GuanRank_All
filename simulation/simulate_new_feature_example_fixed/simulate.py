import numpy  as np
import random
import sys

random.seed(42)
np.random.seed(42)

number_example=int(sys.argv[1])
noise_factor=0.1
death_rate_1=0.002
death_rate_2=0.002
sd_1=0.001
sd_2=0.001

### create death rate per day a random value between [0,0.001]
death_rate=[]
i=0
while (i<number_example):
    rrr1=np.random.normal(loc=death_rate_1, scale=sd_1, size=1)
    rrr2=np.random.normal(loc=death_rate_2, scale=sd_2, size=1)
    rrr=random.random()
    if (rrr>0.5):
        r=rrr1
    else:
        r=rrr2
    if r<0.0000001:
        r=0.0000001
    death_rate.append(r)
    i=i+1

#### create a list of censoring date between [0,2000]
total_date=1000
censor_date=[]
i=0
while (i<number_example):
    r=random.random()*total_date
    censor_date.append(r)
    i=i+1

### create a list of last observation date.
observation_date=[]
death_status=[]
i=0
while (i<number_example):
    last_date=censor_date[i]
    vector=np.random.binomial(1, death_rate[i], size=total_date)
    date_i=0
    while (date_i<total_date):
        if (vector[date_i]==1):
            last_date=date_i
            break
        date_i=date_i+1
    if (last_date == censor_date[i]):
        death_status.append(0)
    else:
        death_status.append(1)
    last_date=last_date+1
    observation_date.append(last_date)
    i=i+1


np.savetxt('death_status.txt',np.asarray(death_status))
np.savetxt('observation_date.txt',np.asarray(observation_date))


feature_num=int(sys.argv[2])
feature_i=0
feature_vector_all=[]
while (feature_i<feature_num):
    feature_scale=random.random()*100
    feature_vector=[]
    i=0
    while (i<number_example):
        feature=death_rate[i]*feature_scale+(random.random()-0.5)*noise_factor*feature_scale*death_rate[i]
        feature_vector.append(feature)
        i=i+1
    feature_vector=np.asarray(feature_vector)
    feature_vector_all.append(feature_vector)
    feature_i=feature_i+1

np.savetxt('feature.txt',np.asarray(feature_vector_all).T)

