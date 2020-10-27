#!/bin/bash

set -e

cancer='coad'

for seed in {0..5}
do
    for fold in {0..4}
    do
        num=20; dir=epoch${num}; echo $dir; mkdir -p $dir
        python train.py -c $cancer -f $fold -s $seed | tee -a log_fold${fold}_seed${seed}.txt
        cp weights_fold${fold}_seed${seed}.h5 $dir
        python pred.py -c $cancer -f $fold -s $seed -e $num | tee $dir/log_pred_fold${fold}_seed${seed}.txt
        
        for num in {40..60..20}
        do
            dir=epoch${num}; echo $dir; mkdir -p $dir
            sed -e 's/#model.load_weights(name_model)/model.load_weights(name_model)/g; s/the_lr=1e-3/the_lr=1e-4/g; s/model.summary()/#model.summary()/g' train.py > continue_train.py
            python continue_train.py -c $cancer -f $fold -s $seed | tee -a log_fold${fold}_seed${seed}.txt
            cp weights_fold${fold}_seed${seed}.h5 $dir
            python pred.py -c $cancer -f $fold -s $seed -e $num | tee $dir/log_pred_fold${fold}_seed${seed}.txt
        done
    done
done

for fold in {0..4}
do
    for num in {20..60..20}
    do
        dir=epoch${num}; echo $dir; mkdir -p $dir
        python pred_consensus.py -c $cancer -f $fold -e $num | tee $dir/log_pred_fold${fold}_consensus.txt
    done
done



