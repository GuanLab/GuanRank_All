#!/bin/bash

set -e

cancer='coad'

for fold in {0..4}
do
    python lgbm.py -c $cancer -f $fold | tee -a log_fold${fold}.txt &
done
wait

for fold in {0..4}
do
    echo 'fold'${fold}
    Rscript score.r pred_lgbm/pred_fold${fold}_consensus.tsv | tee -a log_cindex.txt
done
