#!/bin/bash

set -e

#fold=$1
#seed=4

for fold in {0..4}
do
    echo 'fold'${fold}
    for num in {20..60..20}
    do
        echo 'epoch'$num
        dir=epoch${num}
        #Rscript score.r ${dir}/pred_fold${fold}_seed${seed}.tsv
        Rscript score.r ${dir}/pred_fold${fold}_consensus.tsv
    done
done





