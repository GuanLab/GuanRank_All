#!/bin/bash


for noisefactor in 0.01 0.05 0.1 0.5 1 5 10 50 100
do
    for feature in 2 5 10 20 50 100
    do
        cp -rf base code_${noisefactor}_${feature}

        cd code_${noisefactor}_${feature}
        sh bash_all.sh ${noisefactor} ${feature} &

        cd ../
    done

done
