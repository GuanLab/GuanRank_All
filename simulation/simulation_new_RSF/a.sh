#!/bin/bash


for noisefactor in 100
do
    for feature in 10
    do
        cp -rf base code_${noisefactor}_${feature}

        cd code_${noisefactor}_${feature}
        sh bash_all.sh ${noisefactor} ${feature} &

        cd ../
    done

done
