#!/bin/bash


for noisefactor in 0.01 0.05 0.1 0.5 1 5 10 50 100
do
    for numberexample in 100 500 1000 5000 10000
   # for numberexample in 50000 100000
    do
        cp -rf base code_${noisefactor}_${numberexample}

        cd code_${noisefactor}_${numberexample}
        sh bash_all.sh ${noisefactor} ${numberexample} &

        cd ../
    done

done
