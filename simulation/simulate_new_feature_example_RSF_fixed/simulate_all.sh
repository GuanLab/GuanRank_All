#!/bin/bash


for numberexample in 100 500 1000 5000 10000
do
    for feature in 5 10 50 100 500 1000
    do
        cp -rf base code_${numberexample}_${feature}

        cd code_${numberexample}_${feature}
        sh bash_all.sh ${numberexample} ${feature} &

        cd ../
    done

done
