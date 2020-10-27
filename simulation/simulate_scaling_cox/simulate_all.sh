#!/bin/bash


for delta in 1 2 10 20 100 200 1000 2000
do
        cp -rf base code_${delta}

        cd code_${delta}
        sh bash_all.sh ${delta} &

        cd ../

done
