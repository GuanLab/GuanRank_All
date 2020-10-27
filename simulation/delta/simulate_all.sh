#!/bin/bash


for delta in 0.0001 0.0002 0.0003 0.0004
do
        cp -rf base code_${delta}

        cd code_${delta}
        sh bash_all.sh ${delta} &
        cd ../

done
