#!/bin/bash

python split.py ${1}
perl guanrank.pl
perl prepare_train.pl
python preprocess_test.py

python preprocess.py --seed 1 
python main.py
python predict.py
mv prediction.dat prediction.dat.1

python preprocess.py --seed 2
python main.py
python predict.py
mv prediction.dat prediction.dat.2


python preprocess.py --seed 3
python main.py
python predict.py
mv prediction.dat prediction.dat.3

python preprocess.py --seed 4
python main.py
python predict.py
mv prediction.dat prediction.dat.4


python preprocess.py --seed 5
python main.py
python predict.py
mv prediction.dat prediction.dat.5

perl combine.pl

cut -f 2-3 test_gs.dat >tmp.txt
paste tmp.txt prediction.dat >input.txt

R CMD BATCH test_cv.R

mv cIndex.txt cIndex.txt.${1}

