#!/bin/bash

python split.py ${1}


perl prepare_train.pl

R CMD BATCH glm.R
cut -f 2-3 gs_test.dat >tmp.txt
paste tmp.txt prediction.dat >input.txt

R CMD BATCH test_cv.R

mv cIndex.txt cIndex.txt.${1}

