# README

Deep learning for survival analysis using PyTorch.

## Structure

- main.py: Code for main entry
- net.py: Code for LSTM model
- preprocess.py: Code for preprocessing
- dataset.py: Dataset for disk survival analysis

## Requirements

- python 3.6
- pytorch 1.3
- tqdm 4.43
- pandas 0.25
- numpy 1.16
- sci-kit learn 0.21
- scikit-survival 0.6

## Run

```
$ SEED=42 && echo ${SEED}
$ python preprocess.py --seed ${SEED} # preprocess and use 10k for training
$ python main.py --use-gpu --epochs 20 --weighted # use weighted loss
$ python main.py --use-gpu --phase test # evaluate on test set
$ python predict.py --use-gpu # save predictions
$ cut -f 2-3 test_gs.dat > tmp.txt
$ paste tmp.txt prediction.dat > input_${SEED}.txt
$ python cIndex.py --seed ${SEED} # R CMD BATCH test_cv.R && mv cIndex.txt cIndex.txt.${SEED}
```

## Model Tuning

- Preprocessing:
  - Data removing: remove low quality data and remove non-important features
  - Data imputation: lots of missing data especially at early times
  - Data normalization: normalize features
  - Timesteps: the number of timesteps to be used
- Model:
  - Tuning: deeper?
