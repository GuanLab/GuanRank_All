# GuanRank_All
Application examples of GuanRank in deep learning (images), lightGBM and recurrent neural networks (time-series)

# README FOR SIMULATION EXPERIMENT
/simulation
In each of directory, run simulate_all.sh will return all relevant simulation results.

## delta, delta_RSF_fixed, and delta_cox_fixed
The *delta* directory simulates how the death rate scale will affect the performance of the ranking algorithm.
The *delta_RSF_fixed* directory simulates how the death rate scale will affect the performance of the random survival forest algorithm.
The *delta_cox_fixed* directory simulates how the death rate scale will affect the performance of the Cox model.

## simulate_new_feature_example_fixed, simulate_new_feature_example_RSF_fixed, simulate_new_feature_example_cox_fixed
The *simulate_new_feature_example_fixed* directory simulates how the number of features and number of examples will affect the performance of the ranking algorithm.
The *simulate_new_feature_example_RSF_fixed* directory simulates how the number of features and number of examples will affect the performance of the random survival forest algorithm.
The *simulate_new_feature_example_cox_fixed* directory simulates how the number of features and number of examples will affect the performance of the Cox model.
Because two parameters are tested in this section, we create a matrix of performance estimations.

## simulate_new_noise_example_fixed, simulate_new_noise_example_RSF, and simulate_new_noise_example_cox_fixed
The *simulate_new_noise_example_fixed* directory simulates how the noise level and the number of examples will affect the performance of the ranking algorithm.
The *simulate_new_noise_example_RSF* directory simulates how the noise level and the number of examples will affect the performance of the random survival forest algorithm.
The *simulate_new_noise_example_cox_fixed* directory simulates how the noise level and the number of examples will affect the performance of the Cox model.
Because two parameters are tested in this section, we create a matrix of performance estimations.


## simulate_new_noise_example_fixed, simulate_new_noise_example_RSF, simulate_new_noise_example_cox_fixed
The *simulate_new_noise_example_fixed* directory simulates how the noise level and the number of examples will affect the performance of the ranking algorithm.
The *simulate_new_noise_example_RSF* directory simulates how the noise level and the number of examples will affect the performance of the random survival forest algorithm.
The *simulate_new_noise_example_cox_fixed* directory simulates how the noise level and the number of examples will affect the performance of the Cox model.
Because two parameters are tested in this section, we create a matrix of performance estimations.

## simulate_scaling，simulate_scaling_RSF, simulate_scaling_cox
The *simulate_scaling* director simulates how the feature scales will affect the performance of the ranking algorithm.
The *simulate_scaling_RSF* director simulates how the feature scales will affect the performance of the random survival forest algorithm.
The *simulate_scaling_cox* director simulates how the feature scales will affect the performance of the Cox model.

## simulation_new, simulation_new_RSF, simulation_new_cox
The *simulation_new* directory simulates how the noise level and the number of features will affect the performance of the ranking algorithm.
The *simulation_new_RSF* directory simulates how the noise level and the number of features will affect the performance of the random survival forest algorithm.
The *simulation_new_cox* directory simulates how the noise level and the number of features will affect the performance of the Cox model.
Because two parameters are tested in this section, we create a matrix of performance estimations.



# README FOR LSTM 1D time series model

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
sh bash.sh will run the entire cross-validation


## Model Tuning

- Preprocessing:
  - Data removing: remove low quality data and remove non-important features
  - Data imputation: lots of missing data especially at early times
  - Data normalization: normalize features
  - Timesteps: the number of timesteps to be used





