## Required dependencies

* [python3](https://www.python.org) 
* [numpy](http://www.numpy.org/) It comes pre-packaged in Anaconda.
* [tensorflow](https://www.tensorflow.org/) (1.14)
* [keras](https://keras.io/) (2.2.5)
* [lightgbm](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html)

## Data preprocessing and GuanRank calculation (the *data* directory)

- *download_image.py* : download images from gdc portal
- *convert_svs_into_png.py* : convert the format of images from svs into png
- *extract_clinical_feature.py* : extract clincal features and one-hot encode them
- *calculate_guanrank.r* and *guanrank.R* : calculate GuanRank labels

## Machine learning models

- *nn1_code_only* : deep neural network models that use images as input
- *lgbm1_code_only* : tree-based lightGBM models that use clinical features as input



