import argparse
import os
import pickle
import warnings

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings('ignore', category=DataConversionWarning)


def fit_scaler(X_train):
    scaler = StandardScaler()
    scaler.fit(np.vstack(X_train))
    with open('./models/standard_scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
    return scaler


def transform(X, sample_ids, scaler, num_features, timesteps, output_dir):
    for i, sample_id in tqdm(enumerate(sample_ids)):
        x = X[i]
        try:
            df = pd.DataFrame(scaler.transform(x)) # apply scaler
        except:
            df=pd.DataFrame(scaler.transform((np.zeros(87)).reshape(1, -1)))
            pass
        df = df.fillna(0) # impute with 0s
        df_zeros = pd.DataFrame(np.zeros((timesteps, num_features)), columns=range(num_features))
        df = pd.concat([df_zeros, df]) # padding and use last 100 rows
        df[-timesteps:].to_csv(os.path.join(output_dir, sample_id), index=None, header=None)


def read_data(input_dir, sample_ids, num_features):
    X = []
    for sample_id in tqdm(sample_ids):
        x = pd.read_csv(os.path.join(input_dir, sample_id), sep='\t', header=None, names=range(num_features))
        x.index = [sample_id] * len(x)
        X.append(x)
    return X


def describe(X):
    return pd.concat(X).describe().T


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument('-i', '--input-dir', type=str, default='tmp_data', help='Input directory')
    parser.add_argument('-i', '--input-dir', type=str, default='../../preprocess/individual_cut/', help='Input directory')
    parser.add_argument('-o', '--output-dir', type=str, default='data', help='Output directory')
    parser.add_argument('--num-features', type=int, default=87, help='Number of features')
    parser.add_argument('--timesteps', type=int, default=100, help='Number of timesteps to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    input_dir, output_dir = args.input_dir, args.output_dir
    num_features, timesteps = args.num_features, args.timesteps

    train_ids = pd.read_csv('train_gs.dat', sep='\t', header=None).sample(n=50000, random_state=args.seed)[0]
    train_ids, val_ids = train_test_split(train_ids, test_size=0.2, random_state=args.seed)
    test_ids = pd.read_csv('test_gs.dat', sep='\t', header=None)[0]

    df_scores = pd.read_csv('train_gs.dat', sep='\t', header=None, index_col=0)[2]
    df_test_scores = pd.read_csv('test_target.txt', sep='\t', header=None, index_col=0)[1]
    
    X_train = read_data(input_dir, train_ids, num_features)
    X_val = read_data(input_dir, val_ids, num_features)
    print(len(X_val))
    print(len(X_val[0]))
    print(len(val_ids))
    print(len(X_train))
    print(len(X_train[0]))
    print(len(train_ids))
    X_test = read_data(input_dir, test_ids, num_features)

    scaler = fit_scaler(X_train)
    transform(X_train, train_ids, scaler, num_features, timesteps, output_dir)
    transform(X_test, test_ids, scaler, num_features, timesteps, output_dir)
    transform(X_val, val_ids, scaler, num_features, timesteps, output_dir)

    # save score dicts
    df_scores.reindex(train_ids).to_json(os.path.join(output_dir, 'train.json'))
    df_scores.reindex(val_ids).to_json(os.path.join(output_dir, 'val.json'))
    df_test_scores.reindex(test_ids).to_json(os.path.join(output_dir, 'test.json'))
