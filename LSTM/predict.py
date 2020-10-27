import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from tqdm import tqdm

from net import LSTM
torch.backends.cudnn.enabled = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-gpu', default=True, action='store_true', help='Use GPU')
    args = parser.parse_args()
    device = torch.device('cuda')
    batch_size = 32
    model = LSTM(input_size=87, hidden_size=256, output_size=1).to(device)
    model.load_state_dict(torch.load('models/lstm.pth', map_location=device))
    model.eval()

    test_dict = json.load(open('data/test.json')) # use validation for now
    n = len(test_dict)
    outputs = []
    for batch in tqdm(range(0, n, batch_size)):
        start = batch
        end = n if start + batch_size >= n else start + batch_size
        X, y = [], []
        for i in range(start, end):
            sample_id = list(test_dict.keys())[i]
            df = pd.read_csv(os.path.join('data', sample_id), header=None)
            X.append(df.to_numpy())
        X = torch.from_numpy(np.array(X)).float().to(device)
        yy = torch.sigmoid(model(X))
        outputs.append(yy.data.cpu().numpy())
    outputs = np.vstack(outputs).flatten()
    np.savetxt('prediction.dat',outputs)
