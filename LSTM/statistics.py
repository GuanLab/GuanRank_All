import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv('count.txt', header=None, names=['id', 'length'])
    x = df['length']
    plt.hist(x, bins=50)
    plt.savefig('./length_dist.png')
