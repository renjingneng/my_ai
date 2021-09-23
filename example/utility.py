import sys
import os
import json
import time

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import my_ai


def test1():
    from tqdm import tqdm
    bar = tqdm(range(5))
    bar.set_description("start")
    for i, value in enumerate(bar):
        time.sleep(1)
        bar.set_description("this is{} ".format(
            i+1))

def test2():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir', type=str, help="the corpus directory")
    parser.add_argument('--recovery', action="store_true", help="continue to train from the saved model in model_dir")
    parser.add_argument('--num_epoch', type=int, default=20, help="number of epoch to train")
    args = parser.parse_args()
    print(args)
    print(vars(args))

def test3():
    # save file
    losses = [[1, 1, 10, 7],[1, 2, 8, 5],[2, 1, 7, 4],[2, 2, 2, 1]]
    pd.DataFrame(data=losses, columns=["epoch", "batch", "train_loss", "val_loss"]).to_csv('./loss.csv', index=False)
    # load file and show it
    df = pd.read_csv("./loss.csv")
    """ffill means forward fill
    >>>df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                   [3, 4, np.nan, 1],
                   [np.nan, np.nan, np.nan, 5],
                   [np.nan, 3, np.nan, 4]],
                  columns=list("ABCD"))
       print(df.fillna(method="ffill"))
    ...     A    B   C  D
        0  NaN  2.0 NaN  0
        1  3.0  4.0 NaN  1
        2  3.0  4.0 NaN  5
        3  3.0  3.0 NaN  4        
    """
    df[["train_loss", "val_loss"]].ffill().plot(grid=True)
    plt.show()





def run():
    test3()


if __name__ == '__main__':
    run()