import sys
import os
import json
import time
import math

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
            i + 1))


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
    losses = [[1, 1, 10, 7], [1, 2, 8, 5], [2, 1, 7, 4], [2, 2, 2, 1]]
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

def test4():
    temp1 =  torch.tensor([1,2,3,4],dtype=torch.float)
    print(temp1.exp().sum(-1).log())
    print(my_ai.utility.log_sum_exp(temp1))

def test5():
    from collections import defaultdict, OrderedDict
    def prepare_sequence(seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    with open('./res/misc/doccano_export.json', 'r') as f:
        lines = f.readlines()

    # The numerical doccano label to actual label (B-I-O scheme)
    ix_to_label = {4: 'O', 3: 'I', 2: 'B'}

    # train/test data
    data = []

    # Vocabulary
    vocab = set()

    # Loop over each data point (a corpus of labeled text) to extract words
    for line in lines:
        # An ordered dict will keep items in order for further manipulation
        # so we initialize here
        orddict = OrderedDict({})
        # Lists to hold the words and labels
        words = []
        labels = []
        # Convert line to json
        injson = json.loads(line)

        annots = injson['annotations']
        text = injson['text']

        # Add each word annotation to OrderedDict
        for ann in annots:
            orddict[ann['start_offset']] = ann

        # Sort ordered dict because there's no guarantee reading json
        # maintained order
        orddict = sorted(orddict.items(), key=lambda x: x[1]['start_offset'])

        for item in orddict:
            # the item is a tuple where second value is the actual value we want
            ann = item[1]
            # Subset text string
            word = text[ann['start_offset']:(ann['end_offset'] + 1)].rstrip()
            label = ix_to_label[ann['label']]
            # Add to list for this datum/corpus
            words.append(word)
            labels.append(label)
            vocab.add(word)
        # Add to overall data containers
        data.append((words, labels))


    num_train = math.floor(len(data) * 0.8) # 80% to train
    training_data, test_data = data[:num_train], data[num_train:]
    # Create a lookup dict for all possible words and record their index
    word_to_ix = {k: v for (k, v) in zip(vocab, range(len(vocab)))}
    tag_to_ix = {"B": 0, "I": 1, "O": 2, "<START>": 3, "<STOP>": 4}
    ix_to_tag = {0: "B", 1: "I", 2: "O"}

    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(precheck_sent)


def run():
    test5()



if __name__ == '__main__':
    run()
