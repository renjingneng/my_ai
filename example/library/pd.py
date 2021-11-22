import numpy as np
import pandas


def basic():
    data = pandas.read_csv('../data/misc/test.csv')
    # print(data)
    print(data.dtypes)
    print(data.describe())
    print(data.sort_index())


if __name__ == '__main__':
    basic()
