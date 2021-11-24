import pandas as pd
import matplotlib.pyplot as plt


def basic():
    data = pd.read_csv('../data/misc/test.csv')
    print(data)
    print('\r\n')
    print(data.head(2))
    print('\r\n')
    print(data.dtypes)
    print('\r\n')
    print(data.describe())


def test1():
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


if __name__ == '__main__':
    basic()
