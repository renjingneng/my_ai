import time
from tqdm import tqdm


def test1():
    bar = tqdm(range(5))
    bar.set_description("start")
    for i, value in enumerate(bar):
        time.sleep(1)
        bar.set_description("finished {} ".format(
            i + 1))


if __name__ == '__main__':
    test1()
