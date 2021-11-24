import argparse


def test1():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir', type=str, help="the corpus directory")
    parser.add_argument('--recovery', action="store_true", help="continue to train from the saved model in model_dir")
    parser.add_argument('--num_epoch', type=int, default=20, help="number of epoch to train")
    args = parser.parse_args()
    print(args)
    print(vars(args))


if __name__ == '__main__':
    test1()
