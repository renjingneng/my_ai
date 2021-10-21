def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir', type=str, help="the corpus directory")
    parser.add_argument('--model_dir', type=str, default="model_dir", help="the output directory for model files")

    parser.add_argument('--num_epoch', type=int, default=20, help="number of epoch to train")
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0., help='the L2 normalization parameter')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size for training')
    parser.add_argument('--device', type=str, default=None,
                        help='the training device: "cuda:0", "cpu:0". It will be auto-detected by default')
    parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length within training')
    parser.add_argument('--val_split', type=float, default=0.2, help='the split for the validation dataset')
    parser.add_argument('--test_split', type=float, default=0.2, help='the split for the testing dataset')
    parser.add_argument('--recovery', action="store_true", help="continue to train from the saved model in model_dir")
    parser.add_argument('--save_best_val_model', action="store_true",
                        help="save the model whose validation score is smallest")

    parser.add_argument('--embedding_dim', type=int, default=100, help='the dimension of the embedding layer')
    parser.add_argument('--hidden_dim', type=int, default=128, help='the dimension of the RNN hidden state')
    parser.add_argument('--num_rnn_layers', type=int, default=1, help='the number of RNN layers')
    parser.add_argument('--rnn_type', type=str, default="lstm", help='RNN type, choice: "lstm", "gru"')

    args = parser.parse_args()
    print(args)
    # continue your code being executed in cmd mode TODO


main()
