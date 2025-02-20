import argparse


def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', default=1e-4, type=float, help="learning rate")
    parser.add_argument('-hidden_dim', default=256, type=int, help="embedding dimension")
    parser.add_argument('-weight_decay', default=1e-6, type=float, help="l2 regularization parameter")
    parser.add_argument('-epoch', default=5000, type=int, help="training epoch")
    parser.add_argument('-noise_ratio', default=0, type=int, help="noise ratio")
    parser.add_argument('-model_type', default="GCN", type=str, help="model type")
    parser.add_argument('-dataset', default="Office", type=str, help="dataset")
    parser.add_argument('-cleaned', default=False, type=bool, help="if is cleaned")
    parser.add_argument('-feature', default=1, type=int, help="feature num")
    parser.add_argument('-soft', default=True, type=bool, help="soft label")
    args = parser.parse_args()
    return args

