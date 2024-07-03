import sys
import os
import re
import argparse
import pandas as pd
import numpy as np


sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", help="Train file from where to read text.")
    parser.add_argument("--val-file", help="Val file from where to read text.")
    parser.add_argument("--test-file", help="Test file from where to read text.")
    parser.add_argument("--raw-test-file", help="Test file from where to read text.")
    parser.add_argument("--surprisals-file", help="Test file from where to read text.")
    parser.add_argument("--tgt-file", help="Target file where to write text.")
    parser.add_argument("--per-word", action='store_true', help="Count words instead of tokens.")
    return parser.parse_args()


def read_surprisals(src_fname):
    df = pd.read_csv(src_fname, sep='\t', keep_default_na=False)
    del df['Unnamed: 0']

    return df


def get_xent(src_fname):
    df = read_surprisals(src_fname)

    xent = df.score.mean()
    surp_tokens, surp_words = df.n_subtokens.sum(), df.shape[0]
    return xent, surp_tokens, surp_words


def process_file(src_file):
    n_tokens, n_chars = 0, 0
    with open(src_file, 'r', encoding='utf-8') as f:
        for line in f:
            n_tokens += len(line.split(' '))
            n_chars += len(line)

    return (n_tokens, n_chars)


def process_files(train_file, val_file, test_file, raw_test_file, surprisals_file, tgt_file, per_word):
    xent, surp_tokens, surp_words = get_xent(surprisals_file)

    train_tokens, _ = process_file(train_file)
    val_tokens, _ = process_file(val_file)
    test_tokens, _ = process_file(test_file)
    test_words, test_chars = process_file(raw_test_file)
    bpc = xent * surp_words / test_chars

    cols = ['xent', 'bpc', 'surp_tokens', 'surp_words',
            'train_tokens', 'val_tokens', 'test_tokens',
            'test_words', 'test_chars']
    values = [xent, bpc, surp_tokens, surp_words,
              train_tokens, val_tokens, test_tokens,
              test_words, test_chars]
              
    df = pd.DataFrame([values], columns=cols)
    df.to_csv(tgt_file, sep='\t')


def main():
    args = get_args()
    process_files(args.train_file, args.val_file, args.test_file, 
                  args.raw_test_file, args.surprisals_file,
                  args.tgt_file, args.per_word)


if __name__ == "__main__":
    main()