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
    parser.add_argument("--src-file", help="Source file from where to read text.")
    parser.add_argument("--tgt-file", help="Target file where to write text.")
    return parser.parse_args()


def process_file(src_file, tgt_file):
    with open(src_file, 'r', encoding='utf-8') as f:
        n_tokens = [len(line.split(' ')) for line in f]
    n_tokens = np.array(n_tokens)

    df = pd.DataFrame([[sum(n_tokens)]], columns=['n_tokens'])
    df.to_csv(tgt_file, sep='\t')


def main():
    args = get_args()
    process_file(args.src_file, args.tgt_file)


if __name__ == "__main__":
    main()