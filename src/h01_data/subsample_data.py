import sys
import os
import argparse
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-file", help="Source file from where to read text.")
    parser.add_argument("--tgt-file", help="Target file where to write text.")
    parser.add_argument("--max-tokens", type=int, help="Max number of tokens kept.")
    return parser.parse_args()


def process_file(src_file, tgt_file, max_tokens):

    with open(src_file, 'r', encoding='utf-8') as f:
        n_tokens = [len(line.split(' ')) for line in f]
    n_tokens = np.array(n_tokens)

    print(f'# Tokens in dataset ${n_tokens.sum()}. Subsampled to {max_tokens}. Ratio {max_tokens / n_tokens.sum()}')

    if max_tokens == -1:
        keep_ids = set(np.arange(len(n_tokens)))
    else:
        sentence_ids = np.arange(len(n_tokens))
        np.random.shuffle(sentence_ids)
        n_sentences = (n_tokens[sentence_ids].cumsum() < max_tokens).sum() + 1
        keep_ids = set(sentence_ids[:n_sentences])

    with open(src_file, 'r', encoding='utf-8') as f_in:
        with open(tgt_file, 'w', encoding='utf-8') as f_out:
            for sentence_id, line in enumerate(f_in):
                if sentence_id in keep_ids:
                    f_out.write(line)


def main():
    args = get_args()
    process_file(args.src_file, args.tgt_file, args.max_tokens)


if __name__ == "__main__":
    main()