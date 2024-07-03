import sys
import os
import re
import argparse
import math
from collections import defaultdict
from collections import Counter
import pandas as pd
import numpy as np
from tqdm import tqdm


sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import util


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", help="Train file from where to read text.")
    parser.add_argument("--val-file", help="Val file from where to read text.")
    parser.add_argument("--test-file", help="Test file from where to read text.")
    parser.add_argument("--tgt-file", help="Target file where to write text.")
    parser.add_argument("--per-word", action='store_true', help="Count words instead of tokens.")
    parser.add_argument("--keep-test-only", action='store_true', help="Keep only words in testset.")
    parser.add_argument("--tokenizer", required=True, help="Used tokenizer in src file.")
    return parser.parse_args()


def get_iterator(src_file):
    with open(src_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Getting token frequencies', mininterval=1):
            if line.strip() == '':
                continue

            yield line


def split_line(line, per_word, tokenizer):
    if per_word and tokenizer == 'unigramlm':
        line = line[1:] if line[0] == '▁' else line
        line_tokens = line.split(' ▁')
    else:
        line_tokens = line.split(' ')

    return line_tokens


def count_tokens(src_file, per_word, tokenizer):
    counts = {}
    for line in get_iterator(src_file):
        line_tokens = split_line(line, per_word, tokenizer)

        for token in line_tokens:
            word = token.replace(' ', '')
            counts[word] = counts.get(word, 0) + 1

    return counts


def process_tokens(src_file, per_word, tokenizer):
    results = defaultdict(lambda : {'count': 0, 'subwords': set()})
    for line in get_iterator(src_file):
        line_tokens = split_line(line, per_word, tokenizer)

        for token in line_tokens:
            word = token.replace(' ', '')

            if per_word and tokenizer == 'unigramlm':
                subwords = [x for x in ('▁' + token).split(' ') if x]
                subwords = tuple(subwords)
            else:
                subwords = (token,)
            # subtokens[word] = subtokens.get(word, set()) | set([subwords])

            results[word]['count'] += 1
            results[word]['subwords'] |= set([subwords])

    return results


# def prob_perword(train_file, test_file, per_word, tokenizer):
def prob_pertoken(train_file, val_file, test_file):
    counts_train = count_tokens(train_file, False, None)
    counts_val = count_tokens(val_file, False, None)
    counts_test = count_tokens(test_file, False, None)
    types = set(counts_train.keys()) | set(counts_val.keys()) | set(counts_test.keys())
    # counts_smoothed = {word: - math.log(count / freqtot) for word in types}

    freqtot = sum(counts_train.values()) + len(types)
    logprobs_token = {token: - math.log((counts_train.get(token, 0) + 1) / freqtot) for token in types}

    return logprobs_token

    # logprobs_word = set()

    # for line in get_iterator(test_file):
    #     line_tokens = split_line(line, per_word, tokenizer)

    #     for tokens in line_tokens:
    #         if tokenizer == 'unigramlm':
    #             word = tokens.replace(' ', '')
    #             tokens = '▁' + tokens
    #             logprob = sum([logprobs_token[x] for x in tokens.split(' ') if x])
    #         else:
    #             word = tokens
    #             logprob = logprobs_token[tokens]

    #         logprobs_word |= set([(word, logprob)])

    # df = pd.DataFrame(list(logprobs_word))
    # df.rename(columns={0: 'word', 1: 'count'}, inplace=True)
    # import ipdb; ipdb.set_trace()

    # if (df.shape[0] != df.word.unique().shape[0]):
    #     print(f'Number of types in testset {df.word.unique().shape[0]}. Number of logprobs {df.shape[0]}')
    #     df = df.groupby('word')['count'].agg('min').reset_index()
    # return df


def process_file(src_file, per_word, tokenizer):
    # counts = count_tokens(src_file, per_word, tokenizer)
    tokens = process_tokens(src_file, per_word, tokenizer)

    df = pd.DataFrame.from_dict(tokens, orient='index').reset_index()
    df.rename(columns={'index': 'word'}, inplace=True)
    df = df.explode('subwords')
    # df = pd.DataFrame.from_dict(counts, orient='index').reset_index()
    # df.rename(columns={'index': 'word', 0: 'freq'}, inplace=True)
    # import ipdb; ipdb.set_trace()
    if (df.shape[0] != df.word.unique().shape[0]):
        print(f'Number of types in data {df.word.unique().shape[0]}. Number of logprobs {df.shape[0]}')
        # df = df.groupby('word')['count'].agg('min').reset_index()
    return df


def process_files(train_file, val_file, test_file, tgt_file, per_word, tokenizer, keep_test_only):
    # Get train/val/test counts
    dfs = []
    for fname, mode in [(train_file, 'train_freq'), (val_file, 'val_freq'), 
                        (test_file, 'test_freq')]:
        df_temp = process_file(fname, per_word, tokenizer)
        df_temp['mode'] = mode
        dfs += [df_temp]

    df = pd.concat(dfs)

    # Pivot table
    df = df.pivot_table(values='count', index=['word', 'subwords'], columns=['mode'], fill_value=0)
    df.reset_index(inplace=True)

    # import ipdb; ipdb.set_trace()
    # Compute logprobs using trainset
    logprobs = prob_pertoken(train_file, val_file, test_file)
    df['logprob'] = df.subwords.apply(lambda x: sum([logprobs[subword] for subword in x]))

    # import ipdb; ipdb.set_trace()
    # df.sort_values('train_freq', ascending=False)
    # df[df.test_freq > 0].corr('spearman')

    # Filter data
    if keep_test_only:
        df = df[df.test_freq > 0].copy()

    # Save table
    df.to_csv(tgt_file, sep='\t')


def main():
    args = get_args()
    process_files(args.train_file, args.val_file, args.test_file, 
                  args.tgt_file, args.per_word, args.tokenizer,
                  args.keep_test_only)


if __name__ == "__main__":
    main()