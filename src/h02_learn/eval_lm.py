#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the perplexity of a trained language model.
"""

import logging
import math
import os
import sys
from argparse import Namespace
from typing import Iterable, List, Optional

from tqdm import tqdm
import pandas as pd
import torch
from omegaconf import DictConfig

import fairseq
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter
from fairseq.sequence_scorer import SequenceScorer
from fairseq.criterions.fairseq_criterion import FairseqCriterion
from fairseq.tasks import FairseqTask


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.eval_lm")


class SequenceEntropyScorer(SequenceScorer):
    """Scores the target for a given source sentence."""

    def batch_for_softmax(self, dec_out, target):
        # assumes decoder_out[0] is the only thing needed (may not be correct for future models!)
        first, rest = dec_out[0], dec_out[1:]
        bsz, tsz, dim = first.shape
        if bsz * tsz < self.softmax_batch:
            yield dec_out, target, True
        else:
            flat = first.contiguous().view(1, -1, dim)
            flat_tgt = target.contiguous().view(flat.shape[:-1])
            s = 0
            while s < flat.size(1):
                e = s + self.softmax_batch
                yield (flat[:, s:e],) + rest, flat_tgt[:, s:e], False
                s = e

    @staticmethod
    def gather_target_probs(probs, target):
        probs = probs.gather(
            dim=2,
            index=target.unsqueeze(-1),
        )
        return probs

    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Score a batch of translations."""
        net_input = sample["net_input"]


        orig_target = sample["target"]

        # compute scores for each model in the ensemble
        avg_probs = None
        avg_probs_full = None
        avg_attn = None
        for model in models:
            model.eval()
            decoder_out = model(**net_input)
            attn = decoder_out[1] if len(decoder_out) > 1 else None
            if type(attn) is dict:
                attn = attn.get("attn", None)

            batched = self.batch_for_softmax(decoder_out, orig_target)
            probs, idx = None, 0
            for bd, tgt, is_single in batched:
                sample["target"] = tgt
                curr_prob = model.get_normalized_probs(
                    bd, log_probs=len(models) == 1, sample=sample
                ).data
                assert is_single
                probs = self.gather_target_probs(curr_prob, orig_target)

                sample["target"] = orig_target

            probs = probs.view(sample["target"].shape)

            if avg_probs is None:
                avg_probs = probs
                avg_probs_full = curr_prob
            else:
                avg_probs.add_(probs)
                avg_probs_full.add_(curr_prob)

            if attn is not None:
                if torch.is_tensor(attn):
                    attn = attn.data
                else:
                    attn = attn[0]
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)

        if len(models) > 1:
            raise NotImplementedError('Not implemented for multiple models')

        bsz = avg_probs.size(0)
        hypos = []
        start_idxs = sample["start_indices"] if "start_indices" in sample else [0] * bsz
        for i in range(bsz):
            # remove padding from ref
            ref = (
                utils.strip_pad(sample["target"][i, start_idxs[i] :], self.pad)
                if sample["target"] is not None
                else None
            )
            tgt_len = ref.numel()
            avg_probs_i = avg_probs[i][start_idxs[i] : start_idxs[i] + tgt_len]
            full_vocab_probs_i = avg_probs_full[i][start_idxs[i] : start_idxs[i] + tgt_len]

            score_i = avg_probs_i.sum() / tgt_len
            if avg_attn is not None:
                avg_attn_i = avg_attn[i]
                if self.compute_alignment:
                    alignment = utils.extract_hard_alignment(
                        avg_attn_i,
                        sample["net_input"]["src_tokens"][i],
                        sample["target"][i],
                        self.pad,
                        self.eos,
                    )
                else:
                    alignment = None
            else:
                avg_attn_i = alignment = None
            hypos.append(
                [
                    {
                        "tokens": ref,
                        "score": score_i,
                        "attention": avg_attn_i,
                        "alignment": alignment,
                        "positional_scores": avg_probs_i,
                        "full_scores": full_vocab_probs_i,
                    }
                ]
            )
        return hypos

class RawWordsPostprocessor:
    def postprocess(self, df):
        df['sow'] = 1
        df['word_id'] = df.groupby('sample_id').sow.cumsum()
        df['word'] = df.symbol
        df['n_subtokens'] = 1

        return df[['sample_id', 'word_id', 'word', 'score', 'shannon', 'renyi', 'n_subtokens']]


class UnigramLMPostprocessor:
    new_word_symbol = '\u2581'
    new_word_len = len(new_word_symbol)

    def __init__(self, dictionary):
        self.dictionary = dictionary

    @classmethod
    def is_beginning_of_word(cls, x: str) -> bool:
        if x in ["<unk>", "<s>", "</s>", "<pad>"]:
            # special elements are always considered beginnings
            return True
        return x.startswith(cls.new_word_symbol)

    def clean_wordpiece(cls, x: str) -> str:
        if x.startswith(cls.new_word_symbol):
            return x[cls.new_word_len:] 
        else:
            return x

    def postprocess(self, df):
        df['sow'] = df.symbol.apply(self.is_beginning_of_word)
        df['word_id'] = df.groupby('sample_id').sow.cumsum()
        df['word'] = df.symbol.apply(self.clean_wordpiece)
        df['n_subtokens'] = 1
        df_perword = df.groupby(['sample_id', 'word_id']).agg({
            'word': sum, 'score': sum, 'shannon': 'first',
            'renyi': 'first', 'n_subtokens': sum})
        
        return df_perword.reset_index()


def eval_lm(
    models: List[fairseq.models.FairseqModel],
    source_dictionary: fairseq.data.Dictionary,
    batch_iterator: Iterable,
    post_process: Optional[str] = None,
    target_dictionary: Optional[fairseq.data.Dictionary] = None,
    softmax_batch: int = 0,
    results_path: Optional[str] = None,
    device: Optional[torch.device] = None,
):
    """
    Args:
        models (List[~fairseq.models.FairseqModel]): list of models to
            evaluate. Models are essentially `nn.Module` instances, but
            must be compatible with fairseq's `SequenceScorer`.
        source_dictionary (~fairseq.data.Dictionary): dictionary for
            applying any relevant post processing or outputing word
            probs/stats.
        batch_iterator (Iterable): yield batches of data
        post_process (Optional[str]): post-process text by removing BPE,
            letter segmentation, etc. Valid options can be found in
            fairseq.data.utils.post_process, although not all options
            are implemented here.
        target_dictionary (Optional[~fairseq.data.Dictionary]): output
            dictionary (defaults to *source_dictionary*)
        softmax_batch (Optional[bool]): if BxT is more than this, will
            batch the softmax over vocab to this amount of tokens, in
            order to fit into GPU memory
        results_path (Optional[str]): path where to write results
        device (Optional[torch.device]): device to use for evaluation
            (defaults to device of first model parameter)
    """
    if target_dictionary is None:
        target_dictionary = source_dictionary
    if device is None:
        device = next(models[0].parameters()).device
    model = models[0]

    gen_timer = StopwatchMeter()
    scorer = SequenceEntropyScorer(target_dictionary, softmax_batch)

    score_sum = 0.0
    count = 0

    results = [['instance_id', 'batch_id', 'hypo_id', 'batch_position_id', 'symbol_id', 'symbol', 'score', 'shannon', 'renyi']]

    for batch_id, sample in enumerate(tqdm(batch_iterator, desc='Evaluating all batches', file=sys.stdout)):
        if "net_input" not in sample:
            continue

        sample = utils.move_to_cuda(sample, device=device)

        gen_timer.start()
        hypos = scorer.generate(models, sample)
        gen_timer.stop(sample["ntokens"])

        for hypo_id, hypos_i in enumerate(hypos):
            hypo = hypos_i[0]
            instance_id = sample["id"][hypo_id].item()

            tokens = hypo["tokens"]
            tgt_len = tokens.numel()
            pos_scores = - hypo["positional_scores"].float()
            symbols = [source_dictionary[token] for token in tokens]

            logprobs_full = hypo['full_scores'].float()
            entropy_shannons = - (logprobs_full.exp() * logprobs_full).sum(-1)
            entropy_renyis = (1 / (1 - .5)) * (logprobs_full.exp().pow(.5).sum(-1).log())

            inf_scores = pos_scores.eq(float("inf")) | pos_scores.eq(float("-inf"))
            assert not inf_scores.any(), 'Tokens should not have inf score'

            assert len(symbols) == pos_scores.shape[0]
            results += [[instance_id, batch_id, hypo_id, i, symbol_id.item(), symbol,
                         score.item(), shannon.item(), renyi.item()]
                        for i, (symbol_id, symbol, score, shannon, renyi) in 
                        enumerate(zip(tokens, symbols, pos_scores, entropy_shannons, entropy_renyis))]

    df = pd.DataFrame(results[1:], columns=results[0])

    if post_process is not None:
        if post_process in ['unigramlm', 'bpe']:
            postprocessor = UnigramLMPostprocessor(source_dictionary)
        elif post_process == 'rawwords':
            postprocessor = RawWordsPostprocessor()
        else:
            raise NotImplementedError(
                f"--post-process={post_process} is not implemented"
            )

        # Create dataframe
        logger.info("Creating dataframe from surprisals")
        df_full = df

        # Split tested sentences
        logger.info("Splitting tested sentences")
        df_full['eos'] = False
        df_full.loc[df_full.symbol == '</s>', 'eos'] = True
        df_full['sos'] = df_full.eos.shift(1)
        df_full.loc[0, 'sos'] = True
        assert (df_full.eos.sum() == df_full.sos.sum())
        df_full['sample_id'] = df_full.sos.cumsum() - 1

        # Get surprisals per word
        logger.info("Merging surprisals per word")
        df = postprocessor.postprocess(df_full)

    # Print scores
    avg_nll_loss = (
        df.score.sum() / df.shape[0] / math.log(2)
    )  # convert to base 2
    logger.info(
        "Evaluated {:,} tokens in {:.1f}s ({:.2f} tokens/s)".format(
            gen_timer.n, gen_timer.sum, 1.0 / gen_timer.avg if gen_timer.avg > 0 else 0
        )
    )

    if results_path:
        logger.info("Saving surprisal info")
        df.to_csv(results_path, sep='\t')
    return {
        "loss": avg_nll_loss,
        "perplexity": 2**avg_nll_loss,
    }


def main(cfg: DictConfig, **unused_kwargs):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    logger.info(cfg)

    if cfg.eval_lm.context_window > 0:
        # reduce tokens per sample by the required context window size
        cfg.task.tokens_per_sample -= cfg.eval_lm.context_window

    # Initialize the task using the current *cfg*
    task = tasks.setup_task(cfg.task, bpe=cfg.bpe)

    # Load ensemble
    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [cfg.common_eval.path],
        arg_overrides=eval(cfg.common_eval.model_overrides),
        suffix=cfg.checkpoint.checkpoint_suffix,
        strict=(cfg.checkpoint.checkpoint_shard_count == 1),
        num_shards=cfg.checkpoint.checkpoint_shard_count,
        task=task,
    )

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu
    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    # Optimize ensemble for generation and set the source and dest dicts on the model
    # (required by scorer)
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    assert len(models) > 0

    logger.info(
        "num. model params: {:,}".format(sum(p.numel() for p in models[0].parameters()))
    )

    # Load dataset splits
    task.load_dataset(cfg.dataset.gen_subset)
    dataset = task.dataset(cfg.dataset.gen_subset)
    logger.info(
        "{} {} {:,} examples".format(
            cfg.task.data, cfg.dataset.gen_subset, len(dataset)
        )
    )

    itr = task.eval_lm_dataloader(
        dataset=dataset,
        max_tokens=cfg.dataset.max_tokens or 36000,
        batch_size=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            *[model.max_positions() for model in models]
        ),
        num_shards=max(
            cfg.dataset.num_shards,
            cfg.distributed_training.distributed_world_size,
        ),
        shard_id=max(
            cfg.dataset.shard_id,
            cfg.distributed_training.distributed_rank,
        ),
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
        context_window=cfg.eval_lm.context_window,
    )

    # itr = progress_bar.progress_bar(
    #     itr,
    #     log_format=cfg.common.log_format,
    #     log_interval=cfg.common.log_interval,
    #     default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    # )

    results = eval_lm(
        models=models,
        source_dictionary=task.source_dictionary,
        batch_iterator=itr,
        post_process=cfg.common_eval.post_process,
        target_dictionary=task.target_dictionary,
        softmax_batch=cfg.eval_lm.softmax_batch,
        results_path=cfg.common_eval.results_path,
    )

    logger.info(
        "Loss (base 2): {:.4f}, Perplexity: {:.2f}".format(
            results["loss"], results["perplexity"]
        )
    )

    return results


def cli_main():
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser)

    distributed_utils.call_main(convert_namespace_to_omegaconf(args), main)


if __name__ == "__main__":
    cli_main()
