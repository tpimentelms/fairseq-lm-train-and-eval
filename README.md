# fairseq-lm-train-and-eval

Code to train language models using fairseq and evaluate them to get per word surprisals.

## Install Dependencies

First, create a conda environment with
```bash
$ conda env create -f scripts/environment.yml
```
Then activate the environment and install your appropriate version of [PyTorch](https://pytorch.org/get-started/locally/).
```bash
$ conda install pytorch torchvision cpuonly -c pytorch
$ # conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
$ # conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
$ pip install transformers
$ pip install --force-reinstall charset-normalizer==3.1.0
```
Also, install the fairseq library:
```bash
$ git submodule update --init --recursive
$ cd fairseq
$ pip install --editable ./
```
Alternatively, in hpc, install fairseq with `pip install fairseq`, then change file `<conda_path>/envs/lang-model-training/lib/python3.10/site-packages/fairseq_cli/train.py`'s line `task = tasks.setup_task(cfg.task)` to `task = tasks.setup_task(cfg.task, bpe=cfg.bpe)`.

## Run Pipeline

Train the models with command:
```bash
$ make LANGUAGE=${lang} DATASET=wiki40b MAX_TOKENS_TRAIN=${max_tokens}
```

Then get surprisals using command:
```bash
$ make -f MakefileEval LANGUAGE=${lang} DATASET=${dataset} DATASET_TRAIN=wiki40b MAX_TOKENS_TRAIN=${max_tokens}
```
