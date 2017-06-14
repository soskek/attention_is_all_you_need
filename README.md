# [WIP] Transformer - Attention Is All You Need
[Chainer](https://github.com/chainer/chainer/)-based Python implementation of Transformer, an attention-based seq2seq model without convolution and recurrence.

This is derived from [my other repository](https://github.com/soskek/convolutional_seq2seq), which is derived from Chainer's official [seq2seq example](https://github.com/chainer/chainer/tree/seq2seq-europal/examples/seq2seq).

See [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017.

## Requirement

- Python 3.6.0+
- [Chainer](https://github.com/chainer/chainer/) 2.0.0+ (this version is strictly required)
- [numpy](https://github.com/numpy/numpy) 1.12.1+
- [cupy](https://github.com/cupy/cupy) 1.0.0+ (if using gpu)
- and their dependencies

## Prepare Dataset
You can use any parallel corpus.  
For example, run `download_wmt.sh` which downloads and decompresses [training dataset](http://www.statmt.org/europarl/v7/fr-en.tgz) and [development dataset](http://www.statmt.org/wmt15/dev-v2.tgz) from [WMT](http://www.statmt.org/wmt15/translation-task.html#download)/[europal](http://www.statmt.org/europarl/) into your current directory. These files and their paths are set in training script `seq2seq.py` as default.

## How to Run
```
PYTHONIOENCODING=utf-8 python -u seq2seq.py -g=0 -i DATA_DIR -o SAVE_DIR
```

During training, logs for loss, perplexity, word accuracy and time are printed at a certain internval, in addition to validation tests (perplexity and BLEU for generation) every half epoch. And also, generation test is performed and printed for checking training progress.

### Arguments

- `-g`: your gpu id. If cpu, set `-1`.
- `-i DATA_DIR`, `-s SOURCE`, `-t TARGET`, `-svalid SVALID`, `-tvalid TVALID`:  
  `DATA_DIR` directory needs to include a pair of training dataset `SOURCE` and `TARGET` with a pair of validation dataset `SVALID` and `TVALID`. Each pair should be parallell corpus with line-by-line sentence alignment.
- `-o SAVE_DIR`: JSON log report file and a model snapshot will be saved in `SAVE_DIR` directory (if it does not exist, it will be automatically made).
- `-e`: max epochs of training corpus.
- `-b`: minibatch size.
- `-u`: size of units and word embeddings.
- `-l`: number of layers in both the encoder and the decoder.
- `--source-vocab`: max size of vocabulary set of source language
- `--target-vocab`: max size of vocabulary set of target language
