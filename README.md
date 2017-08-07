# Transformer - Attention Is All You Need
[Chainer](https://github.com/chainer/chainer/)-based Python implementation of Transformer, an attention-based seq2seq model without convolution and recurrence. If you want to understand its architecture, please see [net.py](https://github.com/soskek/attention_is_all_you_need/blob/master/net.py).
This is WIP, but works.

This repository does not aim for complete validation of results in the paper, so I have not eagerly confirmed validity of performance. But, I expect my implementation is almost compatible with a model described in the paper. Some differences where I am aware is as follows:  
- Optimization/training strategy. Detailed information about batchsize, parameter initialization, etc. is unclear in the paper. Additionally, the learning rate proposed in the paper may work only with a large batchsize (e.g. 4000) for deep layer nets. I modified it by multiplying 0.5, though there is room for improvement. I also changed `relu` into `leaky relu` in feedforward net layers for easy gradient propagation.
- Vocabulary set, dataset, preprocessing and evaluation. This repo uses a common word-based tokenization, although the paper uses byte-pair encoding. Size of token set also differs. Evaluation (validation) is little unfair and incompatible with one in the paper, e.g., even validation set replaces unknown words to a single "unk" token.
- Beam search. This is not yet added.
- Model size. The setting of a model in this repo is one of "base model" in the paper, although you can modify some lines for using "big model".
- This follows some settings used in [tensor2tensor repository](https://github.com/tensorflow/tensor2tensor/tree/master/tensor2tensor/models), which includes a Transformer model. For example, positional encoding used in the repository seems to differ from one in the paper. This code follows the former one.

This is derived from my [convolutional seq2seq](https://github.com/soskek/convolutional_seq2seq) repo, which is derived from Chainer's official [seq2seq example](https://github.com/chainer/chainer/tree/seq2seq-europal/examples/seq2seq).

See "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)", Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017.

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

Some of them is as follows:
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

Please see the others by `python seq2seq.py -h`.
