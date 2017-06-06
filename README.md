# Convolutional Sequence to Sequence Learning
This repository includes an implementation of a convolutional seq2seq model by [Chainer](https://github.com/chainer/chainer/).
This is derived from Chainer's official [seq2seq example](https://github.com/chainer/chainer/tree/seq2seq-europal/examples/seq2seq).

See [Convolutional Sequence to Sequence Learning](https://arxiv.org/abs/1705.03122), Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, Yann N. Dauphin, arxiv, 2017. [blog post](https://code.facebook.com/posts/1978007565818999/a-%20novel-approach-to-neural-machine-translation/), [Torch code](https://github.com/facebookresearch/fairseq).


## How to Run
```
PYTHONIOENCODING=utf-8 python -u seq2seq.py -g=0 -i DATA_DIR -o SAVE_DIR -b 48 -e 100
```

`DATA_DIR` directory needs to include training pair data `giga-fren.release2.fixed.en` and `giga-fren.release2.fixed.fr`, in addition to validation pair data `dev/newstest2013.en` and `dev/newstest2013.fr`. Other pair dataset which has line-by-line sentence alignment can be used.

JSON log report file and a model snapshot will be saved in `SAVE_DIR` directory (if it does not exist, it will be automatically made).
