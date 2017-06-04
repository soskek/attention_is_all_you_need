# Convolutional Sequence to Sequence Learning [WIP]
This repository includes an implementation of a convolutional seq2seq model by [Chainer](https://github.com/chainer/chainer/).
This is derived from Chainer's official [seq2seq example](https://github.com/chainer/chainer/tree/seq2seq-europal/examples/seq2seq).

See *Convolutional Sequence to Sequence Learning*, Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, Yann N. Dauphin, [arxiv](https://arxiv.org/abs/1705.03122), [blog post](https://code.facebook.com/posts/1978007565818999/a-%20novel-approach-to-neural-machine-translation/), [Torch code](https://github.com/facebookresearch/fairseq)


## How to Run
```
PYTHONIOENCODING=utf-8 python -u seq2seq.py -g=0 -i DATA_DIR -o SAVE_DIR/SAVE_NAME -b 48 -e 100 > LOG_NAME &
```

`DATA_DIR` need to include training pair data `giga-fren.release2.fixed.en` and `giga-fren.release2.fixed.fr`, in addition to validation pair data `dev/newstest2013.en` and `dev/newstest2013.fr`. Other pair dataset which has line-by-line sentence alignment can be used.
