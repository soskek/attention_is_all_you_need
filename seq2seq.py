# encoding: utf-8

import argparse
import json
import os.path

from nltk.translate import bleu_score
import numpy
import six

import chainer
from chainer import cuda
from chainer.dataset import convert
from chainer import reporter
from chainer import training
from chainer.training import extensions

import europal
import net
from subfuncs import FailMinValueTrigger


def seq2seq_pad_concat_convert(xy_batch, device, eos_id=0):
    """
    Args:
        xy_batch (list of tuple of two numpy.ndarray-s or cupy.ndarray-s):
            xy_batch[i][0] is an array
            of token ids of i-th input sentence in a minibatch.
            xy_batch[i][1] is an array
            of token ids of i-th target sentence in a minibatch.
            The shape of each array is `(sentence length, )`.
        device (int or None): Device ID to which an array is sent. If it is
            negative value, an array is sent to CPU. If it is positive, an
            array is sent to GPU with the given ID. If it is ``None``, an
            array is left in the original device.

    Returns:
        Tuple of Converted array.
            (input_sent_batch_array, target_sent_batch_input_array,
            target_sent_batch_output_array).
            The shape of each array is `(batchsize, max_sentence_length)`.
            All sentences are padded with -1 to reach max_sentence_length.
    """

    x_seqs, y_seqs = zip(*xy_batch)

    x_block = convert.concat_examples(x_seqs, device, padding=-1)
    y_block = convert.concat_examples(y_seqs, device, padding=-1)
    xp = cuda.get_array_module(x_block)

    # add eos
    x_block = xp.pad(x_block, ((0, 0), (0, 1)),
                     'constant', constant_values=-1)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id

    y_out_block = xp.pad(y_block, ((0, 0), (0, 1)),
                         'constant', constant_values=-1)
    for i_batch, seq in enumerate(y_seqs):
        y_out_block[i_batch, len(seq)] = eos_id

    y_in_block = xp.pad(y_block, ((0, 0), (1, 0)),
                        'constant', constant_values=eos_id)
    return (x_block, y_in_block, y_out_block)


def source_pad_concat_convert(x_seqs, device, eos_id=0):
    x_block = convert.concat_examples(x_seqs, device, padding=-1)
    xp = cuda.get_array_module(x_block)

    # add eos
    x_block = xp.pad(x_block, ((0, 0), (0, 1)),
                     'constant', constant_values=-1)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id
    return x_block


class CalculateBleu(chainer.training.Extension):

    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(
            self, model, test_data, key, batch=50, device=-1, max_length=50):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = device
        self.max_length = max_length

    def __call__(self, trainer):
        print('## Calculate BLEU')
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                references = []
                hypotheses = []
                for i in range(0, len(self.test_data), self.batch):
                    sources, targets = zip(*self.test_data[i:i + self.batch])
                    references.extend([[t.tolist()] for t in targets])

                    sources = [
                        chainer.dataset.to_device(self.device, x) for x in sources]
                    ys = [y.tolist()
                          for y in self.model.translate(sources, self.max_length)]
                    hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(
            references, hypotheses,
            smoothing_function=bleu_score.SmoothingFunction().method1) * 100
        print('BLEU:', bleu)
        reporter.report({self.key: bleu})


def main():
    parser = argparse.ArgumentParser(
        description='Chainer example: convolutional seq2seq')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=512,
                        help='Number of units')
    parser.add_argument('--layer', '-l', type=int, default=15,
                        help='Number of layers')
    parser.add_argument('--input', '-i', type=str, default='./',
                        help='Input directory')
    parser.add_argument('--source', '-s', type=str,
                        default='europarl-v7.fr-en.en',
                        help='Filename of train data for source language')
    parser.add_argument('--target', '-t', type=str,
                        default='europarl-v7.fr-en.fr',
                        help='Filename of train data for target language')
    parser.add_argument('--source-valid', '-svalid', type=str,
                        default='dev/newstest2013.en',
                        help='Filename of validation data for source language')
    parser.add_argument('--target-valid', '-tvalid', type=str,
                        default='dev/newstest2013.fr',
                        help='Filename of validation data for target language')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--source-vocab', type=int, default=40000,
                        help='Vocabulary size of source language')
    parser.add_argument('--target-vocab', type=int, default=40000,
                        help='Vocabulary size of target language')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=4))

    # Check file
    en_path = os.path.join(args.input, args.source)
    source_vocab = ['<eos>', '<unk>'] + \
        europal.count_words(en_path, args.source_vocab)
    source_data = europal.make_dataset(en_path, source_vocab)
    fr_path = os.path.join(args.input, args.target)
    target_vocab = ['<eos>', '<unk>'] + \
        europal.count_words(fr_path, args.target_vocab)
    target_data = europal.make_dataset(fr_path, target_vocab)
    assert len(source_data) == len(target_data)
    print('Original training data size: %d' % len(source_data))
    train_data = [(s, t)
                  for s, t in six.moves.zip(source_data, target_data)
                  if 0 < len(s) < 50 and 0 < len(t) < 50]
    print('Filtered training data size: %d' % len(train_data))

    en_path = os.path.join(args.input, args.source_valid)
    source_data = europal.make_dataset(en_path, source_vocab)
    fr_path = os.path.join(args.input, args.target_valid)
    target_data = europal.make_dataset(fr_path, target_vocab)
    assert len(source_data) == len(target_data)
    test_data = [(s, t) for s, t in six.moves.zip(source_data, target_data)
                 if 0 < len(s) and 0 < len(t)]

    source_ids = {word: index for index, word in enumerate(source_vocab)}
    target_ids = {word: index for index, word in enumerate(target_vocab)}

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}

    # Define Model
    model = net.Seq2seq(
        args.layer, len(source_ids), len(target_ids), args.unit)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    # Setup Optimizer
    optimizer = chainer.optimizers.NesterovAG(lr=0.25, momentum=0.99)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(0.1))

    # Setup Trainer
    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize,
                                                 repeat=False, shuffle=False)
    iter_per_epoch = len(train_data) // args.batchsize
    print('Number of iter/epoch =', iter_per_epoch)

    updater = training.StandardUpdater(
        train_iter, optimizer,
        converter=seq2seq_pad_concat_convert, device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # If you want to change a logging interval, change this number
    log_trigger = (min(1000, iter_per_epoch // 2), 'iteration')

    def floor_step(trigger):
        floored = trigger[0] - trigger[0] % log_trigger[0]
        if floored <= 0:
            floored = trigger[0]
        return (floored, trigger[1])

    # Validation every half epoch
    eval_trigger = floor_step((iter_per_epoch // 2, 'iteration'))
    fail_trigger = FailMinValueTrigger('val/main/perp', eval_trigger)
    record_trigger = training.triggers.MaxValueTrigger(
        'val/main/perp', eval_trigger)

    evaluator = extensions.Evaluator(
        test_iter, model,
        converter=seq2seq_pad_concat_convert,
        device=args.gpu)
    evaluator.default_name = 'val'
    trainer.extend(evaluator, trigger=eval_trigger)
    # Only if validation perplexity fails to be improved,
    # lr is decayed (until 1e-4).
    trainer.extend(extensions.ExponentialShift('lr', 0.1, target=1e-4),
                   trigger=fail_trigger)
    trainer.extend(extensions.observe_lr(), trigger=eval_trigger)
    # Only if a model gets best validation score,
    # save the model
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}.npz'),
        trigger=record_trigger)

    def translate_one(source, target):
        words = europal.split_sentence(source)
        print('# source : ' + ' '.join(words))
        x = model.xp.array(
            [source_ids.get(w, 1) for w in words], 'i')
        ys = model.translate([x])[0]
        words = [target_words[y] for y in ys]
        print('#  result : ' + ' '.join(words))
        print('#  expect : ' + target)

    @chainer.training.make_extension(trigger=(200, 'iteration'))
    def translate(trainer):
        translate_one(
            'Who are we ?',
            'Qui sommes-nous?')
        translate_one(
            'And it often costs over a hundred dollars ' +
            'to obtain the required identity card .',
            'Or, il en coûte souvent plus de cent dollars ' +
            'pour obtenir la carte d\'identité requise.')

        source, target = test_data[numpy.random.choice(len(test_data))]
        source = ' '.join([source_words[i] for i in source])
        target = ' '.join([target_words[i] for i in target])
        translate_one(source, target)

    # Gereneration Test
    trainer.extend(
        translate,
        trigger=(min(200, iter_per_epoch), 'iteration'))
    # Calculate BLEU every half epoch
    trainer.extend(
        CalculateBleu(
            model, test_data, 'val/main/bleu',
            device=args.gpu, batch=args.batchsize // 4),
        trigger=floor_step((iter_per_epoch // 2, 'iteration')))

    # Log
    trainer.extend(extensions.LogReport(trigger=log_trigger),
                   trigger=log_trigger)
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration',
         'main/loss', 'val/main/loss',
         'main/perp', 'val/main/perp',
         'main/acc', 'val/main/acc',
         'val/main/bleu',
         'lr',
         'elapsed_time']),
        trigger=log_trigger)

    print('start training')
    trainer.run()


if __name__ == '__main__':
    main()
