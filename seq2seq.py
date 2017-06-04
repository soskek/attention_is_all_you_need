# encoding: utf-8

import argparse
import collections
import os.path

from nltk.corpus import comtrans
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
            smoothing_function=bleu_score.SmoothingFunction().method1)
        reporter.report({self.key: bleu})


def main():
    parser = argparse.ArgumentParser(description='Chainer example: seq2seq')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=512,
                        help='Number of units')
    parser.add_argument('--input', '-i', type=str, default='wmt',
                        help='Input directory')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()

    if False:
        sentences = comtrans.aligned_sents('alignment-en-fr.txt')
        source_ids = collections.defaultdict(lambda: len(source_ids))
        target_ids = collections.defaultdict(lambda: len(target_ids))
        target_ids['eos']
        data = []
        for sentence in sentences:
            source = numpy.array([source_ids[w] for w in sentence.words], 'i')
            target = numpy.array([target_ids[w] for w in sentence.mots], 'i')
            data.append((source, target))
        print('Source vocabulary: %d' % len(source_ids))
        print('Target vocabulary: %d' % len(target_ids))

        test_data = data[:len(data) / 10]
        train_data = data[len(data) / 10:]
    else:
        # Check file
        en_path = os.path.join(args.input, 'giga-fren.release2.fixed.en')
        source_vocab = ['<eos>', '<unk>'] + europal.count_words(en_path)
        source_data = europal.make_dataset(en_path, source_vocab)
        fr_path = os.path.join(args.input, 'giga-fren.release2.fixed.fr')
        target_vocab = ['<eos>', '<unk>'] + europal.count_words(fr_path)
        target_data = europal.make_dataset(fr_path, target_vocab)
        assert len(source_data) == len(target_data)
        print('Original training data size: %d' % len(source_data))
        train_data = [(s, t)
                      for s, t in six.moves.zip(source_data, target_data)
                      if 0 < len(s) < 45 and 0 < len(t) < 45]
        print('Filtered training data size: %d' % len(train_data))

        en_path = os.path.join(args.input, 'dev', 'newstest2013.en')
        source_data = europal.make_dataset(en_path, source_vocab)
        fr_path = os.path.join(args.input, 'dev', 'newstest2013.fr')
        target_data = europal.make_dataset(fr_path, target_vocab)
        assert len(source_data) == len(target_data)
        test_data = [(s, t) for s, t in six.moves.zip(source_data, target_data)
                     if 0 < len(s) and 0 < len(t)]

        source_ids = {word: index for index, word in enumerate(source_vocab)}
        target_ids = {word: index for index, word in enumerate(target_vocab)}

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}

    model = net.Seq2seq(15, len(source_ids), len(target_ids), args.unit)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    #optimizer = chainer.optimizers.NesterovAG(lr=0.1, momentum=0.99)
    optimizer = chainer.optimizers.NesterovAG(lr=0.25, momentum=0.99)
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer.GradientClipping(0.25))
    optimizer.add_hook(chainer.optimizer.GradientClipping(0.1))

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    updater = training.StandardUpdater(
        train_iter, optimizer,
        converter=seq2seq_pad_concat_convert, device=args.gpu)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=(200, 'iteration')),
                   trigger=(200, 'iteration'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
         'main/perp', 'validation/main/perp', 'validation/main/bleu',
         'elapsed_time']),
        trigger=(200, 'iteration'))
    # TODO: realize "We use a learning rate of 0.25 and once the validation
    # perplexity stops improving, we reduce the learning rate by an order of
    # magnitude after each epoch until it falls below 10^-4"
    # trainer.extend(extensions.ExponentialShift('lr', 0.25),
    #               trigger=(1, 'epoch'))
    trainer.extend(extensions.ExponentialShift('lr', 0.75),
                   trigger=(1, 'epoch'))

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

    trainer.extend(translate, trigger=(200, 'iteration'))
    trainer.extend(
        CalculateBleu(
            model, test_data, 'validation/main/bleu',
            device=args.gpu, batch=args.batchsize // 4),
        trigger=(20000, 'iteration'))
    print('start training')
    trainer.run()


if __name__ == '__main__':
    main()
