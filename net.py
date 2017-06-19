# encoding: utf-8

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import reporter

from seq2seq import source_pad_concat_convert


def sentence_block_embed(embed, x):
    batch, length = x.shape
    e = embed(x.reshape((batch * length, )))
    # (batch * length, units)
    e = F.transpose(F.stack(F.split_axis(e, batch, axis=0), axis=0), (0, 2, 1))
    # (batch, units, length)
    return e


def seq_func(func, x):
    batch, units, length = x.shape
    e = F.transpose(x, (0, 2, 1)).reshape(batch * length, units)
    e = func(e)
    out_units = e.shape[1]
    e = F.transpose(e.reshape((batch, length, out_units)), (0, 2, 1))
    return e


class AttentionLayer(chainer.Chain):
    def __init__(self, n_units, h=8, dropout=0.1):
        super(AttentionLayer, self).__init__(
            W_Q=L.Linear(n_units, n_units),
            W_K=L.Linear(n_units, n_units),
            W_V=L.Linear(n_units, n_units),
            FinishingLinearLayer=L.Linear(n_units, n_units),
        )
        self.h = h
        self.dropout = dropout

    def __call__(self, x, z, mask):
        # TODO: shape check
        """
        Input shapes:
            q=(b, units, n_querys), k=(b, units, n_keys),
            m=(b, n_querys, n_keys)
        """

        query = seq_func(self.W_Q, x)
        key = seq_func(self.W_K, z)
        value = seq_func(self.W_V, z)
        batch, n_units, n_querys = query.shape

        # Calculate Attention Scores with Mask for Zero-padded Areas
        # Perform Multi-head Attention using pseudo batching
        # all together at once for efficiency

        pseudo_batch_query = F.concat(F.split_axis(query, self.h, axis=1),
                                      axis=0)
        pseudo_batch_key = F.concat(F.split_axis(key, self.h, axis=1),
                                    axis=0)
        pseudo_batch_value = F.concat(F.split_axis(value, self.h, axis=1),
                                      axis=0)
        # q.shape = (b * h, n_units // h, n_querys)
        # k.shape = (b * h, n_units // h, n_keys)
        # v.shape = (b * h, n_units // h, n_keys)

        a = F.batch_matmul(
            pseudo_batch_query, pseudo_batch_key, transa=True)
        # a.shape = (b * h, n_querys, n_keys)
        a /= (n_units // self.h) ** 0.5
        minfs = self.xp.full(a.shape, -np.inf, a.dtype)
        mask = self.xp.concatenate([mask] * self.h, axis=0)
        a = F.where(mask, a, minfs)
        a = F.softmax(a, axis=2)
        # if values in axis=2 are all -inf, they become nan. thus do re-mask.
        a = F.where(self.xp.isnan(a.data),
                    self.xp.zeros(a.shape, dtype=a.dtype), a)
        # Calculate Weighted Sum
        a, pseudo_batch_value = F.broadcast(
            a[:, None], pseudo_batch_value[:, :, None])
        # shape = (b * h, n_units // h, n_querys, n_keys)
        multi_c = F.sum(a * pseudo_batch_value, axis=3)
        # (b * h, units // h, n_querys)
        c = F.concat(F.split_axis(multi_c, self.h, axis=0), axis=1)
        lineared_c = seq_func(self.FinishingLinearLayer, c)
        return lineared_c

# Section 3.3 says the inner-layer has dimension 2048.
# But, Table 4 says d_{ff} = 1024 (for "base model").
# d_{ff}'s denotation is unclear, but it seems to denote the same one.
# So, we use 1024.


class EncoderLayer(chainer.Chain):
    def __init__(self, n_units, h=8, dropout=0.1, nopad=False):
        n_inner_units = n_units * 2
        super(EncoderLayer, self).__init__(
            W_1=L.Linear(n_units, n_inner_units),
            W_2=L.Linear(n_inner_units, n_units),
            SelfAttention=AttentionLayer(n_units, h),
            LN_1=L.LayerNormalization(n_units, eps=1e-9),
            LN_2=L.LayerNormalization(n_units, eps=1e-9),
        )
        self.dropout = dropout

    def __call__(self, e, xx_mask):
        e = e + F.dropout(self.SelfAttention(e, e, xx_mask),
                          ratio=self.dropout)
        e = seq_func(self.LN_1, e)
        e = e + F.dropout(seq_func(self.W_2, F.relu(seq_func(self.W_1, e))),
                          ratio=self.dropout)
        e = seq_func(self.LN_2, e)
        return e


class DecoderLayer(chainer.Chain):
    def __init__(self, n_units, h=8, dropout=0.1, nopad=False):
        n_inner_units = n_units * 2
        super(DecoderLayer, self).__init__(
            W_1=L.Linear(n_units, n_inner_units),
            W_2=L.Linear(n_inner_units, n_units),
            SourceAttention=AttentionLayer(n_units, h),
            SelfAttention=AttentionLayer(n_units, h),
            LN_1=L.LayerNormalization(n_units, eps=1e-9),
            LN_2=L.LayerNormalization(n_units, eps=1e-9),
            LN_3=L.LayerNormalization(n_units, eps=1e-9),
        )
        self.dropout = dropout

    def __call__(self, e, source, xy_mask, yy_mask):
        e = e + F.dropout(self.SelfAttention(e, e, yy_mask),
                          ratio=self.dropout)
        e = seq_func(self.LN_1, e)
        e = e + F.dropout(self.SourceAttention(e, source, xy_mask),
                          ratio=self.dropout)
        e = seq_func(self.LN_2, e)
        e = e + F.dropout(seq_func(self.W_2, F.relu(seq_func(self.W_1, e))),
                          ratio=self.dropout)
        e = seq_func(self.LN_3, e)
        return e


class Encoder(chainer.Chain):
    def __init__(self, n_layers, n_units, h=8, dropout=0.1):
        super(Encoder, self).__init__()
        links = [('l{}'.format(i + 1),
                  EncoderLayer(n_units, h=h, dropout=dropout))
                 for i in range(n_layers)]
        for link in links:
            self.add_link(*link)
        self.layer_names = [name for name, _ in links]

    def __call__(self, e, xx_mask):
        for name in self.layer_names:
            e = getattr(self, name)(e, xx_mask)
        return e


class Decoder(chainer.Chain):
    def __init__(self, n_layers, n_units, h=8, dropout=0.1):
        super(Decoder, self).__init__()
        links = [('l{}'.format(i + 1),
                  DecoderLayer(n_units, h=h, dropout=dropout))
                 for i in range(n_layers)]
        for link in links:
            self.add_link(*link)
        self.layer_names = [name for name, _ in links]

    def __call__(self, e, source, xy_mask, yy_mask):
        for name in self.layer_names:
            e = getattr(self, name)(e, source, xy_mask, yy_mask)
        return e


def make_position_encoding(xp, batch, length, n_units):
    # TODO: we can memoize as (1, n_units, max_len) at least
    one_start = False
    assert(n_units % 2 == 0)
    position_block = xp.broadcast_to(
        xp.arange(one_start, length + one_start)[None, None, :],
        (batch, n_units // 2, length)).astype('f')
    unit_block = xp.broadcast_to(
        xp.arange(one_start, n_units // 2 + one_start)[None, :, None],
        (batch, n_units // 2, length)).astype('f')
    rad_block = position_block / 10000. ** (unit_block / (n_units // 2))
    sin_block = xp.sin(rad_block)
    cos_block = xp.cos(rad_block)
    emb_block = xp.concatenate([sin_block, cos_block], axis=1)
    return emb_block

# TODO: remove eos?


class Seq2seq(chainer.Chain):

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units,
                 h=8, dropout=0.1):
        init = chainer.initializers.HeNormal(scale=0.5**0.5)
        #init = chainer.initializers.GlorotNormal(scale=1.)
        #init = None
        super(Seq2seq, self).__init__(
            embed_x=L.EmbedID(n_source_vocab, n_units, ignore_label=-1,
                              initialW=init),
            embed_y=L.EmbedID(n_target_vocab, n_units, ignore_label=-1,
                              initialW=init),
            encoder=Encoder(n_layers, n_units, h, dropout),
            decoder=Decoder(n_layers, n_units, h, dropout),
        )
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_target_vocab = n_target_vocab
        self.dropout = dropout

    def __call__(self, x_block, y_in_block, y_out_block, get_prediction=False):
        batch, x_length = x_block.shape
        batch, y_length = y_in_block.shape

        # Embed Words
        ex_block = sentence_block_embed(self.embed_x, x_block)
        ex_block *= self.n_units ** 0.5
        ey_block = sentence_block_embed(self.embed_y, y_in_block)
        ey_block *= self.n_units ** 0.5
        # (batch, n_units, x_length)

        # Encode Positions
        p_block = make_position_encoding(
            self.xp, batch, max(x_length, y_length), self.n_units)
        ex_block += p_block[:, :, :x_length]
        ey_block += p_block[:, :, :y_length]
        # (batch, n_units, x_length)

        ex_block = F.dropout(ex_block, ratio=self.dropout)
        ey_block = F.dropout(ey_block, ratio=self.dropout)

        # Make Masks for Encoding
        xx_mask = (x_block[:, None, :] >= 0) * \
            (x_block[:, :, None] >= 0)
        # (batch, x_length, x_length)

        # Encode Sources
        z_block = self.encoder(ex_block, xx_mask)
        # (batch, n_units, x_length)

        # Make Masks for Decoding
        xy_mask = (x_block[:, None, :] >= 0) * \
            (y_in_block[:, :, None] >= 0)
        # (batch, y_length, x_length)
        yy_mask = (y_in_block[:, None, :] >= 0) * \
            (y_in_block[:, :, None] >= 0)
        # (batch, y_length, y_length)

        # Add mask to Prevent Seeing Future
        arange = self.xp.arange(y_length)
        yy_history_mask = (arange[None, ] <= arange[:, None])[None, ]
        yy_mask *= yy_history_mask

        # Encode Targets with Sources (Decode without Output)
        h_block = self.decoder(
            ey_block, z_block, xy_mask, yy_mask)
        h_block = F.dropout(h_block, ratio=self.dropout)
        # (batch, n_units, y_length)

        if get_prediction:
            pred_tail = F.linear(
                h_block[:, :, -1], self.embed_y.W)
            return pred_tail
        else:
            # Output (all together at once for efficiency)
            concat_h_block = F.transpose(h_block, (0, 2, 1)).reshape(
                (batch * y_length, self.n_units))
            concat_pred_block = F.linear(
                concat_h_block, self.embed_y.W)

            # Calculate Loss, Accuracy, Perplexity
            concat_y_out_block = y_out_block.reshape((batch * y_length))
            ignore_mask = (concat_y_out_block >= 0)
            broad_ignore_mask = self.xp.broadcast_to(
                ignore_mask[:, None],
                concat_pred_block.shape)
            n_example = ignore_mask.sum()

            log_probability = F.log_softmax(concat_pred_block)
            pseudo_batchsize = concat_y_out_block.shape[0]
            pre_loss = log_probability[self.xp.arange(pseudo_batchsize),
                                       concat_y_out_block] * ignore_mask
            loss = - F.sum(pre_loss) / n_example
            accuracy = F.accuracy(
                concat_pred_block, concat_y_out_block, ignore_label=-1)
            perp = self.xp.exp(loss.data)
            # Report the Values
            reporter.report({'loss': loss.data,
                             'acc': accuracy.data,
                             'perp': perp}, self)

            do_ls = True
            if do_ls:
                label_smoothing = F.sum(
                    - 1. / self.n_target_vocab * log_probability * broad_ignore_mask) \
                    / n_example
                loss = 0.9 * loss + 0.1 * label_smoothing
            return loss

    def translate(self, x_block, max_length=50):
        # TODO: efficient inference by re-using convolution result
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                x_block = source_pad_concat_convert(
                    x_block, device=None)
                batch, x_length = x_block.shape
                y_block = self.xp.zeros((batch, 1), dtype=x_block.dtype)
                eos_flags = self.xp.zeros((batch, ), dtype=x_block.dtype)
                result = []
                for i in range(max_length):
                    log_prob_tail = self(x_block, y_block, y_block,
                                         get_prediction=True)
                    ys = self.xp.argmax(log_prob_tail.data, axis=1).astype('i')
                    result.append(ys)
                    y_block = F.concat([y_block, ys[:, None]], axis=1).data
                    eos_flags += (ys == 0)
                    if self.xp.all(eos_flags):
                        break

        result = cuda.to_cpu(self.xp.stack(result).T)

        # Remove EOS taggs
        outs = []
        for y in result:
            inds = np.argwhere(y == 0)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            if len(y) == 0:
                y = np.array([1], 'i')
            outs.append(y)
        return outs
