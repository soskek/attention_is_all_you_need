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
    h = func(F.transpose(x, (0, 2, 1)).reshape(batch * length, units))
    out_units = h.shape[1]
    return F.transpose(h.reshape((batch, length, out_units)), (0, 2, 1))


class AttentionLayer(chainer.Chain):
    def __init__(self, n_units, h, dropout=0.2):
        super(AttentionLayer, self).__init__(
            W_Q=L.Linear(n_units, n_units),
            W_K=L.Linear(n_units, n_units),
            W_V=L.Linear(n_units, n_units),
        )
        self.h = h
        self.dropout = dropout

    def __call__(self, x, z, mask):
        # TODO: shape check
        # TODO: dropout attention
        """
        Input shapes:
            q=(b, units, n_querys), k=(b, units, n_keys),
            v=(b, units, n_querys, n_keys), m=(b, n_querys, n_keys)
        """

        query = seq_func(self.W_Q, x)
        key = seq_func(self.W_K, z)
        value = seq_func(self.W_V, z)
        batch, n_units, n_querys = query.shape
        n_keys = key.shape[-1]
        value = F.broadcast_to(value, (batch, n_units, n_keys))

        # Calculate Attention Scores with Mask for Zero-padded Areas
        # Perform Multi-head Attention using pseudo batching
        # all together at once for efficiency
        children_query = F.split_axis(query, self.h, axis=1)
        # [(b, n_units // h, n_querys), ...]
        pseudo_batch_query = F.concat(children_query, axis=0)
        # (b * h, n_units // h, n_querys)

        children_key = F.split_axis(key, self.h, axis=1)
        # [(b, n_units // h, n_keys), ...]
        pseudo_batch_key = F.concat(children_key, axis=0)
        # (b * h, n_units // h, n_keys)

        pre_a = F.batch_matmul(
            pseudo_batch_query, pseudo_batch_key, transa=True)
        # (b * h, n_querys, n_keys)
        minfs = self.xp.full(pre_a.shape, -np.inf, pre_a.dtype)
        mask = self.xp.concatenate([mask] * self.h, axis=0)
        pre_a = F.where(mask, pre_a, minfs)
        pre_a /= (n_units // self.h) ** 0.5
        a = F.softmax(pre_a, axis=2)

        # if values in axis=2 are all -inf, they become nan. thus do re-mask.
        a = F.where(self.xp.isnan(a.data),
                    self.xp.zeros(a.shape, dtype=a.dtype), a)
        # (b, n_querys, n_keys)

        # Calculate Weighted Sum
        children_value = F.split_axis(value, self.h, axis=1)
        # [(b, n_units // h, n_keys), ...]
        pseudo_batch_value = F.concat(children_value, axis=0)
        # (b * h, n_units // h, n_keys)
        pseudo_batch_value = F.broadcast_to(
            pseudo_batch_value[:, :, None],
            (batch * self.h, n_units // self.h, n_querys, n_keys))

        a = F.dropout(a, ratio=self.dropout)
        reshaped_a = F.broadcast_to(
            a[:, None],
            (batch * self.h, n_units // self.h, n_querys, n_keys))

        pre_c = reshaped_a * pseudo_batch_value
        c = F.sum(pre_c, axis=3)  # (b * h, units // h, n_querys)
        c = F.concat(F.split_axis(c, self.h, axis=0), axis=1)
        return c


class EncoderLayer(chainer.Chain):
    def __init__(self, n_units, h=8, dropout=0.1, nopad=False):
        n_inner_units = n_units * 4
        super(EncoderLayer, self).__init__(
            W_1=L.Linear(n_units, n_inner_units),
            W_2=L.Linear(n_inner_units, n_units),
            Attention=AttentionLayer(n_units, h),
            LN_1=L.LayerNormalization(n_units),
            LN_2=L.LayerNormalization(n_units),
        )
        self.dropout = dropout

    def __call__(self, x, xx_mask):
        x = x + F.dropout(self.Attention(x, x, xx_mask),
                          ratio=self.dropout)
        x = self.LN_1(x)
        x = x + F.dropout(seq_func(self.W_2, F.relu(seq_func(self.W_1, x))),
                          ratio=self.dropout)
        x = self.LN_2(x)
        return x


class DecoderLayer(chainer.Chain):
    def __init__(self, n_units, h=8, dropout=0.1, nopad=False):
        n_inner_units = n_units * 4
        super(DecoderLayer, self).__init__(
            W_1=L.Linear(n_units, n_inner_units),
            W_2=L.Linear(n_inner_units, n_units),
            Attention=AttentionLayer(n_units, h),
            SelfAttention=AttentionLayer(n_units, h),
            LN_1=L.LayerNormalization(n_units),
            LN_2=L.LayerNormalization(n_units),
            LN_3=L.LayerNormalization(n_units),
        )
        self.dropout = dropout

    def __call__(self, x, source, xy_mask, yy_mask):
        x = x + F.dropout(self.SelfAttention(x, x, yy_mask),
                          ratio=self.dropout)
        x = self.LN_1(x)
        x = x + F.dropout(self.Attention(x, source, xy_mask),
                          ratio=self.dropout)
        x = self.LN_2(x)
        x = x + F.dropout(seq_func(self.W_2, F.relu(seq_func(self.W_1, x))),
                          ratio=self.dropout)
        x = self.LN_3(x)
        return x


class Encoder(chainer.Chain):
    def __init__(self, n_layers, n_units, h=8, dropout=0.1):
        super(Encoder, self).__init__()
        links = [('l{}'.format(i + 1),
                  EncoderLayer(n_units, h=h, dropout=dropout))
                 for i in range(n_layers)]
        for link in links:
            self.add_link(*link)
        self.layer_names = [name for name, _ in links]

    def __call__(self, x, ex_mask, xx_mask):
        for name in self.layer_names:
            x = getattr(self, name)(x, xx_mask)
        x *= ex_mask
        return x


class Decoder(chainer.Chain):
    def __init__(self, n_layers, n_units, h=8, dropout=0.1):
        super(Decoder, self).__init__()
        links = [('l{}'.format(i + 1),
                  DecoderLayer(n_units, h=h, dropout=dropout))
                 for i in range(n_layers)]
        for link in links:
            self.add_link(*link)
        self.layer_names = [name for name, _ in links]

    def __call__(self, x, source, ey_mask, xy_mask, yy_mask):
        for name in self.layer_names:
            x = getattr(self, name)(x, source, xy_mask, yy_mask)
        x *= ey_mask
        return x


def make_position_encoding(xp, batch, length, n_units):
    # TODO: we can memoize as (1, n_units, max_len) at least
    assert(n_units % 2 == 0)
    position_block = xp.broadcast_to(
        xp.arange(length)[None, None, :],
        (batch, n_units // 2, length)).astype('f')
    unit_block = xp.broadcast_to(
        xp.arange(n_units // 2)[None, :, None],
        (batch, n_units // 2, length)).astype('f')
    rad_block = position_block / 10000. ** (2. * unit_block / n_units)
    sin_block = xp.sin(rad_block)
    cos_block = xp.cos(rad_block)
    emb_block = xp.concatenate([sin_block, cos_block], axis=1)
    return emb_block

# TODO: remove eos?


class Seq2seq(chainer.Chain):

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units,
                 h=8, dropout=0.1):
        init = chainer.initializers.HeNormal(scale=0.5**2)
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

        # Encode Positions
        max_len = max(x_length, y_length)
        p_block = make_position_encoding(self.xp, batch, max_len, self.n_units)
        ex_block += p_block[:, :, :x_length]
        ey_block += p_block[:, :, :y_length]

        ex_block = F.dropout(ex_block, ratio=self.dropout)
        ey_block = F.dropout(ey_block, ratio=self.dropout)

        # Encode Sources
        ex_mask = self.xp.broadcast_to(
            x_block[:, None, :] >= 0, ex_block.shape)
        # (batch, x_length, n_units)
        xx_mask = (x_block[:, None, :] >= 0) * \
            (x_block[:, :, None] >= 0)
        # (batch, x_length, x_length)
        z_block = self.encoder(ex_block, ex_mask, xx_mask)

        # Encode Targets with Sources (Decode without Output)
        ey_mask = self.xp.broadcast_to(
            y_in_block[:, None, :] >= 0, ey_block.shape)
        # (batch, y_length, n_units)
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

        h_block = self.decoder(
            ey_block, z_block, ey_mask, xy_mask, yy_mask)
        assert(h_block.shape == (batch, self.n_units, y_length))

        if get_prediction:
            pred_tail = F.linear(
                F.dropout(h_block[:, :, -1], ratio=self.dropout),
                self.embed_y.W)
            return pred_tail
        else:
            # Output (all together at once for efficiency)
            concat_h_block = F.transpose(h_block, (0, 2, 1)).reshape(
                (batch * y_length, self.n_units))
            concat_h_block = F.dropout(concat_h_block, ratio=self.dropout)
            concat_pred_block = F.linear(
                concat_h_block, self.embed_y.W)

            # Calculate Loss, Accuracy, Perplexity
            concat_y_out_block = y_out_block.reshape((batch * y_length))
            loss = F.softmax_cross_entropy(
                concat_pred_block, concat_y_out_block, reduce='mean')
            accuracy = F.accuracy(
                concat_pred_block, concat_y_out_block, ignore_label=-1)
            perp = self.xp.exp(loss.data)

            # Report the Values
            reporter.report({'loss': loss.data,
                             'acc': accuracy.data,
                             'perp': perp}, self)
            return loss

    def translate(self, x_block, max_length=50):
        # TODO: efficient inference by re-using convolution result
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                # if isinstance(x_block, list):
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
