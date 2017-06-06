# encoding: utf-8

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import reporter

from seq2seq import source_pad_concat_convert
from subfuncs import gradient_multiplier
from weight_normalization import weight_normalization as WN

scale05 = 0.5 ** 0.5


def sentence_block_embed(embed, x):
    batch, length = x.shape
    e = embed(x.reshape((batch * length, )))
    # (batch * length, units)
    e = F.transpose(F.stack(F.split_axis(e, batch, axis=0), axis=0), (0, 2, 1))
    # (batch, units, length)
    return e


def seq_linear(linear, x):
    batch, units, length, _ = x.shape
    h = linear(F.transpose(x, (0, 2, 1, 3)).reshape(batch * length, units))
    return F.transpose(h.reshape((batch, length, units, 1)), (0, 2, 1, 3))


class VarInNormal(chainer.initializer.Initializer):

    """Initializes array with root-scaled Gaussian distribution.

    Each element of the array is initialized by the value drawn
    independently from Gaussian distribution whose mean is 0,
    and standard deviation is
    :math:`\\sqrt{\\frac{scale}{fan_{in}}}`,
    where :math:`fan_{in}` is the number of input units.

    Args:
        scale (float): A constant that determines the scale
            of the variance.
        dtype: Data type specifier.

    """

    def __init__(self, scale=1.0, dtype=None):
        self.scale = scale
        super(VarInNormal, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype
        fan_in, fan_out = chainer.initializer.get_fans(array.shape)
        s = np.sqrt(self.scale / fan_in)
        chainer.initializers.normal.Normal(s)(array)


class ConvGLU(chainer.Chain):
    def __init__(self, n_units, width=3, dropout=0.2, nopad=False):
        init_conv = VarInNormal(4. * (1. - dropout))
        super(ConvGLU, self).__init__(
            conv=WN.convert_with_weight_normalization(
                L.Convolution2D,
                n_units, 2 * n_units,
                ksize=(width, 1),
                stride=(1, 1),
                pad=(width // 2 * (1 - nopad), 0),
                initialW=init_conv)
        )
        self.dropout = dropout

    def __call__(self, x, mask=None):
        x = F.dropout(x, ratio=self.dropout)
        out, pregate = F.split_axis(self.conv(x), 2, axis=1)
        out = out * F.sigmoid(pregate)
        if mask is not None:
            out *= mask
        return out

# TODO: For layers whose output is not directly fed to a gated linear
# unit, we initialize weights from N (0, p 1/nl) where nl is the number of
# input connections for each neuron.


class ConvGLUEncoder(chainer.Chain):
    def __init__(self, n_layers, n_units, width=3, dropout=0.2):
        super(ConvGLUEncoder, self).__init__()
        links = [('l{}'.format(i + 1),
                  ConvGLU(n_units, width=width, dropout=dropout))
                 for i in range(n_layers)]
        for link in links:
            self.add_link(*link)
        self.conv_names = [name for name, _ in links]

    def __call__(self, x, mask=None):
        for name in self.conv_names:
            x = x + getattr(self, name)(x, mask)
            x *= scale05
        return x


class ConvGLUDecoder(chainer.Chain):
    def __init__(self, n_layers, n_units, width=3, dropout=0.2):
        super(ConvGLUDecoder, self).__init__()
        links = [('l{}'.format(i + 1),
                  ConvGLU(n_units, width=width,
                          dropout=dropout, nopad=True))
                 for i in range(n_layers)]
        for link in links:
            self.add_link(*link)
        self.conv_names = [name for name, _ in links]
        self.width = width

        init_preatt = VarInNormal(1.)
        links = [('preatt{}'.format(i + 1),
                  L.Linear(n_units, n_units, initialW=init_preatt))
                 for i in range(n_layers)]
        for link in links:
            self.add_link(*link)
        self.preatt_names = [name for name, _ in links]

    def __call__(self, x, z, ze, mask, conv_mask):
        att_scale = self.xp.sum(
            mask, axis=2, keepdims=True)[:, None, :, :] ** 0.5
        pad = self.xp.zeros(
            (x.shape[0], x.shape[1], self.width - 1, 1), dtype=x.dtype)
        base_x = x
        z = F.squeeze(z, axis=3)
        # conv_mask = mask[:, :, 0][:, None, :, None]
        # (batch, 1, dec_l, 1)
        # Note: these behaviors of input, output, and attention result
        # may refer to the code by authors, which looks little different
        # from the paper's saying.
        for conv_name, preatt_name in zip(self.conv_names, self.preatt_names):
            out = getattr(self, conv_name)(
                F.concat([pad, x], axis=2), conv_mask)

            preatt = seq_linear(getattr(self, preatt_name), out)
            query = base_x + preatt
            query = F.squeeze(query, axis=3)
            c = self.attend(query, z, ze, mask) * att_scale
            x = (x + (c + out) * scale05) * scale05

        return x

    def attend(self, query, key, value, mask, minfs=None):
        # TODO reshape
        # (b, units, dec_xl) (b, units, enc_l) (b, units, dec_l, enc_l)
        pre_a = F.batch_matmul(query, key, transa=True)
        # (b, dec_xl, enc_l)
        minfs = self.xp.full(pre_a.shape, -np.inf, pre_a.dtype) \
            if minfs is None else minfs
        pre_a = F.where(mask, pre_a, minfs)
        a = F.softmax(pre_a, axis=2)
        # if values in axis=2 are all -inf, they become nan. thus do re-mask.
        a = F.where(self.xp.isnan(a.data),
                    self.xp.zeros(a.shape, dtype=a.dtype), a)
        reshaped_a = a[:, None]
        # (b, 1, dec_xl, enc_l)
        pre_c = F.broadcast_to(reshaped_a, value.shape) * value
        c = F.sum(pre_c, axis=3, keepdims=True)
        # (b, units, dec_xl, 1)
        return c


class Seq2seq(chainer.Chain):

    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units,
                 max_length=128, dropout=0.2):
        init_emb = chainer.initializers.Normal(0.1)
        init_out = VarInNormal(1.)
        super(Seq2seq, self).__init__(
            embed_x=L.EmbedID(n_source_vocab, n_units, ignore_label=-1,
                              initialW=init_emb),
            embed_y=L.EmbedID(n_target_vocab, n_units, ignore_label=-1,
                              initialW=init_emb),
            embed_position_x=L.EmbedID(max_length, n_units,
                                       initialW=init_emb),
            embed_position_y=L.EmbedID(max_length, n_units,
                                       initialW=init_emb),
            encoder=ConvGLUEncoder(n_layers, n_units, 3, dropout),
            decoder=ConvGLUDecoder(n_layers, n_units, 3, dropout),
            W=L.Linear(n_units, n_target_vocab, initialW=init_out),
        )
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_target_vocab = n_target_vocab
        self.max_length = max_length
        self.dropout = dropout

    def __call__(self, x_block, y_in_block, y_out_block, get_prediction=False):
        batch, x_length = x_block.shape
        batch, y_length = y_in_block.shape

        # Embed
        ex_block = sentence_block_embed(self.embed_x, x_block)
        ey_block = sentence_block_embed(self.embed_y, y_in_block)
        max_len = max(x_length, y_length)
        position_block = self.xp.broadcast_to(
            self.xp.clip(
                self.xp.arange(max_len), 0, self.max_length - 1)[None, ],
            (batch, max_len)).astype('i')

        px_block = sentence_block_embed(
            self.embed_position_x, position_block[:, :x_length])
        py_block = sentence_block_embed(
            self.embed_position_y, position_block[:, :y_length])
        ex_block += px_block
        ey_block += py_block

        # Encode
        ex_mask = self.xp.broadcast_to(
            x_block[:, None, :, None] >= 0, ex_block[:, :, :, None].shape)
        z_block = self.encoder(ex_block[:, :, :, None], ex_mask)

        # Prepare attention
        z_block = gradient_multiplier(z_block, 1. / self.n_layers / 2)
        ze_block = F.broadcast_to(
            F.transpose(
                (z_block + ex_block[:, :, :, None]) * scale05, (0, 1, 3, 2)),
            (batch, self.n_units, y_length, x_length))
        z_mask = (x_block[:, None, :] >= 0) * \
            (y_in_block[:, :, None] >= 0)

        # Decode (target-encode before output)
        ey_mask = self.xp.broadcast_to(
            y_in_block[:, None, :, None] >= 0, ey_block[:, :, :, None].shape)
        h_block = self.decoder(ey_block[:, :, :, None],
                               z_block, ze_block, z_mask, ey_mask)
        h_block = F.squeeze(h_block, axis=3)

        # It is faster to concatenate data before calculating loss
        # because only one matrix multiplication is called.
        assert(h_block.shape == (batch, self.n_units, y_length))

        if get_prediction:
            """
            concat_h_block = F.transpose(h_block, (0, 2, 1)).reshape(
                (batch * y_length, self.n_units))
            concat_h_block = F.dropout(concat_h_block, ratio=self.dropout)
            concat_pred_block = self.W(concat_h_block)

            pred_block = concat_pred_block.reshape(
                (batch, y_length, self.n_target_vocab))
            return pred_block
            """
            pred_tail = self.W(
                F.dropout(h_block[:, :, -1], ratio=self.dropout))
            return pred_tail
        else:
            concat_h_block = F.transpose(h_block, (0, 2, 1)).reshape(
                (batch * y_length, self.n_units))
            concat_h_block = F.dropout(concat_h_block, ratio=self.dropout)
            concat_pred_block = self.W(concat_h_block)

            concat_y_out_block = y_out_block.reshape((batch * y_length))
            loss = F.softmax_cross_entropy(
                concat_pred_block, concat_y_out_block, reduce='mean')
            accuracy = F.accuracy(
                concat_pred_block, concat_y_out_block, ignore_label=-1)
            perp = self.xp.exp(loss.data)
            rep = {'loss': loss.data, 'acc': accuracy.data, 'perp': perp}
            reporter.report(rep, self)
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
                    """
                    log_prob_block = self(x_block, y_block, y_block,
                                          get_prediction=True)
                    log_prob_tail = log_prob_block[:, -1, :]
                    """
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
