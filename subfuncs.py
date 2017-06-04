import numpy

import chainer
from chainer import cuda
from chainer import function
from chainer import utils
from chainer.utils import type_check


if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()
    # _mode = libcudnn.CUDNN_ACTIVATION_RELU


class GradientMultiplier(function.Function):

    """Gradient Multiplier."""

    def __init__(self, coefficient=1.):
        self.coefficient = float(coefficient)

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

    def forward_cpu(self, x):
        self.retain_inputs(())
        self.retain_outputs((0,))
        return x[0],

    def forward_gpu(self, x):
        if (chainer.should_use_cudnn('==always') and
                x[0].flags.c_contiguous and
                (_cudnn_version >= 3000 or x[0].dtype != numpy.float16)):
            self._use_cudnn = True
        else:
            self.retain_inputs(())
            self._use_cudnn = False
        self.retain_outputs((0,))
        return x[0],

    def backward_cpu(self, x, gy):
        return utils.force_array(gy[0] * self.coefficient),

    def backward_gpu(self, x, gy):
        gx = cuda.elementwise(
            'T c, T gy', 'T gx',
            'gx = gy * c',
            'gradmul_bwd')(self.coefficient, gy[0])
        return gx,


def gradient_multiplier(x, coefficient=1.):
    """Gradient Multiplier function.

    .. math:: f(x)=x. f'(x)=coefficient.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    """
    return GradientMultiplier(coefficient)(x)
