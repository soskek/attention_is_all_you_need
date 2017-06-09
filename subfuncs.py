import numpy

import chainer
from chainer import cuda
from chainer import function
from chainer import reporter
from chainer.training import util
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


class FailBestValueTrigger(object):

    """Trigger invoked when specific value fails to become best.

    Args:
        key (str): Key of value.
        compare (function): Compare function which takes current best value and
            new value and returns whether new value is better than current
            best.
        trigger: Trigger that decides the comparison interval between current
            best value and new value. This must be a tuple in the form of
            ``<int>, 'epoch'`` or ``<int>, 'iteration'`` which is passed to
            :class:`~chainer.training.triggers.IntervalTrigger`.

    """

    def __init__(self, key, compare, trigger=(1, 'epoch'), print_triger=False):
        self._key = key
        self._best_value = None
        self._interval_trigger = util.get_trigger(trigger)
        self._init_summary()
        self._compare = compare
        self._print_triger = print_triger

    def __call__(self, trainer):
        """Decides whether the extension should be called on this iteration.

        Args:
            trainer (~chainer.training.Trainer): Trainer object that this
                trigger is associated with. The ``observation`` of this trainer
                is used to determine if the trigger should fire.

        Returns:
            bool: ``True`` if the corresponding extension should be invoked in
                this iteration.

        """

        observation = trainer.observation
        summary = self._summary
        key = self._key
        if key in observation:
            summary.add({key: observation[key]})

        if not self._interval_trigger(trainer):
            return False

        stats = summary.compute_mean()
        value = float(stats[key])  # copy to CPU
        self._init_summary()

        if self._best_value is None or self._compare(self._best_value, value):
            self._best_value = value
            return False
        return True

    def _init_summary(self):
        self._summary = reporter.DictSummary()


class FailMaxValueTrigger(FailBestValueTrigger):
    def __init__(self, key, trigger=(1, 'epoch')):
        super(FailMaxValueTrigger, self).__init__(
            key, lambda max_value, new_value: new_value > max_value, trigger)


class FailMinValueTrigger(FailBestValueTrigger):
    def __init__(self, key, trigger=(1, 'epoch')):
        super(FailMinValueTrigger, self).__init__(
            key, lambda min_value, new_value: new_value < min_value, trigger)
