from __future__ import division

from chainer import reporter
from chainer.training import extension
from chainer.training import util


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


class VaswaniRule(extension.Extension):

    """Trainer extension to shift an optimizer attribute magically by Vaswani.

    Args:
        attr (str): Name of the attribute to shift.
        rate (float): Rate of the exponential shift. This value is multiplied
            to the attribute at each call.
        init (float): Initial value of the attribute. If it is ``None``, the
            extension extracts the attribute at the first call and uses it as
            the initial value.
        target (float): Target value of the attribute. If the attribute reaches
            this value, the shift stops.
        optimizer (~chainer.Optimizer): Target optimizer to adjust the
            attribute. If it is ``None``, the main optimizer of the updater is
            used.

    """

    def __init__(self, attr, d, warmup_steps=4000,
                 init=None, target=None, optimizer=None):
        self._attr = attr
        self._d_inv05 = d ** (-0.5)
        self._warmup_steps_inv15 = warmup_steps ** (-1.5)
        self._init = init
        self._target = target
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None

    def initialize(self, trainer):
        optimizer = self._get_optimizer(trainer)
        # ensure that _init is set
        if self._init is None:
            # self._init = getattr(optimizer, self._attr)
            self._init = self._d_inv05 * (1. * self._warmup_steps_inv15)

        if self._last_value is not None:  # resuming from a snapshot
            self._update_value(optimizer, self._last_value)
        else:
            self._update_value(optimizer, self._init)

    def __call__(self, trainer):
        self._t += 1

        optimizer = self._get_optimizer(trainer)
        value = self._d_inv05 * \
            min(self._t ** (-0.5), self._t * self._warmup_steps_inv15)
        self._update_value(optimizer, value)

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)

    def _get_optimizer(self, trainer):
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, value):
        setattr(optimizer, self._attr, value)
        self._last_value = value
