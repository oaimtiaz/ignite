import math
import numpy as np
import torch

from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from torch import nn


class ECE(Metric):
    def __init__(self, num_bins=10, **kwargs):
        self._num_bins = num_bins
        self._true_positive, self._sample_space = np.zeros([num_bins]), np.zeros([num_bins])

    @reinit__is_reduced
    def update(self, output=None, target=None):

        # Converting output and target into tuples
        output = torch.from_numpy(np.asarray(output))
        target = torch.from_numpy(np.asarray(target))

        # Using a sigmoid to get a probabilistic representation
        sig = nn.Sigmoid()
        output = sig(output)

        # Allows results of 1 to be put into bins
        output = torch.clamp(output, 0, 0.9999)

        for pred, t in zip(output, target):
            conf, p_cls = pred.max(), pred.argmax()
            bin_id = int(math.floor(conf * self._num_bins))

            # Placing samples into bins
            self._sample_space[bin_id] += 1
            self._true_positive[bin_id] += int(p_cls == t)

    # Accuracy
    def _acc(self):
        return self._true_positive / np.maximum(1, self._sample_space)

    @reinit__is_reduced
    def reset(self):
        self._true_positive = np.zeros([self._num_bins])
        self._sample_space = np.zeros([self._num_bins])
        super(ECE, self).reset()

    @sync_all_reduce("num_examples", "num_bins")
    def compute(self):
        n = self._sample_space.sum()
        bin_confs = np.linspace(0, 1, self._num_bins)
        return ((self._sample_space / n) * np.abs(self._acc() - bin_confs)).sum()