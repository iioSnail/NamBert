import math
import random

import numpy as np
import torch
from torch import nn


class BinaryFocalLoss(nn.Module):
    """
    Reference: https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, gamma=2.0, alpha=0.25, epsilon=1.e-9):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        multi_hot_key = target
        logits = input

        zero_hot_key = 1 - multi_hot_key
        loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        return loss.mean()


class FocalLoss(nn.Module):
    """
    参考 https://github.com/lonePatient/TorchBlocks
    """

    def __init__(self, gamma=2.0, alpha=1, epsilon=1.e-9, device=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.device = device
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha).to(device)
        else:
            self.alpha = alpha
        self.epsilon = epsilon

    def set_alpha(self, alpha):
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha).to(self.device)
        else:
            self.alpha = alpha

    def get_alpha(self):
        return self.alpha

    def forward(self, input, target):
        """
        Args:
            input: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        one_hot_key = None
        if len(target.size()) == 1:
            num_labels = input.size(-1)
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), num_labels, dtype=torch.float32, device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            one_hot_key[:, 0] = 0  # ignore 0 index.

        if len(target.size()) == 2:
            one_hot_key = target

        logits = torch.softmax(input, dim=-1)
        loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
        loss = loss.sum(1)
        return loss.mean()