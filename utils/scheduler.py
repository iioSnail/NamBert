import warnings

from torch.optim.lr_scheduler import _LRScheduler

class WarmupExponentialLR(_LRScheduler):
    """
    Reference: https://arxiv.org/abs/1706.02677
    """

    def __init__(self, optimizer, gamma, last_epoch=-1, warmup_epochs=2, warmup_factor=1.0 / 3, verbose=False,
                 min_lr=None, **kwargs):
        self.gamma = gamma
        self.warmup_method = 'linear'
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def _get_warmup_factor_at_iter(
            self, method: str, iter: int, warmup_iters: int, warmup_factor: float
    ) -> float:
        if iter >= warmup_iters:
            return 1.0

        if method == "constant":
            return warmup_factor
        elif method == "linear":
            alpha = iter / warmup_iters
            return warmup_factor * (1 - alpha) + alpha
        else:
            raise ValueError("Unknown warmup method: {}".format(method))

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        warmup_factor = self._get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_epochs, self.warmup_factor
        )

        if self.last_epoch <= self.warmup_epochs:
            if self.min_lr is None:
                return [base_lr * warmup_factor for base_lr in self.base_lrs]
            else:
                return [base_lr * warmup_factor if base_lr * warmup_factor > self.min_lr else self.min_lr
                        for base_lr in self.base_lrs]

        if self.min_lr is None:
            return [group['lr'] * self.gamma for group in self.optimizer.param_groups]
        else:
            return [group['lr'] * self.gamma if group['lr'] * self.gamma > self.min_lr else self.min_lr
                    for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma ** self.last_epoch
                for base_lr in self.base_lrs]
