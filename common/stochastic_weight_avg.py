"""
Overwrite the Stochastic Weight Averaging Callback for auto SWA.
"""
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import nn, Tensor

import lightning.pytorch as pl
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.strategies.fsdp import FSDPStrategy
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from common.callbacks import TrainMetricsCallback
from utils.log_utils import log

_AVG_FN = Callable[[Tensor, Tensor, Tensor], Tensor]


class CscStochasticWeightAveraging(Callback):
    def __init__(
            self,
            train_metrics: TrainMetricsCallback,
            device: Optional[Union[torch.device, str]] = torch.device("cpu"),
    ):
        if device is not None and not isinstance(device, (torch.device, str)):
            raise MisconfigurationException(f"device is expected to be a torch.device or a str. Found {device}")

        self._train_metrics = train_metrics

        self.n_averaged: Optional[Tensor] = None
        self._swa_epoch_start = 999999
        self._avg_fn = self.avg_fn
        self._device = device
        self._model_contains_batch_norm: Optional[bool] = None
        self._average_model: Optional["pl.LightningModule"] = None
        self._initialized = False
        self._scheduler_state: Optional[Dict] = None
        self._init_n_averaged = 0
        self._latest_update_epoch = -1
        self.momenta: Dict[nn.modules.batchnorm._BatchNorm, Optional[float]] = {}
        self._max_epochs: int

    @property
    def swa_start(self) -> int:
        assert isinstance(self._swa_epoch_start, int)
        return max(self._swa_epoch_start - 1, 0)  # 0-based

    @property
    def swa_end(self) -> int:
        return self._max_epochs - 1  # 0-based

    @staticmethod
    def pl_module_contains_batch_norm(pl_module: "pl.LightningModule") -> bool:
        return any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in pl_module.modules())

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
        if isinstance(trainer.strategy, (FSDPStrategy, DeepSpeedStrategy)):
            raise MisconfigurationException("SWA does not currently support sharded models.")

        # copy the model before moving it to accelerator device.
        self._average_model = deepcopy(pl_module)

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if len(trainer.optimizers) != 1:
            raise MisconfigurationException("SWA currently works with 1 `optimizer`.")

        if len(trainer.lr_scheduler_configs) > 1:
            raise MisconfigurationException("SWA currently not supported for more than 1 `lr_scheduler`.")

        assert trainer.max_epochs is not None

        self._model_contains_batch_norm = self.pl_module_contains_batch_norm(pl_module)

        self._max_epochs = trainer.max_epochs
        if self._model_contains_batch_norm:
            # virtually increase max_epochs to perform batch norm update on latest epoch.
            assert trainer.fit_loop.max_epochs is not None
            trainer.fit_loop.max_epochs += 1

        if self._scheduler_state is not None:
            self._clear_schedulers(trainer)

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._initialized:
            # The swa started from the epoch when the validation f1 score is starting decrease.
            if len(self._train_metrics.val_f1_list) > 0 and \
                    max(self._train_metrics.val_f1_list) > self._train_metrics.val_f1_list[-1]:
                self._swa_epoch_start = trainer.current_epoch

        if (not self._initialized) and (self.swa_start <= trainer.current_epoch <= self.swa_end):
            log.info("Start SWA...")
            self._initialized = True

            # move average model to request device.
            assert self._average_model is not None
            self._average_model = self._average_model.to(self._device or pl_module.device)

            assert trainer.max_epochs is not None

            if self.n_averaged is None:
                self.n_averaged = torch.tensor(self._init_n_averaged, dtype=torch.long, device=pl_module.device)

        if (self.swa_start <= trainer.current_epoch <= self.swa_end) and (
                trainer.current_epoch > self._latest_update_epoch
        ):
            assert self.n_averaged is not None
            assert self._average_model is not None
            self.update_parameters(self._average_model, pl_module, self.n_averaged, self._avg_fn)
            self._latest_update_epoch = trainer.current_epoch

        # Note: No > here in case the callback is saved with the model and training continues
        if trainer.current_epoch == self.swa_end + 1:
            # Transfer weights from average model to pl_module
            assert self._average_model is not None
            self.transfer_weights(self._average_model, pl_module)

            # Reset BatchNorm for update
            self.reset_batch_norm_and_save_state(pl_module)

            # There is no need to perform either backward or optimizer.step as we are
            # performing only one pass over the train data-loader to compute activation statistics
            # Therefore, we will virtually increase the number of training batches by 1 and skip backward.
            trainer.fit_loop.max_batches += 1
            trainer.fit_loop._skip_backward = True
            self._accumulate_grad_batches = trainer.accumulate_grad_batches
            assert isinstance(trainer.fit_loop.max_batches, int), "Iterable-style datasets are not supported"
            trainer.accumulate_grad_batches = trainer.fit_loop.max_batches

    def on_train_epoch_end(self, trainer: "pl.Trainer", *args: Any) -> None:
        trainer.fit_loop._skip_backward = False

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # the trainer increases the current epoch before this hook is called
        if self._model_contains_batch_norm and trainer.current_epoch - 1 == self.swa_end + 1:
            # BatchNorm epoch update. Reset state
            trainer.accumulate_grad_batches = self._accumulate_grad_batches
            trainer.fit_loop.max_batches -= 1
            assert trainer.fit_loop.max_epochs is not None
            trainer.fit_loop.max_epochs -= 1
            self.reset_momenta()
        elif trainer.current_epoch - 1 == self.swa_end:
            # Last SWA epoch. Transfer weights from average model to pl_module
            assert self._average_model is not None
            self.transfer_weights(self._average_model, pl_module)

    @staticmethod
    def transfer_weights(src_pl_module: "pl.LightningModule", dst_pl_module: "pl.LightningModule") -> None:
        for src_param, dst_param in zip(src_pl_module.parameters(), dst_pl_module.parameters()):
            dst_param.detach().copy_(src_param.to(dst_param.device))

    def reset_batch_norm_and_save_state(self, pl_module: "pl.LightningModule") -> None:
        """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L140-L154."""
        self.momenta = {}
        for module in pl_module.modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                continue
            assert module.running_mean is not None
            module.running_mean = torch.zeros_like(
                module.running_mean,
                device=pl_module.device,
                dtype=module.running_mean.dtype,
            )
            assert module.running_var is not None
            module.running_var = torch.ones_like(
                module.running_var,
                device=pl_module.device,
                dtype=module.running_var.dtype,
            )
            self.momenta[module] = module.momentum
            module.momentum = None  # type: ignore[assignment]
            assert module.num_batches_tracked is not None
            module.num_batches_tracked *= 0

    def reset_momenta(self) -> None:
        """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L164-L165."""
        for bn_module in self.momenta:
            bn_module.momentum = self.momenta[bn_module]  # type: ignore[assignment]

    @staticmethod
    def update_parameters(
            average_model: "pl.LightningModule", model: "pl.LightningModule", n_averaged: Tensor, avg_fn: _AVG_FN
    ) -> None:
        """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L104-L112."""
        for p_swa, p_model in zip(average_model.parameters(), model.parameters()):
            device = p_swa.device
            p_swa_ = p_swa.detach()
            p_model_ = p_model.detach().to(device)
            src = p_model_ if n_averaged == 0 else avg_fn(p_swa_, p_model_, n_averaged.to(device))
            p_swa_.copy_(src)
        n_averaged += 1

    @staticmethod
    def avg_fn(averaged_model_parameter: Tensor, model_parameter: Tensor, num_averaged: Tensor) -> Tensor:
        """Adapted from https://github.com/pytorch/pytorch/blob/v1.7.1/torch/optim/swa_utils.py#L95-L97."""
        return averaged_model_parameter + (model_parameter - averaged_model_parameter) / (num_averaged + 1)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "n_averaged": 0 if self.n_averaged is None else self.n_averaged.item(),
            "latest_update_epoch": self._latest_update_epoch,
            "average_model_state": None if self._average_model is None else self._average_model.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._init_n_averaged = state_dict["n_averaged"]
        self._latest_update_epoch = state_dict["latest_update_epoch"]
        self._load_average_model_state(state_dict["average_model_state"])

    @staticmethod
    def _clear_schedulers(trainer: "pl.Trainer") -> None:
        # If we have scheduler state saved, clear the scheduler configs so that we don't try to
        # load state into the wrong type of schedulers when restoring scheduler checkpoint state.
        # We'll configure the scheduler and re-load its state in on_train_epoch_start.
        # Note that this relies on the callback state being restored before the scheduler state is
        # restored, and doesn't work if restore_checkpoint_after_setup is True, but at the time of
        # writing that is only True for deepspeed which is already not supported by SWA.
        # See https://github.com/Lightning-AI/lightning/issues/11665 for background.
        if trainer.lr_scheduler_configs:
            assert len(trainer.lr_scheduler_configs) == 1
            trainer.lr_scheduler_configs.clear()

    def _load_average_model_state(self, model_state: Any) -> None:
        if self._average_model is None:
            return
        self._average_model.load_state_dict(model_state)
