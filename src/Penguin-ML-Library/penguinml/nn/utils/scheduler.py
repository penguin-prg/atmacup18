from typing import Dict, Union

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)


def get_scheduler(
    scheduler_cfg: Dict[str, Union[int, float, str]],
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler:
    """スケジューラを取得する関数

    Parameters
    ----------
    scheduler_cfg : Dict[str, Union[int, float, str]]
        schedulerの設定
    optimizer : torch.optim.Optimizer
        optimizer

    Returns
    -------
    scheduler : torch.optim.lr_scheduler
        scheduler

    Examples
    --------
    >>> # ReduceLROnPlateau
    >>> scheduler_cfg = {
    >>>     "scheduler": "ReduceLROnPlateau",
    >>>     "factor": 0.1,
    >>>     "patience": 2, # [epoch]
    >>>     "eps": 1e-6,
    >>> }

    >>> # CosineAnnealingLR
    >>> scheduler_cfg = {
    >>>     "scheduler": "CosineAnnealingLR",
    >>>     "max_epoch": 10,
    >>>     "min_lr": 1e-6,
    >>> }

    >>> # CosineAnnealingWarmRestarts
    >>> scheduler_cfg = {
    >>>     "scheduler": "CosineAnnealingWarmRestarts",
    >>>     "restart_epoch": 10,
    >>>     "min_lr": 1e-6,
    >>> }
    """
    if scheduler_cfg["scheduler"] == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_cfg["factor"],
            patience=scheduler_cfg["patience"],
            verbose=True,
            eps=scheduler_cfg["eps"],
        )
    elif scheduler_cfg["scheduler"] == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=scheduler_cfg["max_epoch"],
            eta_min=scheduler_cfg["min_lr"],
            last_epoch=-1,
        )
    elif scheduler_cfg["scheduler"] == "CosineAnnealingWarmRestarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_cfg["restart_epoch"],
            T_mult=1,
            eta_min=scheduler_cfg["min_lr"],
            last_epoch=-1,
        )
    else:
        raise NotImplementedError()

    return scheduler


def step_scheduler(
    scheduler: torch.optim.lr_scheduler,
    val_loss: float,
) -> None:
    """スケジューラを1step進める関数

    Parameters
    ----------
    scheduler : torch.optim.lr_scheduler
        scheduler
    val_loss : float
        評価データでのloss
    """

    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_loss)
    elif isinstance(scheduler, CosineAnnealingLR):
        scheduler.step()
    elif isinstance(scheduler, CosineAnnealingWarmRestarts):
        scheduler.step()
    else:
        raise NotImplementedError()
