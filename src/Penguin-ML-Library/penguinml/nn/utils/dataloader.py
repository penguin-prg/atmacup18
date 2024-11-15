from typing import Tuple

import numpy as np
import torch


def get_dataloaders(
    train_dataset: torch.utils.data.Dataset,
    valid_dataset: torch.utils.data.Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """データローダーを取得する

    Parameters
    ----------
    train_dataset : torch.utils.data.Dataset
        学習用データセット
    valid_dataset : torch.utils.data.Dataset
        検証用データセット
    batch_size : int, optional
        バッチサイズ, by default 32
    num_workers : int, optional
        ワーカー数, by default 4
    pin_memory : bool, optional
        データをメモリに保存するか, by default True
    seed : int, optional
        シード値, by default 42

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        学習用データローダー
    valid_loader : torch.utils.data.DataLoader
        検証用データローダー
    """
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=lambda x: np.random.seed(x + seed),
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=lambda x: np.random.seed(x + seed),
    )
    return train_loader, valid_loader
