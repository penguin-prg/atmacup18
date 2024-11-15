from typing import Any, List

import albumentations as A
import cv2
import numpy as np
import torch
from torch import nn


class SimpleImageDataset(torch.utils.data.Dataset):
    """画像データセット"""

    def __init__(
        self,
        paths: List[str],
        labels: List[Any],
        aug: List[nn.Module] = [],
    ):
        """
        Parameters
        ----------
        paths : List[str]
            画像のパスのリスト.
        labels : List[Any]
            ラベルのリスト
        aug : List[nn.Module], optional
            augmentations, by default []
            ex) `[A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)]`
        """
        self.paths = paths
        self.labels = labels
        self.aug = A.Compose(aug)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        """
        Parameters
        ----------
        index : int
            インデックス

        Returns
        -------
        image : np.ndarray
            画像
        label : int
            ラベル
        """
        # load image
        image = cv2.imread(self.paths[index])  # HWC
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB

        # augmentations
        image = self.aug(image=image)["image"]

        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = image / 255.0
        label = self.labels[index]

        return image.astype(np.float32), label
