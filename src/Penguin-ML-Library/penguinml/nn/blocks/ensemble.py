from typing import Callable, List

import torch
from torch import nn


class EnsembleModel(nn.Module):
    """EnsembleModel"""

    def __init__(
        self,
        models: List[nn.Module],
        agg_fn: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
    ) -> None:
        """
        Parameters
        ----------
        models : List[nn.Module]
            モデルのリスト
        agg_fn : Callable[[torch.Tensor], torch.Tensor], optional
            集約関数, by default torch.mean
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.agg_fn = agg_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward

        Parameters
        ----------
        x : torch.Tensor
            入力

        Returns
        -------
        torch.Tensor
            出力
        """
        preds = [model(x) for model in self.models]
        preds = torch.stack(preds, dim=0)
        preds = self.agg_fn(preds, dim=0)
        return preds
