from typing import Literal, Tuple

import torch
from torch import nn


class MLP(nn.Module):
    """MLP
    (Linear -> Norm -> Activation -> Dropout) * N
    """

    def __init__(
        self,
        dims: Tuple[int],
        activation: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
        norm_type: Literal["bn", "ln"] = "bn1d",
        output_norm_act: bool = True,
    ):
        """MLP

        Parameters
        ----------
        dims : Tuple[int]
            各層の次元数. (入力層, 中間層1, 中間層2, ..., 出力層)
        activation : nn.Module, optional
            活性化関数, by default nn.ReLU()
        dropout : float, optional
            ドロップアウト率, by default 0.0
        norm_type : str, optional
            正規化の種類, by default "bn"
        output_norm_act : bool, optional
            出力層の正規化と活性化を行うかどうか, by default True
        """

        super().__init__()
        self.dims = dims
        self.activation = activation
        self.dropout = dropout
        self.norm_type = norm_type
        self.output_norm = output_norm_act

        self.layers = nn.ModuleList()

        # 中間層
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # linear
            self.layers.append(nn.Linear(in_dim, out_dim))

            # normalization -> activation -> dropout
            if i < len(dims) - 2:
                if self.norm_type == "bn":
                    self.layers.append(nn.BatchNorm1d(out_dim))
                elif self.norm_type == "ln":
                    self.layers.append(nn.LayerNorm(out_dim))
                else:
                    raise ValueError(f"Unknown norm type: {self.norm_type}")

                self.layers.append(self.activation)
                self.layers.append(nn.Dropout(self.dropout))

        # 出力層
        if self.output_norm:
            if self.norm_type == "bn":
                self.layers.append(nn.BatchNorm1d(dims[-1]))
            elif self.norm_type == "ln":
                self.layers.append(nn.LayerNorm(dims[-1]))
            else:
                raise ValueError(f"Unknown norm type: {self.norm_type}")
            self.layers.append(self.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
