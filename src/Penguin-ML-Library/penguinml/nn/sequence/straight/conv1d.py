from typing import Tuple

import torch
from torch import nn

from ...blocks.conv1d import Conv1dBlock
from .base import StraightSeqModel


class Conv1dRNNModel(StraightSeqModel):
    """Conv1dRNNModel
    MLP -> Conv1d -> RNN -> MLP
    """

    def __init__(
        self,
        in_mlp_dims: Tuple[int] = (2, 64, 84),
        conv_kernel_sizes_p1: Tuple[int] = (8, 16, 32, 64, 128),
        conv_hidden_chan: int = 16,
        rnn_dim: int = 128,
        rnn_num_layers: int = 2,
        rnn_module: nn.Module = nn.GRU,
        out_mlp_dims: Tuple[int] = (64, 2),
        act: nn.Module = nn.ReLU(),
        dropout: float = 0.0,
    ):
        """
        Parameters
        ----------
        dropout : float, optional
            dropout, by default 0.0
        in_mlp_dims : Tuple[int], optional
            入力に対するMLPの次元, by default (2, 64, 84)
        conv_kernel_sizes_p1 : Tuple[int], optional
            Conv1dBlockのカーネルサイズ, by default (8, 16, 32, 64, 128)
        conv_hidden_chan : int, optional
            Conv1dBlockの隠れ層の次元, by default 16
        rnn_dim : int, optional
            RNNの隠れ層の次元, by default 128
        rnn_num_layers : int, optional
            RNNの層数, by default 2
        rnn_module : nn.Module, optional
            RNNのモジュール, by default nn.GRU
        out_mlp_dims : Tuple[int], optional
            出力に対するMLPの次元, by default (64, 2)
            実際には (rnn_dim * 2, *out_mlp_dims) のMLPが生成される
        act : nn.Module, optional
            MLPの活性化関数, by default nn.ReLU()
        dropout : float, optional
            MLPのdropout, by default 0.0
        """
        super().__init__(
            in_mlp_dims,
            rnn_dim,
            rnn_num_layers,
            rnn_module,
            out_mlp_dims,
            act,
            dropout,
        )

        d_model = in_mlp_dims[-1]

        self.conv1d_blocks = nn.ModuleList(
            [Conv1dBlock(d_model, conv_hidden_chan, k - 1) for k in conv_kernel_sizes_p1]
        )
        self.last_conv = Conv1dBlock(
            d_model + conv_hidden_chan * len(conv_kernel_sizes_p1),
            d_model,
            1,
        )

    def bottleneck_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = torch.cat([conv(x) for conv in self.conv1d_blocks] + [x], dim=1)
        x = self.last_conv(x)
        x = x.permute(0, 2, 1)
        return x
