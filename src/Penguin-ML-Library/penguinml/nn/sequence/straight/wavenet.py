from typing import Tuple

import torch
from torch import nn

from ...blocks.waveblock import WaveBlock
from .base import StraightSeqModel


class WaveRNNModel(StraightSeqModel):
    """WaveRNNModel
    MLP -> WaveBlock -> RNN -> MLP
    """

    def __init__(
        self,
        in_mlp_dims: Tuple[int] = (2, 64, 84),
        wave_chanels: Tuple[int] = (16, 32, 64),
        wave_dilation_rates: Tuple[int] = (8, 4, 2, 1),
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
        wave_chanels : Tuple[int], optional
            WaveBlockのチャネル数, by default (16, 32, 64)
        wave_dilation_rates : Tuple[int], optional
            WaveBlockのdilation rate, by default (8, 4, 2, 1)
            NOTE: `len(wave_chanels) == len(wave_dilation_rates) + 1`
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
        in_chans = [d_model, *wave_chanels]
        out_chans = [*wave_chanels, d_model]
        self.waveblocks = nn.ModuleList()
        for in_ch, out_ch, dilation_rate in zip(in_chans, out_chans, wave_dilation_rates):
            self.waveblocks.append(WaveBlock(in_ch, out_ch, dilation_rate))

    def bottleneck_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        for waveblock in self.waveblocks:
            x = waveblock(x)
        x = x.permute(0, 2, 1)
        return x
