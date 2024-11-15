from abc import ABCMeta, abstractmethod
from typing import Tuple

import torch
from torch import nn

from ...blocks.mlp import MLP


class StraightSeqModel(nn.Module, metaclass=ABCMeta):
    """StraightSeqModel
    MLP -> `bottleneck` -> RNN -> MLP の基底クラス
    """

    def __init__(
        self,
        in_mlp_dims: Tuple[int] = (2, 64, 84),
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
        in_mlp_dims : Tuple[int], optional
            入力に対するMLPの次元, by default (2, 64, 84)
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
        super().__init__()

        self.input_mlp = MLP(
            in_mlp_dims,
            dropout=dropout,
            activation=act,
            norm_type="ln",
        )

        d_model = in_mlp_dims[-1]

        self.rnn = rnn_module(
            d_model,
            rnn_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.output_mlp = MLP(
            (rnn_dim * 2, *out_mlp_dims),
            dropout=dropout,
            activation=act,
            norm_type="ln",
            output_norm_act=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward

        Parameters
        ----------
        x : nn.Tensor
            入力, shape: (batch_size, seq_len, in_mlp_dims[0])

        Returns
        -------
        nn.Tensor
            出力, shape: (batch_size, seq_len, out_mlp_dims[-1])
        """
        x = self.input_mlp(x)
        x = self.bottleneck_forward(x)
        x, _ = self.rnn(x)
        x = self.output_mlp(x)
        return x

    @abstractmethod
    def bottleneck_forward(self, x: torch.Tensor) -> torch.Tensor:
        """bottleneck

        Parameters
        ----------
        x : nn.Tensor
            入力, shape: (batch_size, seq_len, in_mlp_dims[0])

        Returns
        -------
        nn.Tensor
            出力, shape: (batch_size, seq_len, in_mlp_dims[0])
        """
        raise NotImplementedError()
