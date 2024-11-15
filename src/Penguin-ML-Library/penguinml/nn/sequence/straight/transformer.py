from typing import Tuple

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from ...blocks.pe import PositionalEncoding
from .base import StraightSeqModel


class TransformerRNNModel(StraightSeqModel):
    """TransformerRNNModel
    MLP -> Transformer -> RNN -> MLP
    """

    def __init__(
        self,
        max_len: int,
        dropout: float = 0.0,
        in_mlp_dims: Tuple[int] = (2, 64, 84),
        trf_dim: int = 128,
        trf_num_layers: int = 4,
        trf_num_heads: int = 6,
        trf_dropout: float = 0.0,
        rnn_dim: int = 128,
        rnn_num_layers: int = 2,
        rnn_module: nn.Module = nn.GRU,
        out_mlp_dims: Tuple[int] = (64, 2),
        act: nn.Module = nn.ReLU(),
    ):
        """
        Parameters
        ----------
        max_len : int
            系列長
        in_mlp_dims : Tuple[int], optional
            入力に対するMLPの次元, by default (2, 64, 84)
        trf_dim : int, optional
            Transformerのfeed forwardの次元, by default 128
        trf_num_layers : int, optional
            Transformerの層数, by default 4
        trf_num_heads : int, optional
            Transformerのヘッド数, by default 6
        trf_dropout : float, optional
            Transformerのdropout, by default 0.0
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
        self.pe = PositionalEncoding(
            d_model,
            dropout=0.0,
            max_len=max_len,
            batch_first=True,
        )
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=trf_num_heads,
                dropout=trf_dropout,
                dim_feedforward=trf_dim,
                batch_first=True,
            ),
            num_layers=trf_num_layers,
        )

    def bottleneck_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pe(x)
        x = self.transformer(x)
        return x
