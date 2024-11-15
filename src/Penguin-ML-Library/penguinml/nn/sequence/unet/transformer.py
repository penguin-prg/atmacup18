from typing import Literal

from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from ...blocks.pe import PositionalEncoding
from .base import UNetSeqModel


class UNetTransformer(UNetSeqModel):
    """UNetTransformer
    UNetDecoder -> Transformer -> UNetEncoder
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len: int,
        d_model: int = 1024,
        num_layers: int = 4,
        bilinear: bool = False,
        norm: Literal["bn", "ln"] = "bn",
        se=False,
        scale_factor: int = 2,
        trf_dim: int = 128,
        trf_num_layers: int = 4,
        trf_num_heads: int = 4,
        trf_dropout: float = 0.0,
    ):
        """
        Parameters
        ----------
        in_channels : int
            入力のチャネル数
        out_channels : int
            出力のチャネル数
        seq_len : int
            入力の系列長
        d_model : int, optional
            ボトルネックのチャネル数, by default 1024
        num_layers : int, optional
            ダウンサンプリングの回数, by default 4
        bilinear : bool, optional
            Bilinear Interpolationを使うかどうか, by default False
        norm : Literal["bn", "ln"], optional
            normalize, by default "bn"
        se : bool, optional
            Squeeze-and-Excitationを使うかどうか, by default False
        scale_factor : int, optional
            Up/Down samplingの倍率, by default 2
        trf_dim : int, optional
            Transformerの全結合層の次元数, by default 128
        trf_num_layers : int, optional
            Transformerの層数, by default 4
        trf_num_heads : int, optional
            Transformerのヘッド数, by default 4
        trf_dropout : float, optional
            Transformerのドロップアウト率, by default 0.0
        """
        super().__init__(
            in_channels,
            out_channels,
            seq_len,
            d_model,
            num_layers,
            bilinear,
            norm,
            se,
            scale_factor,
        )

        seq_len = seq_len // (scale_factor**num_layers)
        self.pe = PositionalEncoding(
            d_model,
            dropout=0.0,
            max_len=seq_len,
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

    def bottleneck(self, x: Tensor) -> Tensor:
        x = x.transpose(1, 2)  # (bs, in_chans, seq_len) -> (bs, seq_len, in_chans)
        x = self.pe(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        return x
