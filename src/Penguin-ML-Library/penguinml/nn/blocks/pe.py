import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):
    """Positional Encoding
    (bs, max_len, d_model)のテンソルに対して、Positional Encodingを行う
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 1024,
        batch_first: bool = True,
    ):
        """
        Parameters
        ----------
        d_model : int
            モデルの次元数
        dropout : float, optional
            ドロップアウト率, by default 0.1
        max_len : int, optional
            系列長, by default 1024
        batch_first : bool, optional
            入力の形式がbatch firstならTrue, by default True
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        if batch_first:
            pe = pe.transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        if self.batch_first:
            x = x + self.pe[: x.size(0), :]
        else:
            x = x + self.pe[: x.size(0)]
        return self.dropout(x)
