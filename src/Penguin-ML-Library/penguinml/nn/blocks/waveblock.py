import torch
from torch import nn
from torch.nn import functional as F


class WaveBlock(nn.Module):
    """Waveblock
    from https://www.kaggle.com/hanjoonchoe/wavenet-lstm-pytorch-ignite-ver
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation_rates: int,
        residual: bool = False,
    ):
        """
        Parameters
        ----------
        in_channels : int
            入力のチャネル数
        out_channels : int
            出力のチャネル数
        dilation_rates : int
            dilation rateの数
            ex) 4なら、1, 2, 4, 8の4つのdilation rateを持つ
        residual : bool, optional
            residual connectionをするかどうか, by default False
        """

        super(WaveBlock, self).__init__()
        self.num_rates = dilation_rates
        self.residual = residual
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2**i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=dilation_rate,
                    dilation=dilation_rate,
                )
            )
            self.gate_convs.append(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=dilation_rate,
                    dilation=dilation_rate,
                )
            )
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward

        Parameters
        ----------
        x : torch.Tensor
            入力, shape: (bs, in_channels, seq_len)

        Returns
        -------
        torch.Tensor
            出力, shape: (bs, out_channels, seq_len)
        """
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = F.tanh(self.filter_convs[i](x)) * F.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            if self.residual:
                x += res
            res = torch.add(res, x)
        return res
