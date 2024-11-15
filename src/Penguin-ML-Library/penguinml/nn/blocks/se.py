import torch
from torch import nn


class SEModule2d(nn.Module):
    """Squeeze-and-Excitation Module (2D)
    GAPしてMLPを通した重みでチャネルを強調する
    """

    def __init__(self, channel: int, reduction: int = 16):
        """
        Parameters
        ----------
        channel : int
            入力のチャネル数
        reduction : int, optional
            ボトルネックの削減率, by default 16
        """

        super(SEModule2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEModule1d(nn.Module):
    """Squeeze-and-Excitation Module (1D)
    GAPしてMLPを通した重みでチャネルを強調する
    """

    def __init__(self, channel: int, reduction: int = 8):
        """
        Parameters
        ----------
        channel : int
            入力のチャネル数
        reduction : int, optional
            ボトルネックの削減率, by default 8
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
