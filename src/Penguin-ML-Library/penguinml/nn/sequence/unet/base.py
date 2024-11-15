from functools import partial
from typing import Literal

import torch
from torch import nn

from ...blocks.se import SEModule1d


class UNetSeqModel(nn.Module):
    """UNet for Sequence
    (bs, seq_len, in_chans) -> (bs, seq_len, out_chans)

    # TODO: camaroさんにあわせて [channel, seq_len]の両方で正規化してるけど、時間方向だけでもいいかも？
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
    ):
        """UNet for Sequence

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
        """

        super().__init__()

        assert d_model % (2**num_layers) == 0, "d_model must be divisible by scale_factor**num_layers"
        assert d_model >= 2**num_layers, "d_model must be greater than scale_factor**num_layers"
        assert seq_len % (scale_factor**num_layers) == 0, "seq_len must be divisible by scale_factor**num_layers"

        if bilinear:
            raise NotImplementedError()

        def get_norm(inc: int, length: int):
            if norm == "bn":
                return nn.BatchNorm1d(inc)
            elif norm == "ln":
                return nn.LayerNorm([inc, length])
            else:
                raise ValueError(f"norm must be 'bn' or 'ln', but got {norm}")

        # downsampling
        self.downs = nn.ModuleList(
            [
                DoubleConv(
                    in_channels,
                    d_model // (2**num_layers),
                    norm=partial(get_norm, length=seq_len),
                    se=se,
                    res=False,
                )
            ]
        )
        inc = d_model // (2**num_layers)
        ouc = inc * 2
        length = seq_len // scale_factor
        for _ in range(num_layers):
            self.downs.append(
                Down(
                    inc,
                    ouc,
                    scale_factor,
                    norm=partial(get_norm, length=length),
                    se=se,
                    res=False,
                )
            )
            inc *= 2
            ouc *= 2
            length //= scale_factor

        # upsampling
        self.ups = nn.ModuleList()
        inc = d_model
        ouc = inc // 2
        lentgh = seq_len // (scale_factor ** (num_layers - 1))
        for _ in range(num_layers):
            self.ups.append(
                Up(
                    inc,
                    ouc,
                    bilinear,
                    scale_factor,
                    norm=partial(get_norm, length=lentgh),
                )
            )
            inc //= 2
            ouc //= 2
            lentgh *= scale_factor

        self.cls = nn.Sequential(
            nn.Conv1d(ouc * 2, ouc * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(ouc * 2, out_channels, kernel_size=1),
        )

    def bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            入力, shape: (batch_size, seq_len, in_chans)

        Returns
        -------
        torch.Tensor
            出力, shape: (batch_size, seq_len, out_chans)
        """
        x = x.transpose(1, 2)  # (bs, seq_len, in_chans) -> (bs, in_chans, seq_len)

        # downsampling
        xs = []
        for down in self.downs:
            x = down(x)
            xs.append(x)
        xs.pop()

        # bottleneck
        x = self.bottleneck(x)

        # upsampling
        for up, x_down in zip(self.ups, reversed(xs)):
            x = up(x, x_down)

        # head
        y = self.cls(x)
        y = y.transpose(1, 2)  # (bs, out_chans, seq_len) -> (bs, seq_len, out_chans)
        return y


class Down(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        norm=nn.BatchNorm1d,
        se=False,
        res=False,
    ):
        """Downsampling Block

        Parameters
        ----------
        in_channels : int
            入力のチャネル数
        out_channels : int
            出力のチャネル数
        scale_factor : int, optional
            ダウンサンプリングの倍率, by default 2
        norm : nn.Module, optional
            normalize, by default nn.BatchNorm1d
        se : bool, optional
            Squeeze-and-Excitationを使うかどうか, by default False
        res : bool, optional
            残差結合を使うかどうか, by default False
        """
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool1d(scale_factor),
            DoubleConv(in_channels, out_channels, norm=norm, se=se, res=res),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Up(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        scale_factor: int = 2,
        norm=nn.BatchNorm1d,
    ):
        """Upsampling Block

        Parameters
        ----------
        in_channels : int
            入力のチャネル数
        out_channels : int
            出力のチャネル数
        bilinear : bool, optional
            Bilinear Interpolationを使うかどうか, by default True
        scale_factor : int, optional
            アップサンプリングの倍率, by default 2
        norm : nn.Module, optional
            normalize, by default nn.BatchNorm1d
        """
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm=norm)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels,
                in_channels // 2,
                kernel_size=scale_factor,
                stride=scale_factor,
            )
            self.conv = DoubleConv(in_channels, out_channels, norm=norm)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff = x2.size()[2] - x1.size()[2]
        x1 = nn.functional.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int = None,
        norm=nn.BatchNorm1d,
        se=False,
        res=False,
    ):
        """Double Convolution

        Parameters
        ----------
        in_channels : int
            入力のチャネル数
        out_channels : int
            出力のチャネル数
        mid_channels : int, optional
            中間のチャネル数, by default None
        norm : nn.Module, optional
            正規化層, by default nn.BatchNorm1d
        se : bool, optional
            Squeeze-and-Excitationを使うかどうか, by default False
        res : bool, optional
            残差結合を使うかどうか, by default False
        """

        super().__init__()
        self.res = res
        if res and in_channels != out_channels:
            raise ValueError("in_channels and out_channels must be the same if `res=True`")
        if not mid_channels:
            mid_channels = out_channels
        if se:
            non_linearity = SEModule1d(out_channels)
        else:
            non_linearity = nn.ReLU(inplace=True)
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1),
            norm(mid_channels),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1),
            norm(out_channels),
            non_linearity,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.res:
            return x + self.double_conv(x)
        else:
            return self.double_conv(x)
