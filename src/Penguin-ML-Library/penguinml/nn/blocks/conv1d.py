import torch
from torch import nn


class Conv1dBlock(nn.Module):
    """Conv1dBlock
    conv1d -> layer norm -> SiLU
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int,
    ):
        """
        Parameters
        ----------
        in_ch : int
            入力チャネル数
        out_ch : int
            出力チャネル数
        k : int
            カーネルサイズ
        """
        assert k % 2 == 1, "kernel size must be odd"

        super(Conv1dBlock, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2)
        self.ln = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            入力. (batch_size, in_ch, seq_len)

        Returns
        -------
        torch.Tensor
            出力. (batch_size, out_ch, seq_len)
        """
        x = self.conv(x)  # -> (bs, out_ch, seq_len)
        x = x.permute(0, 2, 1)  # -> (bs, seq_len, out_ch)
        x = self.ln(x)
        x = x.permute(0, 2, 1)  # -> (bs, out_ch, seq_len)
        x = nn.SiLU()(x)
        return x
