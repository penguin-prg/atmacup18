import timm
import torch
from torch import nn


class TimmLSTMModel(nn.Module):
    """TimmBaseModel
    画像の系列->timmで特徴抽出→LSTM
    """

    def __init__(
        self,
        model_name="resnet18d",
        pretrained: bool = True,
        in_chans: int = 3,
        lstm_input_dim: int = 256,
        lstm_hidden_dim: int = 256,
        lstm_num_layers: int = 2,
        head_hidden_dim: int = 256,
        output_dim: int = 1,
        dropout: float = 0.0,
        last_layer: nn.Module = nn.Identity(),
    ):
        """
        Parameters
        ----------
        model_name : str, optional
            モデル名
        pretrained : bool, optional
            モデルを事前学習済みにするかどうか
        in_chans : int, optional
            入力チャンネル数
        lstm_input_dim : int, optional
            LSTMの入力次元数
        lstm_hidden_dim : int, optional
            LSTMの隠れ層の次元数
        lstm_num_layers : int, optional
            LSTMの層数
        head_hidden_dim : int, optional
            headの次元数
        output_dim : int, optional
            出力次元数
        dropout : float, optional
            ドロップアウト率
        last_layer : nn.Module, optional
            最後の層。
            必要に応じて、`nn.Softmax()`などを指定する。
        """
        super().__init__()

        self.lstm_input_dim = lstm_input_dim
        self.output_dim = output_dim

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=lstm_input_dim,
        )

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_input_dim * 2, head_hidden_dim),
            nn.BatchNorm1d(head_hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(head_hidden_dim, output_dim),
        )
        self.last_layer = last_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            入力データ. `(bs, seq_len, in_chans, H, W)`

        Returns
        -------
        torch.Tensor
            出力データ. `(bs, seq_len, output_dim)`
        """
        bs, seq_len, in_chans, H, W = x.shape
        x = x.view(bs * seq_len, in_chans, H, W)
        x = self.backbone(x)
        x = x.view(bs, seq_len, -1)
        x, _ = self.lstm(x)
        x = x.contiguous().view(bs * seq_len, -1)
        x = self.head(x)
        x = self.last_layer(x)
        x = x.view(bs, seq_len, self.output_dim)
        return x
