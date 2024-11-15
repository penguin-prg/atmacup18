import timm
import torch
from torch import nn


class TimmBaseModel(nn.Module):
    """TimmBaseModel
    画像を入力として、任意の次元のベクトルを出力するモデル
    """

    def __init__(
        self,
        model_name="resnet18d",
        pretrained: bool = True,
        in_chans: int = 3,
        hidden_dim: int = 512,
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
        hidden_dim : int, optional
            隠れ層の次元数
        output_dim : int, optional
            出力次元数
        dropout : float, optional
            ドロップアウト率
        last_layer : nn.Module, optional
            最後の層。
            必要に応じて、`nn.Softmax()`などを指定する。
        """
        super().__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
        )
        self.in_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.Linear(self.in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.last_layer = last_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            入力データ. `(bs, in_chans, H, W)`

        Returns
        -------
        torch.Tensor
            出力データ. `(bs, output_dim)`
        """
        x = self.backbone(x)
        x = self.head(x)
        x = self.last_layer(x)
        return x
