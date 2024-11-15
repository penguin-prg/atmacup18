from typing import Dict

import timm
import torch
import torch.nn as nn

from .arcface import ArcMarginProduct, GeM


class TimmArcFaceModel(nn.Module):
    def __init__(
        self,
        n_classes: int,
        encoder_name: str = "resnet18",
        pretrained: bool = True,
        in_channels: int = 3,
        embedding_size: int = 512,
        s: float = 30.0,
        m: float = 0.5,
        easy_margin: bool = False,
        ls_eps: float = 0.0,
    ):
        """Timm -> ArcFace

        Parameters
        ----------
        n_classes : int
            クラス数
        encoder_name : str, optional
            エンコーダ名, by default "resnet18"
        pretrained : bool, optional
            事前学習済みモデルを使用するか, by default True
        in_channels : int, optional
            入力チャンネル数, by default 3
        embedding_size : int, optional
            埋め込みベクトルの次元数, by default 512
        s : float, optional
            スケール, by default 30.0
        m : float, optional
            マージン, by default 0.5
        easy_margin : bool, optional
            イージーマージン, by default False
        ls_eps : float, optional
            ラベルスムージング, by default 0.0
        """
        super(TimmArcFaceModel, self).__init__()

        # Encoder
        self.model = timm.create_model(encoder_name, pretrained=pretrained, in_chans=in_channels)

        # Embedding
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.embedding = nn.Linear(in_features, embedding_size)

        # head
        self.fc = ArcMarginProduct(
            embedding_size,
            n_classes,
            s=s,
            m=m,
            easy_margin=easy_margin,
            ls_eps=ls_eps,
        )

    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """特徴抽出"""
        features = self.model(x)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        return embedding

    def forward(self, x: Dict) -> torch.Tensor:
        """順伝播"""
        image = x["image"]
        label = x["label"]
        features = self.extract(image)
        x = self.fc(features, label)
        return x
