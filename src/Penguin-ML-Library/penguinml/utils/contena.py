import pickle
from typing import List, Optional, Set


class FeatureContena:
    """特徴量を管理するクラス"""

    def __init__(self, path: Optional[str] = None):
        """初期化

        Parameters
        ----------
        path : Optional[str]
            保存された特徴量を読み込む場合のファイルパス
        """
        self._num_features: Set[str] = set()
        self._cat_features: Set[str] = set()

        if path is not None:
            with open(path, "rb") as f:
                contena = pickle.load(f)
                assert isinstance(contena, FeatureContena)
                self._num_features = contena._num_features
                self._cat_features = contena._cat_features

    def add_num_feature(self, feature: str):
        """数値特徴量を1つ追加"""
        assert isinstance(feature, str)
        self._num_features.add(feature)

    def add_num_features(self, features: List[str]):
        """数値特徴量を複数を追加"""
        assert isinstance(features, list)
        self._num_features |= set(features)

    def add_cat_feature(self, feature: str):
        """カテゴリカルな特徴量を1つ追加"""
        assert isinstance(feature, str)
        self._cat_features.add(feature)

    def add_cat_features(self, features: List[str]):
        """カテゴリカルな特徴量を1つ複数"""
        assert isinstance(features, list)
        self._cat_features |= set(features)

    def num_features(self) -> List[str]:
        """数値特徴量をソートして取得"""
        return sorted(list(self._num_features))

    def cat_features(self) -> List[str]:
        """カテゴリカル特徴量をソートして取得"""
        return sorted(list(self._cat_features))

    def all_features(self) -> List[str]:
        """全ての特徴量を取得"""
        return self.num_features() + self.cat_features()

    def clear(self):
        """特徴量を削除"""
        self._num_features = set()
        self._cat_features = set()

    def remove_num_features(self, features: List[str]):
        """指定した数値特徴量を削除"""
        assert isinstance(features, list)
        self._num_features -= set(features)

    def remove_cat_features(self, features: List[str]):
        """指定したカテゴリカル特徴量を削除"""
        assert isinstance(features, list)
        self._cat_features -= set(features)

    def add_suffix(self, suffix: str):
        """特徴量名にサフィックスを追加"""
        assert isinstance(suffix, str)
        self._num_features = set([f"{feature}{suffix}" for feature in self._num_features])
        self._cat_features = set([f"{feature}{suffix}" for feature in self._cat_features])

    def merge(self, other: "FeatureContena") -> None:
        """他のFeatureContenaをマージ

        Parameters
        ----------
        other : FeatureContena
            マージするFeatureContena
        """
        assert isinstance(other, FeatureContena)
        self.add_num_features(other.num_features())
        self.add_cat_features(other.cat_features())

    def save(self, path: str) -> None:
        """特徴量をファイルに保存

        Parameters
        ----------
        path : str
            保存先のファイルパス
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def __str__(self) -> str:
        return f"[num_features]\n{self.num_features()}\n[cat_features]\n{self.cat_features()}"

    def __len__(self) -> int:
        return len(self.all_features())
