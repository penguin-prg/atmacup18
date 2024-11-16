import os
import sys
from typing import Tuple

import numpy as np
import polars as pl

if True:
    sys.path.append("/kaggle/src/Penguin-ML-Library")
    sys.path.append("/kaggle/src")

from penguinml.utils.contena import FeatureContena

from const import CATEGORY_MAPPING, TARGET_COLS

basic_features = [
    "vEgo",
    "aEgo",
    "steeringAngleDeg",
    "steeringTorque",
    "brake",
    "brakePressed",
    "gas",
    "gasPressed",
    "leftBlinker",
    "rightBlinker",
    "offset",
]


def feature_engineer(train: pl.DataFrame) -> Tuple[pl.DataFrame, FeatureContena]:
    features = FeatureContena()

    # そのまま使う特徴量
    train, features = add_basic_features(train, features)

    # ラグ特徴量
    train, features = add_scene_lag_features(train, features)

    # targetをmelt
    train, features = melt_target_columns(train, features)
    train, features = add_dt_features(train, features)

    # シーン内の特徴量
    train, features = add_scene_agg_features(train, features)

    # キャスト
    train = cast_to_float32(train, features)
    return train, features


def cast_to_float32(train: pl.DataFrame, features: FeatureContena) -> pl.DataFrame:
    """特徴量をfloat32にキャストする"""
    for c in features.num_features():
        train = train.with_columns(pl.col(c).cast(pl.Float32))

    for c in features.cat_features():
        mapping = CATEGORY_MAPPING[c]
        train = train.with_columns(pl.col(c).replace_strict(mapping).cast(pl.Int32))
    return train


def add_basic_features(
    train: pl.DataFrame,
    features: FeatureContena,
) -> Tuple[pl.DataFrame, FeatureContena]:
    """基本的な特徴量を追加する"""
    for c in basic_features:
        train = train.with_columns(pl.col(c).cast(pl.Float32))
    features.add_num_features(basic_features)
    features.add_cat_features(["gearShifter"])

    return train, features


def melt_target_columns(
    train: pl.DataFrame,
    features: FeatureContena,
) -> Tuple[pl.DataFrame, FeatureContena]:
    """targetをmeltして特徴量に追加する"""
    train = (
        train.unpivot(index="ID", on=TARGET_COLS, variable_name="target_name", value_name="target")
        .join(
            train.drop(TARGET_COLS),
            on="ID",
            how="left",
        )
        .with_columns(
            (
                pl.col("target_name").map_elements(lambda x: float(x.split("_")[1]), return_dtype=pl.Float32) * 0.5
                + 0.5
            ).alias("dt"),
            pl.col("target_name").map_elements(lambda x: x.split("_")[0], return_dtype=str).alias("xyz"),
        )
        .with_columns(
            pl.col("target_name").alias("target_name_original"),
        )
    )
    features.add_cat_features(["target_name", "xyz"])
    return train, features


def add_scene_lag_features(
    train: pl.DataFrame,
    features: FeatureContena,
) -> Tuple[pl.DataFrame, FeatureContena]:
    """シーン内のラグ特徴量"""
    train = train.sort(["sceneID", "offset"])
    for c in basic_features:
        for diff in [-1, 1]:
            train = train.with_columns(
                pl.col(c).diff(n=diff).over("sceneID").alias(f"{c}_diff_{diff}"),
                pl.col(c).diff(n=diff).over("sceneID").alias(f"{c}_shift_{diff}"),
            )
            features.add_num_features([f"{c}_diff_{diff}", f"{c}_shift_{diff}"])
    return train, features


def add_scene_agg_features(
    train: pl.DataFrame,
    features: FeatureContena,
) -> Tuple[pl.DataFrame, FeatureContena]:
    """シーン内の集約特徴量"""
    for c in basic_features:
        train = train.with_columns(
            pl.col(c).mean().over("sceneID").alias(f"{c}_mean"),
            pl.col(c).std().over("sceneID").alias(f"{c}_std"),
            pl.col(c).max().over("sceneID").alias(f"{c}_max"),
            pl.col(c).min().over("sceneID").alias(f"{c}_min"),
        )
        features.add_num_features([f"{c}_mean", f"{c}_std", f"{c}_max", f"{c}_min"])
    return train, features


def add_dt_features(
    train: pl.DataFrame,
    features: FeatureContena,
) -> Tuple[pl.DataFrame, FeatureContena]:
    """dt秒後の特徴"""
    train = train.with_columns(
        # vt
        (pl.col("vEgo") * pl.col("dt").cast(pl.Float32)).alias("linear_movement@dt"),
        # vt + 0.5at^2
        ((pl.col("vEgo") + 0.5 * pl.col("aEgo") * pl.col("dt").cast(pl.Float32) ** 2).alias("movement@dt")),
        # v + at
        (pl.col("vEgo") + pl.col("aEgo") * pl.col("dt").cast(pl.Float32)).alias("velocity@dt"),
    )
    features.add_num_features(["linear_movement@dt", "movement@dt", "velocity@dt"])
    return train, features
