import gc
import os
import pickle
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import optuna
import polars as pl
from xgboost import XGBClassifier, XGBRegressor

from ..utils.contena import FeatureContena

xgb_base_params = {
    "max_depth": 7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "subsample": 1.0,
    "tree_method": "gpu_hist",
    "random_state": 42,
    "learning_rate": 0.01,
    "n_estimators": 1000000,
}


def fit_xgb(
    data: pl.DataFrame,
    features: FeatureContena,
    params: Dict[str, Any],
    target_col: str = "target",
    fold_col: str = "fold",
    target_type: Literal["regression", "binary", "multiclass"] = "regression",
    es_rounds: int = 200,
    verbose: int = 100,
    val_folds: Optional[List[int]] = None,
    only_train: bool = False,
    save_dir: Optional[str] = None,
) -> Tuple[np.ndarray, List[Union[XGBClassifier, XGBRegressor]]]:
    """xgboostを学習

    Parameters
    ----------
    data : pl.DataFrame
        学習データ
    features : FeatureContena
        特徴量
    params : Dict[str, Any]
        xgboostのパラメータ
    target_col : str, optional
        目的変数のカラム名, by default "target"
    fold_col : str, optional
        foldのカラム名, by default "fold"
    target_type : Literal["regression", "binary", "multiclass"], optional
        目的変数の種類, by default "regression"
    es_rounds : int, optional
        early stoppingの回数, by default 200
    verbose : int, optional
        学習のログを出力する間隔, by default 100
    val_folds : Optional[List[int]], optional
        学習するfoldのリスト, by default None (全てのfoldを学習)
    only_train : bool, optional
        学習のみ行うかどうか, by default False

    Returns
    -------
    oof : np.ndarray
        out of foldの予測値
    models : List[Union[XGBClassifier, XGBRegressor]]
        学習したモデル
    """
    models = []
    data = data.with_columns(pl.Series(range(len(data))).alias("__index"))
    oofs = []

    # 学習するfoldを指定
    if val_folds is None:
        val_folds = sorted(data[fold_col].unique())

    for val_fold in val_folds:
        print(f"== fold {val_fold} ==")

        # モデルを作成
        if target_type == "regression":
            model = XGBRegressor(**params)
        elif target_type == "binary" or target_type == "multiclass":
            model = XGBClassifier(**params)
        else:
            raise ValueError(f"target_type={target_type} is not supported")

        # 評価データ
        if only_train:
            eval_set = []
        else:
            X_val = data.filter(pl.col(fold_col) == val_fold).select(features.all_features())
            y_val = data.filter(pl.col(fold_col) == val_fold)[target_col]
            eval_set = [(X_val, y_val)]

        # 学習
        model = model.fit(
            X=data.filter(pl.col(fold_col) != val_fold).select(features.all_features()),
            y=data.filter(pl.col(fold_col) != val_fold)[target_col],
            eval_set=eval_set,
            early_stopping_rounds=es_rounds,
            verbose=verbose,
        )
        models.append(model)

        # 推論して保存
        if not only_train:
            if target_type == "regression":
                valid_pred = model.predict(eval_set[0][0])
            elif target_type == "binary" or target_type == "multiclass":
                valid_pred = model.predict_proba(eval_set[0][0])
            else:
                raise ValueError(f"target_type={target_type} is not supported")

            indexes = data.filter(pl.col(fold_col) == val_fold)["__index"].to_numpy()
            oofs += list(zip(indexes, valid_pred))
            del eval_set, X_val, y_val, valid_pred
        gc.collect()

    # 一時的に追加した__indexカラムを削除してoofを取り出す
    data = data.drop("__index")
    oofs = sorted(oofs, key=lambda x: x[0])
    oof = np.array([x[1] for x in oofs])

    if save_dir is not None:
        for fold, model in zip(val_folds, models):
            with open(os.path.join(save_dir, f"xgb_fold{fold}.pkl"), "wb") as f:
                pickle.dump(model, f)
    return oof, models


def inference_xgb(
    models: List[Union[XGBClassifier, XGBRegressor]],
    feat_df: pl.DataFrame,
    agg_func: Any = np.mean,
) -> np.ndarray:
    """xgboostを推論

    Parameters
    ----------
    models : List[Union[XGBClassifier, XGBRegressor]]
        学習したモデル
    feat_df : pl.DataFrame
        特徴量

    Returns
    -------
    pred : np.ndarray
        予測値
    """
    if isinstance(models[0], XGBRegressor):
        pred = np.array([model.predict(feat_df) for model in models])
    elif isinstance(models[0], XGBClassifier):
        pred = np.array([model.predict_proba(feat_df) for model in models])
    pred = agg_func(pred, axis=0)
    return pred


def tuning_xgb(
    data: pl.DataFrame,
    features: FeatureContena,
    objective: str,
    eval_metric: Callable[[np.ndarray, np.ndarray], float],
    n_trials: int = 100,
    target_col: str = "target",
    fold_col: str = "fold",
    direction: Literal["minimize", "maximize"] = "minimize",
    target_type: Literal["regression", "binary", "multiclass"] = "regression",
    lr: float = 0.1,
    es_rounds: int = 200,
    val_folds: Optional[List[int]] = None,
    seed: int = 42,
) -> Tuple[pl.DataFrame, Dict[str, float], float]:
    """xgboostのパラメータをチューニング

    Parameters
    ----------
    data : pl.DataFrame
        学習データ
    features : FeatureContena
        特徴量
    objective : str
        xgboostのobjective
    eval_metric : Callable[[np.ndarray, np.ndarray], float]
        評価関数. `f(y_true, y_pred) -> score`
    n_trials : int, optional
        optunaの試行回数, by default 100
    target_col : str, optional
        目的変数のカラム名, by default "target"
    fold_col : str, optional
        foldのカラム名, by default "fold"
    direction : Literal["minimize", "maximize"], optional
        optunaの方向, by default "minimize"
    target_type : Literal["regression", "binary", "multiclass"], optional
        目的変数の種類, by default "regression"
    lr : float, optional
        学習率, by default 0.1
    es_rounds : int, optional
        early stoppingの回数, by default 200
    val_folds : Optional[List[int]], optional
        学習するfoldのリスト, by default None (全てのfoldを学習)
    seed : int, optional
        seed, by default 42

    Returns
    -------
    df : pl.DataFrame
        optunaの結果
    best_params : Dict[str, float]
        最適なパラメータ
    best_score : float
        最適なスコア
    """

    def _objective(trial: optuna.Trial):
        max_depth = trial.suggest_int("max_depth", 3, 10)
        colsample_bytree = trial.suggest_uniform("colsample_bytree", 0.2, 0.8)
        reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-3, 1e3)
        reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-3, 1e3)
        subsample = trial.suggest_loguniform("subsample", 0.5, 1.0)

        params = {
            "objective": objective,
            "learning_rate": lr,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "max_depth": max_depth,
            "colsample_bytree": colsample_bytree,
            "subsample": subsample,
            "tree_method": "gpu_hist",
            "random_state": seed,
            "n_estimators": 5000000,
        }

        oof, _ = fit_xgb(
            data,
            features,
            params,
            target_col=target_col,
            fold_col=fold_col,
            target_type=target_type,
            es_rounds=es_rounds,
            verbose=10000000,
            val_folds=val_folds,
        )
        if val_folds is None:
            target = data[target_col].to_numpy()
        else:
            target = data.filter(pl.col(fold_col).is_in(val_folds))[target_col].to_numpy()
        score = eval_metric(target, oof)
        return score

    study = optuna.create_study(direction=direction)
    study.enqueue_trial(
        {
            "max_depth": 7,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "subsample": 1.0,
        }
    )
    study.optimize(_objective, n_trials=n_trials)

    df = study.trials_dataframe()
    best_params = study.best_params
    best_score = study.best_value
    return df, best_params, best_score


def plot_importance():
    raise NotImplementedError
