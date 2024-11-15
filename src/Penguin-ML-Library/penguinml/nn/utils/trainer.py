from typing import Dict, List, Literal

import numpy as np
import torch
from torch import nn, optim
from torchmetrics import MetricCollection
from tqdm import tqdm

from ...utils.logger import get_logger
from ...utils.timer import Timer


class Trainer:
    """Trainer for PyTorch"""

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        scaler: torch.cuda.amp.GradScaler,
        max_epoch: int,
        amp: bool = True,
        metrics: List = [],
        monitor: str = "valid_loss",
        direction: Literal["minimize", "maximize"] = "minimize",
        device: Literal["cuda", "cpu"] = "cuda",
        save_path: str = "best_model.pth",
        es_patience: int = 2,
        loss_window: int = 100000,
    ):
        """
        Parameters
        ----------
        model : nn.Module
            モデル
        train_dataloader : torch.utils.data.DataLoader
            学習用データローダー
        valid_dataloader : torch.utils.data.DataLoader
            検証用データローダー
        loss_fn : nn.Module
            損失関数
        optimizer : optim.Optimizer
            最適化関数
        scheduler : optim.lr_scheduler._LRScheduler
            スケジューラー
        scaler : torch.cuda.amp.GradScaler
            勾配スケーラー
        max_epoch : int
            最大エポック数
        amp : bool, optional
            自動混合精度, by default True
        metrics : List, optional
            評価指標, by default []
            `metricscollection`と同じIF
        monitor : str, optional
            早期終了・チェックポイントの指標, by default "valid_loss"
        direction : Literal["minimize", "maximize"], optional
            早期終了・チェックポイントの方向, by default "minimize"
        device : Literal["cuda", "cpu"], optional
            デバイス, by default "cuda"
        save_path : str, optional
            チェックポイントの保存先, by default "best_model.pth"
        es_patience : int, optional
            早期終了のパラメータ, by default 2
        loss_window: int, optional
            lossの移動平均, by default 100000
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.max_epoch = max_epoch
        self.amp = amp
        self.train_metrics = MetricCollection([metrics], prefix="train_")
        self.valid_metrics = MetricCollection([metrics], prefix="valid_")
        self.monitor = monitor
        self.direction = direction
        self.device = device
        self.save_path = save_path
        self.es_patience = es_patience
        self.loss_window = loss_window
        self.logger = get_logger(__name__)

    def train(self):
        """学習を実行"""
        self.logger.info("== start training ==")
        self.logger.info(f"train dataloader length: {len(self.train_dataloader)}")
        self.logger.info(f"valid dataloader length: {len(self.valid_dataloader)}")
        self.logger.info("-" * 30)
        timer = Timer()

        best_val_score = np.inf if self.direction == "minimize" else -np.inf
        best_val_epoch = -1
        for epoch in range(self.max_epoch):
            metrics = self.run_epoch()
            s = " ".join([f"{k}: {v:.4}" for k, v in sorted(metrics.items())])
            self.logger.info(f"[epoch {epoch}] {s}")
            val_score = metrics[self.monitor]
            if (self.direction == "minimize") ^ (val_score > best_val_score):
                torch.save(self.model.state_dict(), self.save_path)
                self.logger.info(f"saved best model ({best_val_score:.4} -> {val_score:.4})")
                best_val_score = val_score
                best_val_epoch = epoch
            elif epoch - best_val_epoch >= self.es_patience:
                self.logger.info("early stopped!")
                break

        elapsed_time = timer.time_str()
        self.logger.info(f"finished training ({elapsed_time})")

    @torch.no_grad()
    def inference(self) -> np.ndarray:
        """推論を実行

        Returns
        -------
        preds : np.ndarray
            推論結果
        """
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        preds = []
        for X, _ in tqdm(self.valid_dataloader, leave=False):
            if isinstance(X, dict):
                X = {k: v.to(self.device) for k, v in X.items()}
            else:
                X = X.to(self.device)
            pred = self.model(X)
            preds.append(pred.cpu().numpy())
        preds = np.concatenate(preds)
        return preds

    def run_epoch(self) -> Dict[str, float]:
        """1エポック分の学習&検証を実行

        Returns
        -------
        metrics : Dict[str, float]
            メトリクス
        """
        train_metrics = self.single_loop(mode="train")
        with torch.no_grad():
            valid_metrics = self.single_loop(mode="valid")
        metrics = {**train_metrics, **valid_metrics}
        return metrics

    def single_loop(self, mode: Literal["train", "valid"]) -> Dict[str, float]:
        """1ループ分の学習or検証を実行

        Parameters
        ----------
        mode : Literal["train", "valid"]
            モード

        Returns
        -------
        metrics : Dict[str, float]
            メトリクス.
        """
        assert mode in ["train", "valid"]
        if mode == "train":
            self.model.train()
            datalorder = self.train_dataloader
        else:
            self.model.eval()
            datalorder = self.valid_dataloader

        total_loss = 0
        preds = []
        targets = []
        losses = []
        bar = tqdm(datalorder, leave=False)
        for X, y in bar:
            try:
                if isinstance(X, dict):
                    X = {k: v.to(self.device) for k, v in X.items()}
                else:
                    X = X.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=self.amp):
                    pred = self.model(X)
                    loss = self.loss_fn(pred, y)
                    preds.append(pred.detach().cpu())
                    targets.append(y.detach().cpu())
                losses.append(loss.item())
                bar.set_postfix(
                    {
                        "avg_loss": np.mean(losses[-self.loss_window :]),
                        "loss": loss.item(),
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                )
                total_loss += loss.item()

                if mode == "train":
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
            except KeyboardInterrupt:
                self.logger.info("keyboard interrupted")
                break

        total_loss /= len(datalorder)

        # metrics
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)
        if mode == "train":
            metrics = self.train_metrics(preds, targets)
        else:
            metrics = self.valid_metrics(preds, targets)
        metrics[f"{mode}_loss"] = total_loss

        return metrics
