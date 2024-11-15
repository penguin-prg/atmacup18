import os
import shutil
from typing import Dict

import wandb


def wandb_init(CFG: Dict, project: str, exp_id: str):
    """wandbの初期化

    同名の実験がある場合は削除してから初期化する
    """

    # 古いのを捨てる
    api = wandb.Api()
    runs = api.runs(project)
    for run in runs:
        if run.config.get("exp_name") == exp_id:
            run.delete()

    wandb_dir = "wandb"
    if os.path.exists(wandb_dir):
        for item in os.listdir(wandb_dir):
            if exp_id in item:
                print(f"Deleting local run {item}")
                shutil.rmtree(os.path.join(wandb_dir, item))

    # 新しいのを作る
    wandb.init(project=project, name=exp_id, config=CFG)
