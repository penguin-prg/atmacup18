# #18 Turing × atmaCup, 23th Place Solution

コンペ：https://www.guruguru.science/competitions/25/
解法：https://www.guruguru.science/competitions/25/discussions/a48f1c6a-04ce-4502-b7c1-3601f16bd87c/

# 結果の再現

## Hardware

- CPU: Intel Core i9 13900KF (24 cores, 32 threads)
- GPU: NVIDIA GeForce RTX 4090
- RAM: 64GB

## OS/platform

- WSL2 (version 2.0.9.0, Ubuntu 22.04.2 LTS)

## 3rd-party software

`/kaggle/.devcontainer`を確認してください。

## Training

1. [データセット]
   - コンペデータを`/kaggle/input/atmaCup#18_dataset`に配置
2. [前処理]
   - `/kaggle/input/cv-split/split.ipynb`
   - `/kaggle/input/depth_image/gen.ipynb`
   - `/kaggle/input/depth_image/agg.ipynb`
   - `/kaggle/input/table_image/gen.ipynb`
   - `/kaggle/input/traffic_light/gen_df.ipynb`
   - `/kaggle/input/yolo-det/trial_yolov11.ipynb`
3. [学習]
   - `/kaggle/train/cnn.ipynb`
   - `/kaggle/train/lightgbm.ipynb`
   - `/kaggle/train/xgboost.ipynb`
   - `68ca3e36b490cc9ba88c0928020fb379e33f7e0d`に checkout した後、`/kaggle/train/cnn.ipynb`を再度実行
   - `main`ブランチに戻って、`/kaggle/train/ensemble.ipynb`を実行
   - -> `/kaggle/output/ensemble_001/submission.csv`が最終提出です
