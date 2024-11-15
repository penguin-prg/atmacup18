# Kaggle-Template

Kaggleで使用するテンプレートリポジトリ

## Setup

1. setup.shを実行
   ```
   chmod 755 setup.sh
   ./setup.sh
   ```
   - NOTE: git cloneでプライベートリポジトリを持ってくるので、権限がない人が使う場合はコメントアウトする

## ディレクトリ構成

- /kaggle
  - .devcontainer/ : 環境構築
  - .vscode/ : vscodeの設定ファイル
  - input/ : データセット
  - output/ : 実験結果の出力ディレクトリ
  - working/ : 作業ディレクトリ（一時ファイルのみ）
  - src/ : ソースコード
    - config.yaml : 設定ファイル
    - Penguin-ML-Library/ : MLプロジェクト用ライブラリ
  - eda/ : EDAのnotebook
  - train/ : 学習コード
