# いつもの状態に戻すロボット（仮称）

写真1枚で「いつもの状態」を教えておくと、散らかった部屋や会議室を、移動しながら元の状態に戻してくれるロボットのデモです。
Physical AI 応用講座の最終課題として、NVIDIA Isaac Sim 上で開発しています。

> **ステータス**：M0（技術検証）中

---

## 背景と目的

### 最終課題について

本リポジトリは、Physical AI 応用講座の最終課題の提出物です。課題の要点は次のとおりです（詳細は講座の公式資料を参照してください）。

- AI を活用してロボットにタスクを実行させ、その様子を動画にまとめる（AI は推論のみの利用でも可）
- 提出物は、タスクを実行する動画（1〜5分）、コード一式、取り組みをまとめたレポート
- 評価の観点は、アイデアの独自性、技術的な難しさ、現実世界での実用性、動画と説明のわかりやすさ
- 提出期限は 2026年11月2日

### このプロジェクトの目的

- デモ開発の経験を通じて、Physical AI 開発の勘所（シミュレーション、知覚・計画・制御の統合、評価の設計）を身につけ、実務に活かす
- 優秀作品に選ばれる水準を目指す

---

## コンセプト

### 解決したいこと

部屋の片付けや、会議室のセッティングの変更・原状回復は、日常的に発生するのに手間がかかる作業です。
また「どこに何を戻すか」は人によって違うため、ロボットに頼もうとしても、置き場所を一つずつ指定するのは現実的ではありません。

### アプローチ

**整った状態を写真に撮るだけで、ロボットに「いつもの状態」を教えられる**ようにします。

1. **教える**：整った状態を固定カメラで撮影しておく
2. **頼む**：「片付けて」と指示する（外出先からの遠隔指示も想定）
3. **見比べる**：VLM が「いつもの写真」と今の様子を比べ、動いた物と戻し先を特定する
4. **動く**：ロボットが部屋の中を移動し、物を掴んで元の位置に戻す
5. **確かめる**：もう一度写真と比べ、ずれていればやり直す
6. **報告する**：片付けの前後の写真と、何をしたかを報告する

### 想定するユースケース

- **自宅**：散らかった机や棚を、いつもの状態に戻す
- **会議室**：「会議用」「勉強会用」など、写真で指定したセッティングに並べ替える／使用後に原状回復する
- **将来の展開**：店舗の陳列の整理、工場の工具の定位置管理 など

---

## システム構成

判断が必要な部分だけを AI（VLM）が担い、実行は古典的な手法で確実に行う、階層型の構成です。

```
[エージェント層]  VLM
   天井カメラの「いつもの写真」と現在の画像を比較し、
   「何を・どこから・どこへ動かすか」を決める
        ↓
[移動層]  幾何計算＋経路探索（AI なし）
   目標の物体の近くの、アームが届く位置へ台車を移動する
        ↓
[操作層]  動作計画ライブラリ（AI なし）
   物体を掴み、目標の位置に置く
```

- **シミュレータ**：NVIDIA Isaac Sim 6.1
- **ロボット**：RidgebackFranka（全方向移動台車＋7自由度アーム）
- **AI**：VLM（推論のみ。具体的なモデルは M0 で選定）

---

## 開発ロードマップ

段階的に機能を積み上げます（予定。検証結果に応じて見直します）。

| マイルストーン | 内容 | 目安 |
|---|---|---|
| M0 技術検証 | ロボットの読み込み、台車の移動、アームの動作計画、把持、VLM による差分検出、動画記録の検証 | 〜10/11 |
| M1 停止状態でのタスク | 台車を止めたまま、VLM が差分を見つけて物を元に戻す（物体の位置はシミュレータの真値を使用） | 〜10/16 |
| M2 移動を含むタスク | 部屋の中を移動しながら複数の物を元に戻す（真値を使用） | 〜10/21 |
| M3 位置推定 | 真値を使わず、カメラ画像から物体やロボットの位置を推定する | 〜10/25 |
| M4 確認と応用 | 片付け後の確認とやり直し、会議室の場面、遠隔指示と報告 | 〜10/28 |
| M5 評価と提出物 | 条件を変えた繰り返し試行による定量評価、動画・レポートの作成 | 〜11/1 |

進捗は GitHub の Issues と Milestones で管理しています。

---

## 動作環境

| 項目 | 内容 |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 5070（12GB） |
| RAM | 32GB |
| Python | 3.12 |
| Isaac Sim | 6.1.0（pip 版） |
| PyTorch | 2.11.0（CUDA 12.8） |

実行環境の検証結果は [docs/env_check.md](docs/env_check.md) を参照してください。

---

## セットアップ

```powershell
# 仮想環境の作成
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# PyTorch と Isaac Sim
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install "isaacsim[all,extscache]==6.1.0.0" --extra-index-url https://pypi.nvidia.com

# 開発用の依存パッケージ（pyproject.toml の dev グループ）
pip install --group dev

# 本プロジェクトのパッケージ（src/restorebot）を editable install
pip install -e .
```

Isaac Sim の利用には、NVIDIA Omniverse License Agreement（EULA）への同意が必要です。

```powershell
$env:OMNI_KIT_ACCEPT_EULA = "YES"
```

## 実行方法

```powershell
# 動作確認（headless でシーンを構築し、結果を outputs/ に保存）
.venv\Scripts\python.exe scripts\smoke_test.py

# 高速チェック（ruff format --check、ruff check、単体テスト）
.\scripts\check.ps1

# 全チェック（上記＋Isaac Sim を使う統合テスト）
.\scripts\check.ps1 -Sim
```

---

## リポジトリ構成

```
.
├── src/            # ソースコード（core: Isaac Sim 非依存のロジック / sim: Isaac Sim 依存の部分）
├── tests/          # テスト（tests/sim は Isaac Sim が必要）
├── scripts/        # 実行スクリプト
├── docs/           # 設計書・検証記録
│   ├── env_check.md
│   └── experiments/
└── outputs/        # 実行結果（Git 管理外）
```

---

## 関連研究

- D. Batra et al., "Rearrangement: A Challenge for Embodied AI," 2020.
- L. Weihs et al., "Visual Room Rearrangement," CVPR 2021.
- J. Wu et al., "TidyBot: Personalized Robot Assistance with Large Language Models," Autonomous Robots, 2023.
- W. Huang et al., "Inner Monologue: Embodied Reasoning through Planning with Language Models," CoRL 2022.

## 利用しているソフトウェア・アセット

- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac/sim)
- RidgebackFranka（Clearpath Robotics, BSD-3 License）：Isaac Sim のアセットサーバーから読み込んで使用しています

NVIDIA のアセットは本リポジトリに含めていません。
