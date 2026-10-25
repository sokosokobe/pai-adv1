# いつもの状態に戻すロボット（仮称）

写真1枚で「いつもの状態」を教えておくと、散らかった部屋や会議室を、移動しながら元の状態に戻してくれるロボットのデモです。
Physical AI 応用講座の最終課題として、NVIDIA Isaac Sim 上で開発しています。

> **ステータス**：M1（停止状態でのタスク）の実装中。M0（技術検証）の結果は [docs/experiments/M0_summary.md](docs/experiments/M0_summary.md)、設計書は [docs/design.md](docs/design.md)

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
- **ロボット**：RidgebackFranka（全方向移動台車＋7自由度アーム。Isaac Sim の Multiphysics 版のアセット）
- **AI**：VLM（推論のみ）。Claude Sonnet 5.5 を使う（代わりの候補：Gemini Robotics-ER 2）

M0（技術検証）で決めた主な構成は次のとおりです（詳細は [M0 のまとめ](docs/experiments/M0_summary.md)）。

| 層・機能 | 方法 |
|---|---|
| エージェント層 | 天井カメラ（高さ 2.6m、真下向き、RGB＋深度）の「いつもの写真」と現在の画像を VLM に渡し、動いた物体・今の位置・戻し先を JSON で受け取る |
| 移動層 | 台車の速度指令（上限 0.5m/s・1.0rad/s）。物体が台車の正面 0.55〜0.85m に来る位置に止める |
| 操作層 | cuMotion の RmpFlowController と、Isaac Sim 6.1 の Pick and Place サンプルの把持・配置の手順（PickPlaceController） |
| 動画 | headless で撮影用のカメラの画像を連番で保存し、ffmpeg で mp4 にする |

---

## 開発ロードマップ

段階的に機能を積み上げます（予定。検証結果に応じて見直します）。画像からの位置推定、会議室の場面、遠隔指示の画面、安全性の機能は、余裕があれば取り組みます。

| マイルストーン | 内容 | 目安 |
|---|---|---|
| M0 技術検証 | ロボットの読み込み、台車の移動、アームの動作計画、把持、VLM による差分検出、動画記録、起動時間の検証 | 〜10/11（10/12 完了） |
| 設計 | 片付けの対象物と部屋の決定、ロボットの写り込みへの対策、設計書（[docs/design.md](docs/design.md)）と計画 | 〜10/25（当初 10/14） |
| M1 停止状態でのタスク | 台車を止めたまま、VLM が差分を見つけて物を元に戻す（物体の位置はシミュレータの真値を使用）。合格ラインの動画 | 〜10/27 |
| M2 移動を含むタスク | 部屋の中を移動しながら複数の物を元に戻す（真値を使用）。家具・物体から距離をとる経路計画 | 〜10/30 |
| M3 評価 | 決まったシードでの繰り返し試行による定量評価。片付け後の確認とやり直しは余裕があれば | 〜10/31 |
| M4 動画とレポート | 提出動画の編集、レポートの作成（締切 11/2 10:00） | 〜11/1 |

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

VLM の API キーは、リポジトリ直下の `.env` に書きます（書き方は `.env.example` を参照。`.env` はコミットしません）。

```text
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...   # 代わりの候補（Gemini Robotics-ER 2）を使う場合
```

動画の作成には ffmpeg を使います（開発環境では `winget install --id Gyan.FFmpeg` で導入した FFmpeg 9.0.2 を使用）。

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
│   ├── design.md       # 設計書（作成中）
│   ├── env_check.md
│   └── experiments/    # 検証記録（M0_summary.md が M0 のまとめ）
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
- [Claude API](https://www.anthropic.com/api)（Anthropic）：VLM として使用しています
- [FFmpeg](https://ffmpeg.org/)：動画の作成に使用しています

NVIDIA のアセットは本リポジトリに含めていません。
