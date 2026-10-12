# #8 Isaac Sim の起動時間の短縮

- **Issue**：#8
- **日付**：2026-10-12
- **タイムボックス**：1時間 ／ **実績**：約0.2時間（17:19〜17:30）
- **検証用コード**：`scripts/experiments/startup_time.py`（対策ごとの起動時間の計測、torch を読み込んだ呼び出し元の記録）、`startup_common.py`（非推奨の拡張機能を除いた experience の生成。単体テスト `tests/experiments/test_startup_common.py`）
- **証拠**：`outputs/m0_8/startup.jsonl`（計測結果）、`outputs/m0_8_*.log`（起動のログ）、#5・#6 の起動のログ（`outputs/m0_6_sample.log` など）、Windows のイベントログ（Microsoft-Windows-CodeIntegrity/Operational）

## 目的

1. Isaac Sim の起動に毎回約 200 秒かかる原因（内訳）は何か
2. 起動時間を短くする方法はあるか（スマート アプリ コントロールの設定は変えない）

## 方法

- 起動時間の内訳：#6 の起動のログ（Kit のログの時刻）と、Windows の Code Integrity のイベントログ（イベント ID 3077：コードの整合性のポリシーによる阻止）の時刻を突き合わせた
- 対策ごとに、`SimulationApp` の生成までの時間と、これまでの検証で使った API が動くかを測った（`startup_time.py`）
  - 確認した API：シーンの作成（`GroundPlane`、`Cube`）、物理の開始、`CameraSensor` の画像の取得（中央の赤い立方体が赤く写るか）、cuMotion（`RmpFlowController`）と `PickPlaceController` の import

  | 対策 | 内容 |
  |---|---|
  | baseline | 何もしない（標準の experience `isaacsim.exp.base.python.kit`） |
  | A：no_deprecated | 非推奨の拡張機能（`extsDeprecated/` 配下の 25 個のうち、標準の experience が読み込む 12 個とその依存先）を除いた experience で起動する。インストール先の標準のファイルから `build/kit_apps/` に生成する |
  | B'：stub_torch | `SimulationApp` の前に、中身のない `torch` のモジュールを `sys.modules` に登録する（torch の読み込みをすぐに終わらせる） |
  | A＋B' | 両方 |

  - 計画の案 B（torch を外す）は、調べた結果やめた。非推奨の拡張機能は torch が無いと（ImportError）アプリを終了させる（`isaacsim.core.deprecation_manager` の `import_module`）
  - 案 C（スマート アプリ コントロールの設定を変える）は、方針により対象外

## 結果

### 1. 原因：スマート アプリ コントロールが torch の DLL を断続的に阻止し、1 回ごとに約 13 秒待たされていた

- 阻止されていたのは `torch\lib\torch_cuda.dll`（イベントログ。ポリシー ID {0283ac0f-…}＝スマート アプリ コントロール）。イベントログに残っている範囲では、**15:29:51〜16:56:58** に阻止が続き、それ以降は 1 回もない。Isaac Sim のログでは、15:06 の起動から torch の読み込みの失敗（WinError 4551）が出ている。それより前（#2〜#4、10/03 まで）の起動は約 8 秒だった
- 阻止は 1 回ごとに約 13 秒かかった（イベントの間隔がほぼ 13 秒ちょうど）。#6 の `sample` の起動（約 200 秒）の内訳

  | いつ | どこで torch を読み込んだか | 阻止の回数 | 時間 |
  |---|---|---|---|
  | Kit の起動の前 | `isaacsim/__init__.py`（Windows では、`torch\lib` の DLL をすべて先に読み込み、torch を import する。失敗は無視する） | 6 回 | 約 70 秒 |
  | Kit の拡張機能の読み込み中 | 非推奨の拡張機能 9 個（`isaacsim.core.prims`、`isaacsim.core.api`、`isaacsim.sensors.camera` など）。それぞれが torch を import し、失敗して起動に失敗した | 9 回 | 約 117 秒 |
  | 通常の起動 | - | - | 約 8 秒 |

- **今（17:24 以降）は阻止されていない**。torch は普通に import でき（1.5 秒、CUDA も使える）、Isaac Sim の起動は約 9 秒に戻った。こちらでは何も変えていない（スマート アプリ コントロールはオンのまま）。阻止が始まった・止まった理由は分からない。スマート アプリ コントロールは、署名のないファイルをクラウドの評価で判定するので、その評価（または問い合わせ）が一時的に変わった可能性がある（推測）

### 2. 対策ごとの起動時間（阻止されていない状態、各 3 回）

| 対策 | 起動の合計（中央値） | `SimulationApp` の生成 | 非推奨の拡張機能 | torch の読み込み | API の確認 |
|---|---|---|---|---|---|
| baseline | 8.6 秒 | 7.1 秒 | 13 個が有効 | あり | すべて OK |
| A：no_deprecated | 8.4 秒 | 6.7 秒 | 0 個 | あり（`isaacsim/__init__.py`） | すべて OK |
| B'：stub_torch | - | 7.3 秒 | 13 個が有効 | なし | **失敗**（`torch.Tensor` が無い） |
| A＋B' | - | 6.7 秒 | 0 個 | なし | **失敗**（同上） |

- A は、阻止されていない今は 0.2〜0.4 秒しか速くならない。阻止されている間なら、拡張機能の読み込み中の 9 回（約 117 秒）がなくなる見込み。ただし、`isaacsim/__init__.py` が torch の DLL を先に読み込むので、Kit の起動の前の約 70 秒は残る（`--trace-torch` で、最初に torch を import したのが `isaacsim/__init__.py` であることを確認した）
- B' は、非推奨でない拡張機能（`isaacsim.core.experimental.prims` の座標変換）が、`sys.modules` に torch があると `torch.Tensor` を参照するので、`GroundPlane` の作成で失敗した。中身のない torch では使えない

## 結論

1. **起動に約 200 秒かかった原因は、スマート アプリ コントロールによる torch の DLL（`torch_cuda.dll`）の阻止**。1 回ごとに約 13 秒待たされ、それが起動 1 回あたり約 15 回（Isaac Sim のパッケージの初期化で 6 回、非推奨の拡張機能で 9 回）起きていた
2. **今は阻止が止まり、起動は約 9 秒に戻った**。こちらの変更によるものではない。CLAUDE.md の記載（2 回目以降は約 8 秒）はそのままでよい
3. **推奨：今は起動の方法を変えない**（標準の experience のまま）
   - 阻止がない状態では、A の効果は 0.2〜0.4 秒しかない
   - 阻止が再発した場合の対処は、案 A（非推奨の拡張機能を除いた experience。`startup_common.write_experience_without_deprecated` で生成し、`SimulationApp(..., experience=...)` に渡す）。待ち時間のうち約 6 割（約 117 秒）を減らせる見込み。今回の API の確認はすべて通った
   - 選ばなかった案：B'（中身のない torch）は Isaac Sim の座標変換が動かなくなる。torch を外す案は、非推奨の拡張機能がアプリを終了させる（A と組み合わせれば避けられる可能性はあるが、`isaacsim-core` が torch に依存しており、環境を壊すので試していない）
4. **阻止の再発の見分け方**：Isaac Sim のログに `WinError 4551`（アプリケーション制御ポリシーによってこのファイルがブロックされました）が出る。または、イベントログ（Microsoft-Windows-CodeIntegrity/Operational）にイベント ID 3077 で `torch_cuda.dll` が出る

## 設計への影響

- 起動の方法は変えない。本実装も標準の experience で起動する
- 非推奨の API を使わない方針（CLAUDE.md）は、A の experience を使えば仕組みで守れる（非推奨の拡張機能が読み込まれないので、import すると失敗する）。ただし効果が小さいので、今は採用しない
- **残ったリスク**
  - スマート アプリ コントロールの阻止が再発する可能性がある（理由が分からない）。再発すると、起動 1 回あたり約 3 分待たされる。阻止は torch の DLL だけでなく、署名のない実行ファイル（#5 の llama.cpp）にも及ぶ
  - torch を使うライブラリ（学習済みのモデルなど）を本実装で使う場合は、阻止の間は使えない

## 決定事項（2026-10-12）

- **起動の方法**：正常に起動できている間は、標準の experience のまま進める。阻止が再発したら、案 A（非推奨の拡張機能を除いた experience）を検討する
- **CLAUDE.md**：「環境」の節に、スマート アプリ コントロールによる阻止の症状と、この記録への参照を追記した
