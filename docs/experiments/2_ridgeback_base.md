# #2 RidgebackFranka の読み込みと台車の動かし方

- **Issue**：#2
- **日付**：2026-10-03
- **タイムボックス**：2.5時間 ／ **実績**：約0.5時間（14:28〜14:49）
- **検証用コード**：`scripts/experiments/ridgeback_inspect.py`、`scripts/experiments/ridgeback_base_motion.py`（計算部分は `ridgeback_common.py`、単体テストは `tests/experiments/test_ridgeback_common.py`）
- **証拠**：`outputs/m0_2/`（`inspect_*/`、`motion_*/` に JSON・時系列 CSV・前後の画像）

## 目的

1. RidgebackFranka を Isaac Sim 6.1 で読み込み、移動ロボットとして使えるか
2. 2種類のアセット（`Robots/` と `Robots_Multiphysics/`）のどちらを使うか
3. 台車への指令は、位置指令と速度指令のどちらが扱いやすいか（到達精度・所要時間・アームの揺れ）
4. 障害物に当たったときに、止まる／押す／すり抜けるのどれになるか

## 方法

- Isaac Sim 6.1.0（pip 版）、headless、物理 60Hz（PhysX）、地面＋ドームライト
- アセット（アセットサーバーのパス）
  - Multiphysics 版：`/Isaac/Robots_Multiphysics/Clearpath/RidgebackFranka/ridgeback_franka.usda`
  - 従来版：`/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd`
- API：`isaacsim.core.experimental.prims.Articulation`（`get_dof_*`、`set_dof_position_targets`、`set_dof_velocity_targets`、`switch_dof_control_mode`）。非推奨の API は使っていない
- 関節の一覧：USD 上の関節・リンクを走査（instanceable な参照の中まで見るため `Usd.TraverseInstanceProxies()` を使用）し、`Articulation` から DOF の可動範囲・ゲイン・最大力を取得
- 台車の関節の向き：台車の仮想関節を1つずつ動かし、台車リンク（`base_link`）のワールド姿勢の変化を測った
- 指令方法（台車の3つの仮想関節 x, y, yaw に対して）
  - **position_step**：位置指令。目標値を一度に与える
  - **position_ramp**：位置指令。目標値を一定速度（0.5 m/s、1.0 rad/s）で目標へ動かす
  - **velocity**：速度指令（`switch_dof_control_mode("velocity")` で剛性を 0 にする）。現在姿勢からの比例制御（ゲイン 2.0）で速度を作り、0.5 m/s・1.0 rad/s で飽和させる
- 3パターン：1m 前進、90°回転、斜め移動（x, y とも +1m、向きは変えない）。各 8 秒
- 計測：最終誤差、所要時間（誤差が 1cm・1° 以内に入り、以後ずっと入ったままになった時刻）、目標の行き過ぎ、最高速度・最大加速度、アームの揺れ（台車から見た手先 `panda_hand` の位置のずれの最大値）
- 障害物：一辺 0.4m の箱を台車の前方（中心 x=1.2m）に置き、2m 前進を指令。箱は 5kg、50kg、固定（剛体にしない）の3種類

## 結果

### 1. 関節の一覧（12自由度：台車3＋アーム7＋指2）

台車は車輪で走るモデルではなく、**ワールドに固定された仮想関節 3つ（x 直動・y 直動・z 回転）で動かすモデル**だった。単位は SI（直動 m、回転 rad。剛性・減衰は rad 基準）。

| # | 関節 | 種類 | 可動範囲 | 剛性 | 減衰 | 最大力 / トルク | 最大速度 |
|---|---|---|---|---|---|---|---|
| 0 | `dummy_base_prismatic_x_joint` | 直動 | ±600.28 m | 1e4 | 1e3 | 1000 N | 15 |
| 1 | `dummy_base_prismatic_y_joint` | 直動 | ±600.28 m | 1e4 | 1e3 | 1000 N | 15 |
| 2 | `dummy_base_revolute_z_joint` | 回転 | ±34393.5°（実質無制限） | 5.73e8 | 5.73e6 | 4800 N·m | 15 |
| 3 | `panda_joint1` | 回転 | -170.0° 〜 170.0° | 5.73e8 | 5.73e6 | 5220 | 2.17 |
| 4 | `panda_joint2` | 回転 | -105.0° 〜 105.0° | 5.73e8 | 5.73e6 | 5220 | 2.17 |
| 5 | `panda_joint3` | 回転 | -170.0° 〜 170.0° | 5.73e8 | 5.73e6 | 5220 | 2.17 |
| 6 | `panda_joint4` | 回転 | -171.9° 〜 5.0° | 5.73e8 | 5.73e6 | 5220 | 2.17 |
| 7 | `panda_joint5` | 回転 | -170.0° 〜 170.0° | 5.73e8 | 5.73e6 | 720 | 2.61 |
| 8 | `panda_joint6` | 回転 | -5.0° 〜 171.9° | 5.73e8 | 5.73e6 | 720 | 2.61 |
| 9 | `panda_joint7` | 回転 | -170.0° 〜 170.0° | 5.73e8 | 5.73e6 | 720 | 2.61 |
| 10 | `panda_finger_joint1` | 直動 | 0 〜 0.04 m | 1e7 | 1e5 | 1200 N | 0.2 |
| 11 | `panda_finger_joint2` | 直動 | 0 〜 0.04 m | 1e7 | 1e5 | 1200 N | 0.2 |

- ドライブはすべて `force` 型の位置ドライブ（アセットの既定）
- 固定関節：`FixedJoint`（`world` リンクをワールドに固定）、`base_to_arm_mount_joint`、`panda_arm_mount_joint`、`panda_hand_joint`、`panda_hand_endeffector_joint`、`front_laser_joint`、`rear_laser_joint`
- 衝突形状があるリンク：`base_link`（165.3 kg）、`panda_link0`〜`7`、`panda_hand`、左右の指、前後のレーザー。仮想リンク（`world`、`dummy_base_x/y`）は質量 1e-4 kg で衝突形状なし
- **台車の仮想関節の向き**：x・y 関節はワールド基準（yaw を 90° 回した後に x 関節を 0.5m 動かしても、ワールドの +x に 0.5m 動いた）。したがって **関節角 (x, y, yaw) = 台車のワールド姿勢** で、ワールド座標の目標をそのまま関節の目標にできる。台車基準の速度指令（cmd_vel）を使う場合は、yaw で回してから与える必要がある
- **アセットに含まれる関節摩擦**：台車の x・y 関節に `physxJoint:jointFriction = 0.1`（PhysX の関節摩擦係数）が設定されている（`payloads/Physics/physx.usda`）。影響は下の 3. を参照

### 2. 2種類のアセットの比較

| 項目 | Multiphysics 版（`.usda`） | 従来版（`.usd`） |
|---|---|---|
| DOF（名前・種類・可動範囲・ゲイン・最大力・最大速度） | 12 | 12。**すべて同じ** |
| 衝突形状があるリンク・質量 | — | **同じ** |
| 台車の仮想関節の向き | ワールド基準 | **同じ** |
| ファイル形式 | テキスト（`.usda`）。中身を直接読める | バイナリ（`.usd`） |
| 物理エンジンの切り替え | variant `Physics` で `physx` / `physics` / `none` を選べ、MuJoCo 用の payload もある | 未確認（バイナリのため中身を読んでいない） |
| 6.1 のサンプルとの一貫性 | Franka の Pick and Place サンプルと同じ `Robots_Multiphysics` 系統 | — |
| 台車の関節摩擦（`physxJoint:jointFriction`） | x・y に 0.1 | 未確認 |

※ 以降の動作の計測（3. と 4.）は Multiphysics 版だけで行った。

### 3. 指令方法の比較

条件：v_max = 0.5 m/s、w_max = 1.0 rad/s（position_ramp と velocity で共通）、各 8 秒。所要時間は「誤差が 1cm・1° 以内に入り、以後ずっと入ったままになった時刻」。手先の揺れは、台車から見た `panda_hand` の位置のずれの最大値。「最高速度」は台車の x・y 関節の速度。90°回転でも 0 にならないのは、回転中に台車の位置がわずかに揺れるため（最終誤差は 0）。

#### アセットの関節摩擦（0.1）のままの場合

| パターン | 指令方法 | 最終誤差 [mm] | 所要時間 [s] | 行き過ぎ [cm] | 最高速度 [m/s] | 最大加速度 [m/s²] | 手先の揺れ [mm] |
|---|---|---|---|---|---|---|---|
| 1m 前進 | position_step | 12.4 | 未到達 | 25.6 | 2.45 | 6.4 | 0.0 |
| 1m 前進 | position_ramp | 10.5 | 未到達 | 1.2 | 0.64 | 2.3 | 0.0 |
| 1m 前進 | velocity | **83.4** | 未到達 | 0.0 | **0.31** | 1.5 | 0.0 |
| 90°回転 | position_step | 19.3 | 未到達 | 0.0 | 0.55 | 10.5 | 4.1 |
| 90°回転 | position_ramp | 2.4 | 1.57 | 0.0 | 0.03 | 0.9 | 0.5 |
| 90°回転 | velocity | 1.4 | 2.73 | 0.0 | 0.04 | 1.2 | 0.3 |
| 斜め (1m, 1m) | position_step | 12.7 | 未到達 | 30.2 | 3.39 | 9.8 | 0.1 |
| 斜め (1m, 1m) | position_ramp | 12.9 | 未到達 | 0.0 | 0.64 | 2.3 | 0.0 |
| 斜め (1m, 1m) | velocity | **197.2** | 未到達 | 0.0 | **0.22** | 1.1 | 0.0 |

- 並進では、どの指令方法も目標の手前で**完全に止まった**（速度が厳密に 0 になる。静止摩擦による固着）。速度指令は 0.5 m/s の指令に対して 0.22〜0.31 m/s しか出ず、8〜20cm 手前で止まった
- 原因の切り分け
  - 台車本体と地面の衝突を無効にしても（`--ground-contact filtered`）結果は 4桁まで同じ → **地面との摩擦ではない**
  - `Articulation.set_dof_friction_properties(0, 0, 0)` でも変わらない（このセッターは静止・動摩擦の「力」を設定するもので、アセットの `physxJoint:jointFriction`（係数）とは別のパラメータと考えられる）
  - **物理開始前に USD の `physxJoint:jointFriction` を 0 にすると解消**（下表）。PhysX の関節摩擦は「関節が支える力 × 係数」で、台車の直動関節はロボット全体（約 190 kg、約 1860 N）を支えるため、摩擦は約 186 N になる。位置指令の残り誤差（1e4 N/m × 1.2 cm ≈ 124 N）、速度指令の停止位置（減衰 1e3 × ゲイン 2 × 8 cm ≈ 166 N）と整合する

#### 台車の関節摩擦を 0 にした場合

| パターン | 指令方法 | 最終誤差 [mm] | 所要時間 [s] | 行き過ぎ [cm] | 最高速度 [m/s] | 最大加速度 [m/s²] | 手先の揺れ [mm] |
|---|---|---|---|---|---|---|---|
| 1m 前進 | position_step | 0.8 | 2.62 | **43.5** | **2.69** | 5.4 | 0.0 |
| 1m 前進 | position_ramp | 0.0 | 2.48 | 3.1 | 0.64 | 2.3 | 0.0 |
| 1m 前進 | velocity | 0.1 | 2.50 | 0.5 | 0.50 | 2.4 | 0.0 |
| 90°回転 | position_step | 1.3 | 0.90 | 0.0 | 0.61 | **11.6** | 4.1 |
| 90°回転 | position_ramp | 0.0 | 1.57 | 0.0 | 0.06 | 0.8 | 0.5 |
| 90°回転 | velocity | 0.0 | 2.73 | 0.0 | 0.06 | 0.3 | 0.2 |
| 斜め (1m, 1m) | position_step | 0.5 | 2.72 | **61.5** | **3.80** | 7.7 | 0.1 |
| 斜め (1m, 1m) | position_ramp | 0.0 | 3.32 | 3.1 | 0.64 | 2.3 | 0.0 |
| 斜め (1m, 1m) | velocity | 0.1 | 3.33 | 0.5 | 0.50 | 2.4 | 0.0 |

- **position_step**：最大力 1000 N・質量約 190 kg で一気に加速し、2.7〜3.8 m/s に達して 43〜62 cm 行き過ぎる。実機ではあり得ない動きで、動画にも使えない
- **position_ramp**：最終誤差 0、所要時間は velocity と同程度。ただし PD の追従遅れを取り戻すため、最高速度が指令（0.5 m/s）を超えて 0.64 m/s になり、3 cm 行き過ぎる
- **velocity**：指令どおり 0.5 m/s で動き、最終誤差 0.1 mm、行き過ぎ 0.5 cm。回転は比例制御で目標付近が遅くなるため、所要時間が 2.73 s と ramp より長い（ゲインで調整できる）
- **アームの揺れ**：どの条件でも手先のずれは最大 4 mm（position_step の回転）で、ほぼ 0。アームの関節ドライブの剛性が非常に高い（5.73e8 N·m/rad）ため、シミュレーション上はアームが剛体のように振る舞う。揺れは問題にならないが、実機らしい揺れは再現されない

### 4. 障害物に当たったときの挙動

条件：台車の関節摩擦 0、2m 前進を指令。箱は一辺 0.4 m、中心 x = 1.2 m（台車の前面は x ≈ 0.47 m、箱の後面は x = 1.0 m）。

| 箱 | 指令方法 | 結果 | 箱の移動 [m] | 台車の最終 x [m] | 最終誤差 [cm] | 行き過ぎ [cm] | 最高速度 [m/s] |
|---|---|---|---|---|---|---|---|
| 5 kg | position_step | 押す | 2.68 | 2.001 | 0.1 | 106.1 | 3.94 |
| 5 kg | position_ramp | 押す | 1.50 | 2.000 | 0.0 | 2.9 | 0.64 |
| 5 kg | velocity | 押す | 1.47 | 1.992 | 0.8 | 0.0 | 0.50 |
| 50 kg | position_step | 押す | 2.13 | 2.003 | 0.3 | 57.7 | 3.05 |
| 50 kg | position_ramp | 押す | 1.49 | 2.000 | 0.0 | 1.2 | 0.64 |
| 50 kg | velocity | 押す | 1.33 | 1.858 | 14.2 | 0.0 | 0.50 |
| 固定 | position_step | 止まる | 0.00 | 0.526 | 147.4 | 0.0 | 2.35 |
| 固定 | position_ramp | 止まる | 0.00 | 0.526 | 147.4 | 0.0 | 0.64 |
| 固定 | velocity | 止まる | 0.00 | 0.526 | 147.4 | 0.0 | 0.50 |

- **すり抜けはない**。台車本体（`base_link`）の衝突形状が効いており、固定の壁には接触した位置（めり込み 8 mm 以下）で止まる
- **動かせる物は押してしまう**。台車のドライブは最大 1000 N まで出るので、50 kg の箱も押し出した。障害物を検知して止まる仕組みはない
- position_step は 5 kg の箱を 2.7 m 弾き飛ばした
- 指令方法による違い（壁に当たった後）
  - position_ramp：目標が先へ進み続けるため、誤差が積み上がり、壁を最大 1000 N で押し続ける。壁が無くなると、先へ進んだ目標へ一気に飛び出す（積分のワインドアップと同じ状態）
  - velocity：ドライブの力は「減衰 × (指令速度 − 実速度)」で、最大 1e3 × 0.5 = 500 N に収まる。目標を先へ進めないので、障害物が無くなっても飛び出さない
  - velocity は 50 kg の箱を押すと 14 cm 手前で止まった。目標付近では比例制御の指令速度が小さく、箱の摩擦に勝てないため

### 5. 画像

すべて自分で開いて目視確認した。画像の向きの確認用に、+x 側（x = 3.2 m）に赤、+y 側（y = 2.2 m）に緑の目印を置いている（見た目だけで衝突なし）。カメラは (-4.5, -7, 8) から (1.2, 0.8, 0) を見下ろしているため、+x は画面の右、+y は画面の左奥に写る。

| 画像 | 確認したこと |
|---|---|
| `outputs/m0_2/inspect_multiphysics/overview.png`、`inspect_legacy/overview.png` | 2種類のアセットとも正しく読み込まれ、見た目も同じ。アームは初期姿勢で横に大きく伸びている |
| `outputs/m0_2/motion_multiphysics/diagonal_1m_1m_velocity_{before,after}.png` | 台車が原点から +x・+y 方向（画面の奥）へ移動した。**指令の向きと画像の向きが一致し、上下・左右の反転はない** |
| `outputs/m0_2/motion_multiphysics_nofriction/rotate_90deg_velocity_after.png` | 前面（黄色）が +x 側から +y 側へ向きを変えた（反時計回り＝ yaw の正の向き） |
| `outputs/m0_2/motion_multiphysics_obstacle_static_nofriction/obstacle_forward_2m_velocity_after.png` | 台車が固定の箱に接した位置で止まっている（めり込み・すり抜けなし） |
| `outputs/m0_2/motion_multiphysics_obstacle_light_nofriction/obstacle_forward_2m_position_step_after.png` | 5 kg の箱が赤い目印の先（x = 3.88 m）まで弾き飛ばされた |

各試行の前後の画像は `outputs/m0_2/motion_*/<パターン>_<指令方法>_{before,after}.png`、時系列は同じフォルダの `<パターン>_<指令方法>.csv`、集計は `summary.json` にある。同じ条件で2回実行し、すべての数値が一致すること（決定論的であること）を確認した。

再現手順:

```powershell
.venv\Scripts\python.exe scripts\experiments\ridgeback_inspect.py --asset multiphysics   # legacy も同様
.venv\Scripts\python.exe scripts\experiments\ridgeback_base_motion.py                     # アセット既定の摩擦
.venv\Scripts\python.exe scripts\experiments\ridgeback_base_motion.py --base-joint-friction zero
.venv\Scripts\python.exe scripts\experiments\ridgeback_base_motion.py --base-joint-friction zero --obstacle light   # heavy / static
.venv\Scripts\python.exe scripts\experiments\ridgeback_base_motion.py --ground-contact filtered   # 地面との摩擦の切り分け
```

## 結論

1. **RidgebackFranka は Isaac Sim 6.1 で移動マニピュレータとして使える**。headless で読み込め、12 自由度（台車 3＋アーム 7＋指 2）を `Articulation` から操作できる
2. **アセットは Multiphysics 版（`Robots_Multiphysics/.../ridgeback_franka.usda`）を使う**。DOF・ドライブの設定・衝突形状・質量は従来版と同じで、6.1 のサンプルと同じ系統、テキスト形式で中身を確認しやすく、物理エンジンの切り替え（variant）にも対応しているため
3. **台車の指令方法の推奨案：速度指令**
   - 台車の3つの仮想関節（ワールド基準の x, y, yaw）を `switch_dof_control_mode("velocity")` で速度制御にし、上位の制御（今回は比例制御＋飽和。後で経路追従に置き換える）が作った速度を `set_dof_velocity_targets` で与える
   - **前提として、シーンを読み込んだ後・物理を開始する前に、台車の x・y 関節の `physxJoint:jointFriction` を 0 にする**（アセットのままでは速度指令が 8〜20 cm 手前で止まる）
   - 選んだ理由
     - 指令どおりの速度（0.5 m/s）で動き、行き過ぎ 0.5 cm、最終誤差 0.1 mm。動画で実機らしく見える
     - 障害物に当たっても押す力が 500 N 以下に収まり、障害物が無くなった後に飛び出さない
     - 経路追従やナビゲーションの出力（速度）とそのままつながる
   - 選ばなかった案
     - position_step：2.7〜3.8 m/s まで加速して 43〜62 cm 行き過ぎる。実用にならない
     - position_ramp：精度・所要時間は速度指令と同等で、代案になりうる。ただし速度が指令を超え（0.64 m/s）、障害物で止められると目標だけ先へ進んで最大力で押し続け、解放されると飛び出す
4. **障害物はすり抜けない**。固定物には接触して止まり、動かせる物（50 kg まで確認）は押してしまう

## 設計への影響

- **シーンの読み込み処理に「台車の x・y 関節の `physxJoint:jointFriction` を 0 にする」を入れる**（`src/restorebot/sim/` の担当）。忘れると速度指令で目標に届かない
- **台車の姿勢は関節角から直接得られる**：x・y 関節はワールド基準なので、関節角 (x, y, yaw) ＝ 台車のワールド姿勢。M1〜M2 の真値として使える。台車基準の速度（cmd_vel）を使う場合は yaw で回してから与える（`body_to_world_velocity`）
- **台車の制御則は Isaac Sim 非依存のロジックとして `src/restorebot/core/` に置ける**：今回の `velocity_command`・`pose_error`・`wrap_angle` は、本実装では設計に合わせて書き直し、単体テストを付けて移す
- **ナビゲーション層で障害物を避ける必要がある**：台車は動かせる物を最大 500〜1000 N で押してしまい、検知する仕組みもない。片付け対象の物を台車で押し飛ばさないよう、経路計画で物体・家具から距離をとる。追従誤差が大きいまま進まない状態（スタック）を検知して止める仕組みも検討する
- **アームの揺れは考慮しなくてよい**：アームのドライブ剛性が非常に高く、台車の加減速でも手先はほとんど動かない（物を持った状態は #4 で確認する）
- **速度の上限**：今回は 0.5 m/s・1.0 rad/s。回転は比例制御で目標付近が遅くなるため、回転の所要時間（2.7 s）はゲインで短くできる
- **未解決・申し送り**
  - 実機の Ridgeback は車輪で動くが、このモデルは仮想関節で動く。床との摩擦や滑りは再現されない（デモの用途では問題ない）
  - `Articulation.set_dof_friction_properties` と `physxJoint:jointFriction` の関係は未確認（今回はセッターでは摩擦が消えなかった）
  - #3（アームの動作計画）では、アームの根元（`panda_link0`）の位置を台車の姿勢から計算する必要がある
