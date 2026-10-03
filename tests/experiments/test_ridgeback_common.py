"""scripts/experiments/ridgeback_common.py（台車の検証用の計算）の単体テスト。Isaac Sim 不要。"""

import math

import numpy as np
import pytest

from experiments.ridgeback_common import (
    body_to_world_velocity,
    classify_obstacle_outcome,
    max_deviation,
    pose_error,
    settle_time,
    step_toward,
    velocity_command,
    wrap_angle,
    yaw_from_quat_wxyz,
)

# ---- 角度の正規化 ----


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (0.0, 0.0),
        (math.pi / 2, math.pi / 2),
        (3 * math.pi / 2, -math.pi / 2),
        (-3 * math.pi / 2, math.pi / 2),
        (2 * math.pi, 0.0),
        (math.pi + 0.1, -math.pi + 0.1),
    ],
)
def test_wrap_angle(angle, expected):
    assert wrap_angle(angle) == pytest.approx(expected)


def test_wrap_angle_range_is_half_open():
    # π と -π は同じ向き。値域は (-π, π] とする
    assert wrap_angle(math.pi) == pytest.approx(math.pi)
    assert wrap_angle(-math.pi) == pytest.approx(math.pi)


# ---- クォータニオン（Isaac Sim の並び：w, x, y, z）→ yaw ----


@pytest.mark.parametrize("yaw", [0.0, 0.3, math.pi / 2, -2.5, math.pi])
def test_yaw_from_quat_wxyz(yaw):
    q_wxyz = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    assert wrap_angle(yaw_from_quat_wxyz(q_wxyz) - yaw) == pytest.approx(0.0, abs=1e-9)


def test_yaw_from_quat_wxyz_is_not_xyzw():
    # 並び順を取り違えると結果が変わることを確認（z 軸 90° 回転）
    s = math.sqrt(0.5)
    assert yaw_from_quat_wxyz((s, 0.0, 0.0, s)) == pytest.approx(math.pi / 2)


# ---- 台車基準 → ワールド基準の速度変換 ----


def test_body_to_world_velocity_identity():
    assert body_to_world_velocity(1.0, 0.0, 0.0) == pytest.approx((1.0, 0.0))


def test_body_to_world_velocity_facing_plus_y():
    # 台車が +y を向いている（yaw=90°）とき、前進はワールドの +y、左方向はワールドの -x
    assert body_to_world_velocity(1.0, 0.0, math.pi / 2) == pytest.approx((0.0, 1.0), abs=1e-12)
    assert body_to_world_velocity(0.0, 1.0, math.pi / 2) == pytest.approx((-1.0, 0.0), abs=1e-12)


# ---- 姿勢の誤差 ----


def test_pose_error_wraps_yaw():
    # 179° → -179° は 2° の差（358° ではない）
    err = pose_error(current=(0.0, 0.0, math.radians(179)), target=(1.0, 2.0, math.radians(-179)))
    assert err.dx == pytest.approx(1.0)
    assert err.dy == pytest.approx(2.0)
    assert err.dyaw == pytest.approx(math.radians(2))
    assert err.distance == pytest.approx(math.sqrt(5))


# ---- 速度指令（比例制御＋飽和）----


def test_velocity_command_saturates_but_keeps_direction():
    vx, vy, wz = velocity_command(
        current=(0.0, 0.0, 0.0), target=(3.0, 4.0, 0.0), kp_lin=10.0, kp_ang=1.0, v_max=0.5, w_max=1.0
    )
    assert math.hypot(vx, vy) == pytest.approx(0.5)
    assert vy / vx == pytest.approx(4.0 / 3.0)  # 向きは目標の方向のまま
    assert wz == 0.0


def test_velocity_command_proportional_when_close():
    vx, vy, wz = velocity_command(
        current=(0.0, 0.0, 0.0), target=(0.01, 0.0, 0.1), kp_lin=2.0, kp_ang=3.0, v_max=0.5, w_max=1.0
    )
    assert (vx, vy, wz) == pytest.approx((0.02, 0.0, 0.3))


def test_velocity_command_turns_the_short_way():
    *_, wz = velocity_command(
        current=(0.0, 0.0, math.radians(170)),
        target=(0.0, 0.0, math.radians(-170)),
        kp_lin=1.0,
        kp_ang=1.0,
        v_max=0.5,
        w_max=1.0,
    )
    assert wz > 0  # 340° 逆回りせず、+20° 回る


# ---- 所要時間（許容誤差に入り、その後ずっと入ったままになった最初の時刻）----


def test_settle_time_ignores_overshoot():
    t = [0.0, 1.0, 2.0, 3.0, 4.0]
    pos = [1.0, 0.005, 0.05, 0.004, 0.003]  # t=1 で一度入るが t=2 で外れる
    yaw = [0.0] * 5
    assert settle_time(t, pos, yaw, pos_tol=0.01, yaw_tol=0.02) == 3.0


def test_settle_time_requires_both_position_and_yaw():
    t = [0.0, 1.0, 2.0]
    pos = [0.0, 0.0, 0.0]
    yaw = [1.0, 0.5, 0.01]
    assert settle_time(t, pos, yaw, pos_tol=0.01, yaw_tol=0.02) == 2.0


def test_settle_time_none_when_not_settled_at_end():
    t = [0.0, 1.0, 2.0]
    assert settle_time(t, [1.0, 0.0, 0.5], [0.0, 0.0, 0.0], pos_tol=0.01, yaw_tol=0.02) is None


# ---- 揺れ（基準位置からの最大のずれ）----


def test_max_deviation():
    points = np.array([[0.0, 0.0, 1.0], [0.0, 0.03, 1.0], [0.0, 0.0, 0.96]])
    assert max_deviation(points, reference=points[0]) == pytest.approx(0.04)


# ---- 位置指令を一定速度で目標へ近づける（ランプ）----


def test_step_toward_limits_linear_and_angular_step():
    nxt = step_toward(current=(0.0, 0.0, 0.0), goal=(3.0, 4.0, 1.0), max_lin_step=0.5, max_ang_step=0.1)
    assert nxt[0] == pytest.approx(0.3)
    assert nxt[1] == pytest.approx(0.4)
    assert nxt[2] == pytest.approx(0.1)


def test_step_toward_snaps_to_goal_when_close():
    assert step_toward((0.0, 0.0, 0.0), (0.01, 0.0, -0.05), max_lin_step=0.5, max_ang_step=0.1) == pytest.approx(
        (0.01, 0.0, -0.05)
    )


def test_step_toward_turns_the_short_way_and_stays_continuous():
    # 170° → -170° は +20° 回る。yaw は連続値のまま（関節角として使うため ±π で折り返さない）
    nxt = step_toward((0.0, 0.0, math.radians(170)), (0.0, 0.0, math.radians(-170)), 1.0, math.radians(5))
    assert nxt[2] == pytest.approx(math.radians(175))


# ---- 障害物にぶつかったときの挙動の分類 ----


@pytest.mark.parametrize(
    ("box_moved", "base_progress", "expected"),
    [
        (0.50, 0.90, "push"),  # 箱が動いた → 押す
        (0.00, 0.10, "stop"),  # 箱も台車も止まった → 止まる
        (0.00, 1.00, "pass_through"),  # 箱は動かず台車は目標まで行った → すり抜ける
    ],
)
def test_classify_obstacle_outcome(box_moved, base_progress, expected):
    assert classify_obstacle_outcome(box_moved=box_moved, base_progress=base_progress) == expected
