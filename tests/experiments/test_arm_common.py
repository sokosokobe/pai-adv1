"""scripts/experiments/arm_common.py（アームの検証用の計算）の単体テスト。Isaac Sim 不要。"""

import math

import numpy as np
import pytest

from experiments.arm_common import (
    compose_pose,
    is_success,
    quat_angle_error_wxyz,
    quat_from_yaw_wxyz,
    quat_mul_wxyz,
    reach_grid,
    rotate_vec_wxyz,
    world_from_base,
)

IDENTITY = (1.0, 0.0, 0.0, 0.0)


# ---- クォータニオン（並び：w, x, y, z）----


def test_quat_from_yaw_wxyz():
    q = quat_from_yaw_wxyz(math.pi / 2)
    assert q == pytest.approx((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)))


def test_rotate_vec_by_yaw_90():
    v = rotate_vec_wxyz(quat_from_yaw_wxyz(math.pi / 2), (1.0, 0.0, 0.0))
    assert v == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)


def test_quat_mul_composes_rotations():
    q = quat_mul_wxyz(quat_from_yaw_wxyz(0.3), quat_from_yaw_wxyz(0.4))
    assert quat_angle_error_wxyz(q, quat_from_yaw_wxyz(0.7)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("q1", "q2", "expected"),
    [
        (IDENTITY, IDENTITY, 0.0),
        (IDENTITY, (-1.0, 0.0, 0.0, 0.0), 0.0),  # q と -q は同じ姿勢
        (IDENTITY, (0.0, 1.0, 0.0, 0.0), math.pi),  # x 軸まわり 180°
        (quat_from_yaw_wxyz(0.1), quat_from_yaw_wxyz(-0.2), 0.3),
    ],
)
def test_quat_angle_error(q1, q2, expected):
    assert quat_angle_error_wxyz(q1, q2) == pytest.approx(expected, abs=1e-9)


# ---- 姿勢の合成（台車 → アームの根元 → 目標）----


def test_compose_pose_translation_is_rotated_by_parent():
    parent = ((1.0, 2.0, 0.0), quat_from_yaw_wxyz(math.pi / 2))
    child = ((0.5, 0.0, 0.3), IDENTITY)
    pos, quat = compose_pose(parent, child)
    assert pos == pytest.approx((1.0, 2.5, 0.3), abs=1e-12)
    assert quat_angle_error_wxyz(quat, quat_from_yaw_wxyz(math.pi / 2)) == pytest.approx(0.0, abs=1e-9)


def test_world_from_base_pose2d():
    # 台車 (x=1, y=-0.5, yaw=30°) から見て前方 0.6m、高さ 0.7m の点
    p = world_from_base((1.0, -0.5, math.radians(30)), (0.6, 0.0, 0.7))
    assert p == pytest.approx((1.0 + 0.6 * math.cos(math.radians(30)), -0.5 + 0.6 * math.sin(math.radians(30)), 0.7))


# ---- 到達範囲の格子点 ----


def test_reach_grid_size_and_values():
    grid = reach_grid(distances=[0.4, 0.6], heights=[0.5, 0.7, 0.9], angles_deg=[-45, 0, 45])
    assert len(grid) == 2 * 3 * 3
    first = grid[0]
    assert (first.distance, first.height, first.angle_deg) == (0.4, 0.5, -45)
    a = math.radians(-45)
    assert first.point_in_base == pytest.approx((0.4 * math.cos(a), 0.4 * math.sin(a), 0.5))


def test_reach_grid_point_on_forward_axis():
    (pt,) = reach_grid(distances=[0.5], heights=[0.6], angles_deg=[0])
    np.testing.assert_allclose(pt.point_in_base, (0.5, 0.0, 0.6), atol=1e-12)


# ---- 成功の判定 ----


@pytest.mark.parametrize(
    ("pos_err", "ang_err_deg", "expected"),
    [(0.005, 2.0, True), (0.02, 2.0, False), (0.005, 10.0, False), (None, None, False)],
)
def test_is_success(pos_err, ang_err_deg, expected):
    ang = None if ang_err_deg is None else math.radians(ang_err_deg)
    assert is_success(pos_err, ang, pos_tol=0.01, ang_tol=math.radians(5)) is expected
