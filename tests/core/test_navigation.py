"""restorebot.core.navigation（楽に届く範囲、家具の正面の姿勢）の単体テスト。"""

import math

import pytest

from restorebot.core.config import M1_BASE_POSE, REACH, ROBOT
from restorebot.core.navigation import facing_pose, in_reach
from restorebot.core.scene import HOME_V2, Surface
from restorebot.core.types import Pose2D

BASE = Pose2D(0.0, 0.0, 0.0)  # 原点で +x を向いた台車


@pytest.mark.parametrize(
    ("xy", "expected"),
    [
        ((0.70, 0.0), True),  # 正面
        ((REACH.min_distance - 0.01, 0.0), False),  # 近すぎる
        ((REACH.max_distance + 0.01, 0.0), False),  # 遠すぎる
        ((0.70 * math.cos(math.radians(29)), 0.70 * math.sin(math.radians(29))), True),
        ((0.70 * math.cos(math.radians(31)), 0.70 * math.sin(math.radians(31))), False),  # 正面から 30° を超える
        ((0.70 * math.cos(math.radians(-29)), 0.70 * math.sin(math.radians(-29))), True),
        ((-0.70, 0.0), False),  # 後ろ
    ],
)
def test_in_reach(xy, expected):
    assert in_reach(BASE, xy) is expected


def test_in_reach_follows_base_pose():
    base = Pose2D(1.0, 2.0, math.pi / 2)  # +y を向いている
    assert in_reach(base, (1.0, 2.7))
    assert not in_reach(base, (1.7, 2.0))  # 台車から見て右（-90°）


def test_in_reach_handles_angle_wrap():
    base = Pose2D(0.0, 0.0, math.pi)  # -x を向いている（yaw = ±π の境目）
    assert in_reach(base, (-0.7, 0.05))
    assert in_reach(base, (-0.7, -0.05))


@pytest.mark.parametrize(
    ("side", "expected"),
    [
        ("-x", Pose2D(-0.5 - 0.57, 0.1, 0.0)),
        ("+x", Pose2D(0.5 + 0.57, 0.1, math.pi)),
        ("-y", Pose2D(0.1, -0.3 - 0.57, math.pi / 2)),
        ("+y", Pose2D(0.1, 0.3 + 0.57, -math.pi / 2)),
    ],
)
def test_facing_pose(side, expected):
    s = Surface(center=(0.0, 0.0), size=(1.0, 0.6), key="t", z=0.5)
    pose = facing_pose(s, side, lateral=0.1)
    assert (pose.x, pose.y, pose.yaw) == pytest.approx((expected.x, expected.y, expected.yaw))


def test_center_to_edge_is_half_length_plus_gap():
    assert ROBOT.center_to_edge == pytest.approx(0.47 + 0.10)


def test_m1_base_pose_faces_desk_front_with_gap():
    desk = HOME_V2.surfaces["desk"]
    expected = facing_pose(desk, desk.front, lateral=M1_BASE_POSE.x)
    assert (M1_BASE_POSE.x, M1_BASE_POSE.y, M1_BASE_POSE.yaw) == pytest.approx((expected.x, expected.y, expected.yaw))


def test_m1_base_pose_reaches_all_desk_home_positions():
    for key, p in HOME_V2.tidy.items():
        if p.surface == "desk":
            assert in_reach(M1_BASE_POSE, (p.x, p.y)), key
