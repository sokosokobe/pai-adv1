"""台車の位置・向きの推定（core.perception.estimate_robot_pose。#26）の単体テスト。

幾何計算で深度画像を作る：いつもの部屋（床・天板・物体）に、台車の上面（角を落とした長方形）と、
上面の内側を斜めに隠すアーム（撮影姿勢）、離れた所の雑音を重ねる。
"""

import math

import numpy as np
import pytest

from restorebot.core.config import RobotGeometry
from restorebot.core.navigation import wrap_angle
from restorebot.core.perception import estimate_robot_pose
from restorebot.core.scene import HOME_V2
from restorebot.core.types import Observation, Pose2D
from tests.core.test_tasks import render_depth

CAM = HOME_V2.camera
GEOM = RobotGeometry()
REFERENCE = render_depth(HOME_V2, HOME_V2.tidy)


def _local(pose: Pose2D, z: float):
    """高さ z の水平面で、各画素の中心が台車の座標で (前, 左) のどこに当たるか。"""
    v, u = np.mgrid[0 : CAM.height, 0 : CAM.width].astype(np.float64) + 0.5
    d = CAM.position[2] - z
    x = CAM.position[0] + (u - CAM.cx) * d / CAM.fx
    y = CAM.position[1] - (v - CAM.cy) * d / CAM.fy
    dx, dy = x - pose.x, y - pose.y
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return c * dx + s * dy, -s * dx + c * dy


def _paint(depth: np.ndarray, inside: np.ndarray, z: float) -> None:
    d = CAM.position[2] - z
    depth[inside & (d < depth)] = d


def render_robot(pose: Pose2D, *, arm: bool = True, hide_front: float = 0.0, base=None) -> np.ndarray:
    """部屋の深度 base（省略時はいつもの部屋）に、台車（上面 0.27m）とアーム（0.9m）を重ねた深度。

    hide_front は上面の前側を隠す割合。
    """
    depth = (REFERENCE if base is None else base).copy()
    z = 0.27
    lx, ly = _local(pose, z)
    hl, hw, ch = GEOM.top_length / 2, GEOM.top_width / 2, 0.08  # 角を ch だけ落とす
    top = (np.abs(lx) <= hl) & (np.abs(ly) <= hw) & (np.abs(lx) + np.abs(ly) <= hl + hw - ch)
    _paint(depth, top, z)
    if arm:  # 撮影姿勢のアーム：台車の後ろ寄りから前左へ斜めに伸びる太い帯（上面の内側を大きく隠す）
        ax, ay = _local(pose, 0.9)
        t = np.clip(((ax + 0.25) * 0.55 + (ay + 0.05) * 0.25) / (0.55**2 + 0.25**2), 0.0, 1.0)
        dist = np.hypot(ax - (-0.25 + 0.55 * t), ay - (-0.05 + 0.25 * t))
        _paint(depth, dist <= 0.09, 0.9)
    if hide_front > 0.0:  # 上面の前側を、上にある何か（家具の天板など）が隠す
        hx, hy = _local(pose, 0.7)
        _paint(depth, (hx >= hl * (1 - 2 * hide_front)) & (np.abs(hy) <= hw + 0.05), 0.7)
    depth[100:104, 100:104] = CAM.position[2] - 0.27  # 離れた所の雑音（上面と同じ高さの小さな塊）
    return depth


def _obs(depth):
    return Observation(rgb=np.zeros((CAM.height, CAM.width, 3), np.uint8), depth=depth, time_s=0.0)


POSES = [
    Pose2D(0.0, -0.55, math.pi / 2),  # M2 の出発位置
    Pose2D(0.75, -0.03, math.pi / 2),  # 机の正面（視差でアームの像が上面を横切り、上面が 2 つの塊に分かれる）
    Pose2D(-0.53, -0.10, math.pi),  # 低い棚の正面
    Pose2D(0.30, 0.15, 0.4),  # 斜め
    Pose2D(-0.2, -0.6, -2.5),
]


@pytest.mark.parametrize("pose", POSES)
def test_estimates_position_and_yaw(pose):
    est = estimate_robot_pose(_obs(render_robot(pose)), REFERENCE, pose, CAM)
    assert est.pose is not None, est.reason
    assert math.hypot(est.pose.x - pose.x, est.pose.y - pose.y) < 0.02
    assert abs(math.degrees(wrap_angle(est.pose.yaw - pose.yaw))) < 3.0
    assert est.size[0] == pytest.approx(GEOM.top_length, abs=0.03)
    assert est.size[1] == pytest.approx(GEOM.top_width, abs=0.03)


def test_yaw_180_ambiguity_resolved_by_odometry():
    pose = Pose2D(0.3, 0.15, 0.4)
    depth = render_robot(pose)
    # オドメトリの向きが 20° ずれていても、近い方（正しい向き）を選ぶ
    near = estimate_robot_pose(_obs(depth), REFERENCE, Pose2D(0.3, 0.15, 0.4 + math.radians(20)), CAM)
    assert abs(wrap_angle(near.pose.yaw - 0.4)) < math.radians(3)
    # オドメトリが逆向き（180° 違う）なら、長方形は同じなので逆向きを答える
    flipped = estimate_robot_pose(_obs(depth), REFERENCE, Pose2D(0.3, 0.15, 0.4 + math.pi), CAM)
    assert abs(wrap_angle(flipped.pose.yaw - (0.4 + math.pi))) < math.radians(3)


def test_odometry_drift_is_measured():
    true = Pose2D(0.75, -0.03, math.pi / 2)
    odom = Pose2D(0.80, 0.0, math.pi / 2 + math.radians(4))  # 車輪のずれが溜まったオドメトリ
    est = estimate_robot_pose(_obs(render_robot(true)), REFERENCE, odom, CAM)
    assert math.hypot(est.pose.x - true.x, est.pose.y - true.y) < 0.02  # 写真は本当の位置を答える
    assert abs(math.degrees(wrap_angle(est.pose.yaw - true.yaw))) < 3.0


def test_fails_when_top_is_largely_hidden():
    pose = Pose2D(0.30, 0.15, 0.4)
    est = estimate_robot_pose(_obs(render_robot(pose, hide_front=0.3)), REFERENCE, pose, CAM)
    assert est.pose is None and "大きさ" in est.reason


def test_fails_without_robot():
    est = estimate_robot_pose(_obs(REFERENCE.copy()), REFERENCE, POSES[0], CAM)
    assert est.pose is None and est.n_pixels == 0 and est.reason


def test_min_area_rect_of_rotated_rectangle():
    from restorebot.core.perception import min_area_rect

    a = 0.3
    c, s = math.cos(a), math.sin(a)
    corners = [(1 + c * x - s * y, 2 + s * x + c * y) for x, y in [(-1, -0.5), (1, -0.5), (1, 0.5), (-1, 0.5)]]
    inner = [(1.0, 2.0), (1.2, 2.1)]
    center, axis, size = min_area_rect(np.asarray(corners + inner))
    assert center == pytest.approx((1.0, 2.0), abs=1e-9)
    assert axis == pytest.approx(a, abs=1e-9)
    assert size == pytest.approx((2.0, 1.0), abs=1e-9)
