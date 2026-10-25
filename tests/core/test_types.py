"""restorebot.core.types（型と、クォータニオンの並びの変換）の単体テスト。"""

import math

import pytest

from restorebot.core.types import (
    Pose2D,
    quat_wxyz_from_yaw,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
    yaw_from_quat_wxyz,
)


def test_pose2d_is_frozen():
    pose = Pose2D(1.0, 2.0, 0.5)
    with pytest.raises(AttributeError):
        pose.x = 0.0  # type: ignore[misc]


def test_quat_order_roundtrip():
    q_wxyz = (0.1, 0.2, 0.3, 0.4)
    q_xyzw = quat_wxyz_to_xyzw(q_wxyz)
    assert q_xyzw == (0.2, 0.3, 0.4, 0.1)
    assert quat_xyzw_to_wxyz(q_xyzw) == q_wxyz


def test_quat_wxyz_from_yaw_identity_and_quarter_turn():
    assert quat_wxyz_from_yaw(0.0) == pytest.approx((1.0, 0.0, 0.0, 0.0))
    # z 軸まわりに +90°：w = cos(45°)、z = sin(45°)
    s = math.sqrt(0.5)
    assert quat_wxyz_from_yaw(math.pi / 2) == pytest.approx((s, 0.0, 0.0, s))


@pytest.mark.parametrize("yaw", [0.0, 0.3, -1.2, math.pi / 2, 3.0, -3.0])
def test_yaw_from_quat_roundtrip(yaw):
    assert yaw_from_quat_wxyz(quat_wxyz_from_yaw(yaw)) == pytest.approx(yaw)


def test_yaw_from_quat_ignores_scale():
    # 正規化されていないクォータニオンでも向きは同じ
    w, x, y, z = quat_wxyz_from_yaw(0.7)
    assert yaw_from_quat_wxyz((2 * w, 2 * x, 2 * y, 2 * z)) == pytest.approx(0.7)
