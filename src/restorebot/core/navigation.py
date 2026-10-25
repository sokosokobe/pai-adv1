"""台車の位置に関する計算（設計書 8 章）。

今は、アームが楽に届く範囲の判定と、家具の正面で家具の方を向いた姿勢だけ。止め位置・経路計画・経路追従は #16 で足す。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from restorebot.core.config import REACH, ROBOT, ReachConfig, RobotGeometry
from restorebot.core.types import Pose2D

if TYPE_CHECKING:
    from restorebot.core.scene import Surface


def wrap_angle(a: float) -> float:
    """角度を -π〜π に収める。"""
    return math.atan2(math.sin(a), math.cos(a))


def in_reach(base: Pose2D, xy: Sequence[float], reach: ReachConfig = REACH) -> bool:
    """点 xy（ワールド座標の水平位置）が、台車 base から楽に届く範囲（距離と正面からの角度）にあるか。"""
    dx, dy = xy[0] - base.x, xy[1] - base.y
    distance = math.hypot(dx, dy)
    if not reach.min_distance <= distance <= reach.max_distance:
        return False
    return abs(wrap_angle(math.atan2(dy, dx) - base.yaw)) <= reach.max_angle


def facing_pose(surface: Surface, side: str, lateral: float, robot: RobotGeometry = ROBOT) -> Pose2D:
    """家具 surface の side 側（"-x"・"+x"・"-y"・"+y"）に、家具の方を向いて止まった台車の姿勢。

    台車の前面と家具の縁の隙間は robot.furniture_gap。
    lateral は家具の辺に沿った方向の座標（-x・+x 側なら y、-y・+y 側なら x）。
    """
    d = robot.center_to_edge
    cx, cy = surface.center
    hw, hh = surface.size[0] / 2, surface.size[1] / 2
    if side == "-x":
        return Pose2D(cx - (hw + d), lateral, 0.0)
    if side == "+x":
        return Pose2D(cx + (hw + d), lateral, math.pi)
    if side == "-y":
        return Pose2D(lateral, cy - (hh + d), math.pi / 2)
    if side == "+y":
        return Pose2D(lateral, cy + (hh + d), -math.pi / 2)
    raise ValueError(f"side は -x・+x・-y・+y のどれか: {side}")
