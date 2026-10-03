"""Issue #3 アームの検証用の計算（Isaac Sim に依存しない）。

クォータニオンは Isaac Sim と同じ (w, x, y, z) の並びで扱い、関数名に `_wxyz` を付けて明示する。
単位は SI（m, rad）。台車の姿勢は (x, y, yaw)。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import NamedTuple

Vec3 = tuple[float, float, float]
QuatWxyz = tuple[float, float, float, float]
Pose = tuple[Vec3, QuatWxyz]  # (位置, 姿勢)


def quat_from_yaw_wxyz(yaw: float) -> QuatWxyz:
    """z 軸まわりの回転 yaw [rad] のクォータニオン。"""
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def quat_mul_wxyz(a: Sequence[float], b: Sequence[float]) -> QuatWxyz:
    """クォータニオンの積 a ⊗ b（b の回転の後に a の回転）。"""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def rotate_vec_wxyz(q: Sequence[float], v: Sequence[float]) -> Vec3:
    """ベクトル v をクォータニオン q で回転する。"""
    w, x, y, z = q
    _, rx, ry, rz = quat_mul_wxyz(quat_mul_wxyz(q, (0.0, *v)), (w, -x, -y, -z))
    return (rx, ry, rz)


def quat_angle_error_wxyz(q1: Sequence[float], q2: Sequence[float]) -> float:
    """2つの姿勢の差の回転角 [rad]（0〜π）。q と -q は同じ姿勢として扱う。"""
    n1 = math.sqrt(sum(c * c for c in q1))
    n2 = math.sqrt(sum(c * c for c in q2))
    dot = abs(sum(a * b for a, b in zip(q1, q2, strict=True))) / (n1 * n2)
    return 2.0 * math.acos(min(1.0, dot))


def compose_pose(parent: Pose, child: Pose) -> Pose:
    """親の座標系で表した child を、親の外側（ワールドなど）の座標系に変換する。"""
    (pp, pq), (cp, cq) = parent, child
    r = rotate_vec_wxyz(pq, cp)
    return (pp[0] + r[0], pp[1] + r[1], pp[2] + r[2]), quat_mul_wxyz(pq, cq)


def world_from_base(base_pose2d: Sequence[float], point_in_base: Sequence[float]) -> Vec3:
    """台車座標系（前方 x・左方 y・上方 z、原点は台車の床面上）の点をワールド座標に変換する。"""
    x, y, yaw = base_pose2d
    c, s = math.cos(yaw), math.sin(yaw)
    px, py, pz = point_in_base
    return (x + c * px - s * py, y + s * px + c * py, pz)


class ReachPoint(NamedTuple):
    distance: float  # 台車の中心からの水平距離 [m]
    height: float  # 床からの高さ [m]
    angle_deg: float  # 台車の前方からの方向 [deg]（左が正）
    point_in_base: Vec3


def reach_grid(distances: Iterable[float], heights: Iterable[float], angles_deg: Iterable[float]) -> list[ReachPoint]:
    """到達範囲を調べる格子点（台車座標系）。並びは distance → height → angle の順。"""
    heights, angles_deg = list(heights), list(angles_deg)
    grid = []
    for d in distances:
        for h in heights:
            for a in angles_deg:
                r = math.radians(a)
                grid.append(ReachPoint(d, h, a, (d * math.cos(r), d * math.sin(r), h)))
    return grid


def is_success(pos_err: float | None, ang_err: float | None, *, pos_tol: float, ang_tol: float) -> bool:
    """位置誤差 [m] と姿勢誤差 [rad] が許容値以内なら成功。計画に失敗した（None）場合は失敗。"""
    if pos_err is None or ang_err is None:
        return False
    return pos_err <= pos_tol and ang_err <= ang_tol
