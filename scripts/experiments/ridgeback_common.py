"""Issue #2 台車の検証用の計算（Isaac Sim に依存しない）。

座標・単位の約束（CLAUDE.md）：SI 単位、ワールドは Z 軸が上向き、台車の姿勢は (x, y, yaw)。
クォータニオンは Isaac Sim の並び (w, x, y, z) を受け取り、関数名で明示する。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

Pose2D = tuple[float, float, float]  # (x [m], y [m], yaw [rad])


def wrap_angle(angle: float) -> float:
    """角度を (-π, π] に正規化する。"""
    wrapped = math.remainder(angle, 2 * math.pi)  # [-π, π]
    return math.pi if wrapped <= -math.pi else wrapped


def yaw_from_quat_wxyz(q_wxyz: Sequence[float]) -> float:
    """クォータニオン (w, x, y, z) から z 軸まわりの回転角 yaw を求める。"""
    w, x, y, z = q_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def body_to_world_velocity(vx_body: float, vy_body: float, yaw: float) -> tuple[float, float]:
    """台車基準の並進速度（前方 x、左方 y）をワールド基準に変換する。"""
    c, s = math.cos(yaw), math.sin(yaw)
    return c * vx_body - s * vy_body, s * vx_body + c * vy_body


class PoseError(NamedTuple):
    dx: float  # 目標 - 現在（ワールド基準）[m]
    dy: float
    dyaw: float  # (-π, π] に正規化 [rad]
    distance: float  # 位置の誤差の大きさ [m]


def pose_error(current: Pose2D, target: Pose2D) -> PoseError:
    """目標姿勢と現在姿勢の差（ワールド基準）。"""
    dx, dy = target[0] - current[0], target[1] - current[1]
    return PoseError(dx, dy, wrap_angle(target[2] - current[2]), math.hypot(dx, dy))


def velocity_command(
    current: Pose2D,
    target: Pose2D,
    *,
    kp_lin: float,
    kp_ang: float,
    v_max: float,
    w_max: float,
) -> tuple[float, float, float]:
    """目標姿勢へ向かう速度指令（ワールド基準の vx, vy と角速度 wz）を比例制御で作る。

    並進速度は大きさを v_max で飽和させ、向きは目標方向のまま保つ。角速度は ±w_max で飽和させる。
    """
    err = pose_error(current, target)
    vx, vy = kp_lin * err.dx, kp_lin * err.dy
    speed = math.hypot(vx, vy)
    if speed > v_max:
        vx, vy = vx * v_max / speed, vy * v_max / speed
    wz = max(-w_max, min(w_max, kp_ang * err.dyaw))
    return vx, vy, wz


def settle_time(
    times: Sequence[float],
    pos_errors: Sequence[float],
    yaw_errors: Sequence[float],
    *,
    pos_tol: float,
    yaw_tol: float,
) -> float | None:
    """位置・角度の誤差が許容値に入り、その後ずっと入ったままになった最初の時刻。

    最後の時刻で許容値の外にある場合は None（未到達）。
    """
    settled_at = None
    for t, pe, ye in zip(times, pos_errors, yaw_errors, strict=True):
        inside = abs(pe) <= pos_tol and abs(ye) <= yaw_tol
        if inside and settled_at is None:
            settled_at = t
        elif not inside:
            settled_at = None
    return settled_at


def max_deviation(points: np.ndarray, reference: np.ndarray) -> float:
    """点列 (N, 3) の、基準点からの距離の最大値。"""
    return float(np.max(np.linalg.norm(np.asarray(points) - np.asarray(reference), axis=1)))


def step_toward(current: Pose2D, goal: Pose2D, max_lin_step: float, max_ang_step: float) -> Pose2D:
    """current から goal へ、1ステップあたり並進 max_lin_step [m]・回転 max_ang_step [rad] だけ近づける。

    位置指令を一定速度で動かす（ランプ）ために使う。yaw は関節角として使うので ±π で折り返さず、
    最短方向に連続的に動かす。
    """
    dx, dy = goal[0] - current[0], goal[1] - current[1]
    dist = math.hypot(dx, dy)
    if dist > max_lin_step:
        dx, dy = dx * max_lin_step / dist, dy * max_lin_step / dist
    dyaw = wrap_angle(goal[2] - current[2])
    dyaw = max(-max_ang_step, min(max_ang_step, dyaw))
    return current[0] + dx, current[1] + dy, current[2] + dyaw


def classify_obstacle_outcome(
    *, box_moved: float, base_progress: float, moved_tol: float = 0.05, reach_ratio: float = 0.9
) -> str:
    """障害物に向かって移動させた結果を分類する。

    Args:
        box_moved: 箱が動いた距離 [m]
        base_progress: 台車が進んだ割合（実際の移動距離 / 指令した移動距離）
    Returns:
        "push"（箱を押した）、"pass_through"（箱は動かず台車は目標付近まで進んだ）、"stop"（止まった）
    """
    if box_moved > moved_tol:
        return "push"
    if base_progress >= reach_ratio:
        return "pass_through"
    return "stop"
