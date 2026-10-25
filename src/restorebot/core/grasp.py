# SPDX-License-Identifier: Apache-2.0
#
# 段階を進める判定（StageSequencer）と既定の時間切れは、Isaac Sim 6.1 の Pick and Place サンプルの
# PickPlaceController（isaacsim.robot_motion.examples/manipulation/controllers/pick_place.py、
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES、Apache License 2.0）を移植し、
# 「掴む」と「置く」に分けられるように書き直したもの。
"""掴む・置くの段階の列と、段階を進める判定（設計書 9.2）。Isaac Sim に依存しない。

位置は「掴む位置」（指先の間の中心。panda_hand から手先の z 方向に GRASP_OFFSET_Z だけ先）で表す。
手先は真下を向き、yaw はその真下向きの手先を z 軸まわりに回す角度 [rad]。指は手先の y 方向に閉じる。
sim はこの段階の目標を、panda_hand の目標に直して RMPflow に渡す。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from restorebot.core.navigation import wrap_angle
from restorebot.core.types import Pose2D, Status

GRASP_OFFSET_Z = 0.1034  # panda_hand → 指先の間（サンプルの Franka 設定と同じ）
STANDBY_HAND_IN_BASE = (0.4, 0.0, 0.95)  # 待機姿勢の手先（panda_hand）の位置（台車座標。#3）


class Gripper(Enum):
    OPEN = "open"
    CLOSE = "close"


# PickPlaceController の既定の時間切れ [s]。待機姿勢へ戻る段階はサンプルにないので、目標の真上へ運ぶ段階と同じにする
DEFAULT_TIMEOUTS = {
    "approach": 5.0,
    "descend": 5.0,
    "grasp": 2.0,
    "lift": 5.0,
    "place_approach": 7.0,
    "place_descend": 5.0,
    "release": 2.0,
    "retreat": 5.0,
    "standby": 7.0,
}


@dataclass(frozen=True)
class GraspParams:
    """掴む・置くのパラメータ。高さは #4 の値（サンプルの既定 0.20 / 0.30 では退避位置に届かなかった）。"""

    approach_height: float = 0.15  # 真上へ近づく高さ
    lift_height: float = 0.15  # 持ち上げる・退避する高さ
    place_clearance: float = 0.005  # 置く高さに足す隙間（設計書 9.2）
    position_tolerance: float = 0.015
    orientation_tolerance: float = 0.12
    stability_duration: float = 0.10  # 収束したまま、この時間たったら次へ進む
    gripper_dwell: float = 0.20  # 閉じる・開く段階の最短の時間
    timeouts: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TIMEOUTS))


@dataclass(frozen=True)
class Stage:
    name: str
    position: tuple[float, float, float]  # 掴む位置の目標（ワールド座標）
    yaw: float  # 手先の向き
    gripper: Gripper
    wait_gripper: bool  # 指の動きが終わるまで待つ（閉じる・開く段階）
    object_is_obstacle: bool  # この段階で、掴む物体を動作計画の障害物として扱うか
    timeout: float


def grasp_yaw(shape: str, object_yaw: float, base_yaw: float) -> float:
    """掴むときの手先の向き。

    箱は短い辺（物体の y 方向）を掴むよう、手先の向きを物体の向きに合わせる（#4）。180° 回しても同じ掴み方なので、
    台車の向きに近い方を選び、手首を大きく回さない（台車の向きから ±90° 以内）。円柱は向きによらず台車の向き。
    """
    if shape == "cylinder":
        return base_yaw
    if shape == "cube":
        d = wrap_angle(object_yaw - base_yaw)
        if d > math.pi / 2:
            d -= math.pi
        elif d < -math.pi / 2:
            d += math.pi
        return base_yaw + d
    raise ValueError(f"上から掴めない形です: {shape}")


def hand_down_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    """真下向きの手先（x 軸まわりに 180°）を、さらに z 軸まわりに yaw だけ回した姿勢 (w, x, y, z)。

    quat_wxyz_from_yaw(yaw) ⊗ (0, 1, 0, 0) を展開したもの。yaw = 0 で Pick and Place サンプルの (0, 1, 0, 0)。
    """
    return (0.0, math.cos(yaw / 2), math.sin(yaw / 2), 0.0)


def quat_angle_wxyz(q1: Sequence[float], q2: Sequence[float]) -> float:
    """2 つの姿勢の差の回転角 [rad]（0〜π）。q と -q は同じ姿勢として扱う。"""
    n1 = math.sqrt(sum(c * c for c in q1))
    n2 = math.sqrt(sum(c * c for c in q2))
    dot = abs(sum(a * b for a, b in zip(q1, q2, strict=True))) / (n1 * n2)
    return 2.0 * math.acos(min(1.0, dot))


def standby_grasp_position(base: Pose2D) -> tuple[float, float, float]:
    """待機姿勢（台車座標で手先 (0.4, 0, 0.95)、真下向き）の掴む位置（手先の GRASP_OFFSET_Z 下）。"""
    hx, hy, hz = STANDBY_HAND_IN_BASE
    c, s = math.cos(base.yaw), math.sin(base.yaw)
    return (base.x + c * hx - s * hy, base.y + s * hx + c * hy, hz - GRASP_OFFSET_Z)


def _above(p: Sequence[float], dz: float) -> tuple[float, float, float]:
    return (p[0], p[1], p[2] + dz)


def pick_stages(center: Sequence[float], yaw: float, base: Pose2D, params: GraspParams) -> list[Stage]:
    """掴む：真上 → 下降 → 閉じる → 持ち上げる → 待機姿勢。center は物体の中心（＝掴む位置）。"""
    t = params.timeouts
    p = tuple(center)
    return [
        Stage("approach", _above(p, params.approach_height), yaw, Gripper.OPEN, False, True, t["approach"]),
        Stage("descend", p, yaw, Gripper.OPEN, False, False, t["descend"]),
        Stage("grasp", p, yaw, Gripper.CLOSE, True, False, t["grasp"]),
        Stage("lift", _above(p, params.lift_height), yaw, Gripper.CLOSE, False, False, t["lift"]),
        Stage("standby", standby_grasp_position(base), base.yaw, Gripper.CLOSE, False, False, t["standby"]),
    ]


def place_stages(center: Sequence[float], yaw: float, base: Pose2D, params: GraspParams) -> list[Stage]:
    """置く：真上 → 下降 → 開く → 退避 → 待機姿勢。center は置いたときの物体の中心。手先は回さない（yaw のまま）。"""
    t = params.timeouts
    p = _above(center, params.place_clearance)
    return [
        Stage("approach", _above(p, params.approach_height), yaw, Gripper.CLOSE, False, False, t["place_approach"]),
        Stage("descend", p, yaw, Gripper.CLOSE, False, False, t["place_descend"]),
        Stage("release", p, yaw, Gripper.OPEN, True, False, t["release"]),
        Stage("retreat", _above(p, params.lift_height), yaw, Gripper.OPEN, False, False, t["retreat"]),
        Stage("standby", standby_grasp_position(base), base.yaw, Gripper.OPEN, False, True, t["standby"]),
    ]


class StageSequencer:
    """段階を順に進める。手先が目標に収束して stability_duration たったら次へ（閉じる・開く段階は指の動きも待つ）。

    PickPlaceController の _phase_complete・時間切れと同じ判定。
    """

    def __init__(self, stages: Sequence[Stage], params: GraspParams) -> None:
        if not stages:
            raise ValueError("段階が空です")
        self.stages = list(stages)
        self.params = params
        self.index = 0
        self.failure_reason: str | None = None
        self._stage_started: float | None = None
        self._stable_since: float | None = None
        self._status = Status.RUNNING

    @property
    def stage(self) -> Stage:
        return self.stages[min(self.index, len(self.stages) - 1)]

    @property
    def done(self) -> bool:
        return self._status is Status.SUCCEEDED

    def start(self, t: float) -> None:
        self.index = 0
        self.failure_reason = None
        self._stage_started = t
        self._stable_since = None
        self._status = Status.RUNNING

    def update(self, t: float, position_error: float, orientation_error: float, gripper_done: bool) -> Status:
        """今の手先の誤差（掴む位置 [m]、向き [rad]）と指の状態から、段階を進める。"""
        if self._stage_started is None:
            raise RuntimeError("start() を先に呼んでください")
        if self._status is not Status.RUNNING:
            return self._status
        stage = self.stage
        p = self.params
        converged = position_error <= p.position_tolerance and orientation_error <= p.orientation_tolerance
        if not converged:
            self._stable_since = None
        elif self._stable_since is None:
            self._stable_since = t
        complete = converged and t - self._stable_since >= p.stability_duration
        if complete and stage.wait_gripper:
            complete = t - self._stage_started >= p.gripper_dwell and gripper_done
        if complete:
            self.index += 1
            self._stage_started = t
            self._stable_since = None
            if self.index == len(self.stages):
                self._status = Status.SUCCEEDED
        elif t - self._stage_started >= stage.timeout:
            self.failure_reason = (
                f"{stage.name} が {stage.timeout:.1f} 秒で時間切れ"
                f"（位置の誤差 {position_error:.4f}m、向きの誤差 {orientation_error:.4f}rad）"
            )
            self._status = Status.FAILED
        return self._status


# ------------------------------------------------------------
# 成功の判定（#4 の scripts/experiments/grasp_common.py の classify_trial）
# ------------------------------------------------------------
@dataclass(frozen=True)
class TrialObservation:
    """1 回の掴む・置くで観測したこと。"""

    controller_done: bool  # 段階を最後まで進めた
    moved_before_grasp_m: float  # 指を閉じ始める前に物体が動いた距離（手が当たって押した）
    max_lift_m: float  # 初期の高さからの物体の最大の上昇量
    dropped: bool  # 一度持ち上がった後、置く前に落ちた
    place_xy_err_m: float  # 最終位置と置く位置の水平距離
    tilt_deg: float  # 最終の傾き


def classify_trial(
    o: TrialObservation,
    *,
    place_tol: float = 0.03,
    tilt_tol_deg: float = 10.0,
    lift_min: float = 0.03,
    pre_grasp_move_tol: float = 0.01,
) -> str:
    """結果を分類する（原因に近いものから順に判定）。place_tol は設計書 13 章の成功の基準（3cm）。"""
    if o.moved_before_grasp_m > pre_grasp_move_tol:
        return "collision_before_grasp"
    if o.max_lift_m < lift_min:
        return "grasp_miss"
    if o.dropped:
        return "dropped"
    if o.tilt_deg > tilt_tol_deg:
        return "tipped"
    if o.place_xy_err_m > place_tol:
        return "placement_off"
    if not o.controller_done:
        return "controller_failure"
    return "success"


def tilt_deg_from_quat_wxyz(q_wxyz: Sequence[float]) -> float:
    """物体の z 軸と鉛直上向きのなす角 [deg]（yaw には依存しない）。"""
    w, x, y, z = q_wxyz
    n2 = w * w + x * x + y * y + z * z
    zz = 1.0 - 2.0 * (x * x + y * y) / n2  # 回転行列の (2, 2) 成分 = 物体の z 軸のワールド z 成分
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))
