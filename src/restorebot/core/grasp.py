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
from collections.abc import Mapping, Sequence
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
    # 指を開いた後の待ち時間と、退避の前に少しだけ真上に上げる高さ。
    # #13：指を開いてすぐ 15cm を一気に上げると、指にもたれたペン立てが引っかかって倒れた
    release_settle: float = 0.5
    clear_height: float = 0.03
    # 物を持って動く段階の、手先の目標の速さの上限 [m/s]。
    # #13：待機姿勢へ速く振ると、指の腹と線で接するだけの円柱（ペン立て）が手の中で振れて抜けた
    carry_speed: float = 0.3
    position_tolerance: float = 0.015
    orientation_tolerance: float = 0.12
    stability_duration: float = 0.10  # 収束したまま、この時間たったら次へ進む
    gripper_dwell: float = 0.20  # 閉じる・開く段階の最短の時間
    timeouts: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TIMEOUTS))
    # 隣の物体を避ける手先の向き（choose_grasp_yaw）。手を上から見た長方形（指が閉じる方向の半分の長さ・幅の半分）
    hand_half_length: float = 0.11  # Franka の手は指が閉じる方向に約 20cm
    hand_half_width: float = 0.04
    safe_clearance: float = 0.02  # 手と隣の物体の、上から見た隙間がこれ以上あれば当たらないとみなす
    yaw_step: float = math.radians(15.0)  # 円柱で試す手先の向きの刻み（基準から ±90°）


@dataclass(frozen=True)
class Stage:
    name: str
    position: tuple[float, float, float]  # 掴む位置の目標（ワールド座標）
    yaw: float  # 手先の向き
    gripper: Gripper
    wait_gripper: bool  # 指の動きが終わるまで待つ（閉じる・開く段階）
    object_is_obstacle: bool  # この段階で、掴む物体を動作計画の障害物として扱うか
    timeout: float
    min_duration: float = 0.0  # この段階に最低限とどまる時間（閉じる・開く段階は gripper_dwell とのうち長い方）
    max_speed: float | None = None  # 手先の目標を動かす速さの上限 [m/s]（None は目標をそのまま渡す）


def grasp_yaw(shape: str, object_yaw: float, reference_yaw: float) -> float:
    """掴むときの手先の向き（隣の物体を考えない場合）。

    箱は短い辺（物体の y 方向）を掴むよう、手先の向きを物体の向きに合わせる（#4）。180° 回しても同じ掴み方なので、
    基準の向きに近い方を選び、手首を大きく回さない（基準から ±90° 以内）。円柱は向きによらず基準の向き。
    基準の向きは、台車から物体を見た方向（bearing_yaw）を使う（#13：台車の向きでは左右の端で手首が回りきらなかった）。
    """
    if shape == "cylinder":
        return reference_yaw
    if shape == "cube":
        d = wrap_angle(object_yaw - reference_yaw)
        if d > math.pi / 2:
            d -= math.pi
        elif d < -math.pi / 2:
            d += math.pi
        return reference_yaw + d
    raise ValueError(f"上から掴めない形です: {shape}")


def bearing_yaw(base: Pose2D, xy: Sequence[float]) -> float:
    """台車の中心から点 xy を見た方向 [rad]。"""
    return math.atan2(xy[1] - base.y, xy[0] - base.x)


@dataclass(frozen=True)
class Footprint:
    """上から見た物体の外接円（隣の物体として避けるため）。"""

    xy: tuple[float, float]
    radius: float


def footprints(scene, positions: Mapping[str, Sequence[float]], exclude: str) -> list[Footprint]:
    """ほかの物体（exclude 以外）の今の位置 → 隣の物体の外接円の一覧。scene は SceneSpec。"""
    return [
        Footprint((float(p[0]), float(p[1])), scene.object(k).footprint_radius)
        for k, p in positions.items()
        if k != exclude
    ]


def hand_clearance(center: Sequence[float], yaw: float, other: Footprint, params: GraspParams) -> float:
    """手（center の真上、向き yaw）を上から見た長方形と、隣の物体の外接円の隙間 [m]（重なれば負）。"""
    dx, dy = other.xy[0] - center[0], other.xy[1] - center[1]
    fx, fy = math.sin(yaw), -math.cos(yaw)  # 指が閉じる方向（真下向きの手先の y 方向）
    along = abs(dx * fx + dy * fy)
    across = abs(-dx * fy + dy * fx)
    ex = max(along - params.hand_half_length, 0.0)
    ey = max(across - params.hand_half_width, 0.0)
    return math.hypot(ex, ey) - other.radius


def choose_grasp_yaw(
    shape: str,
    axis_yaw: float,
    reference_yaw: float,
    center: Sequence[float],
    neighbors: Sequence[Footprint],
    params: GraspParams,
) -> float:
    """隣の物体を考えた手先の向き。

    箱：向きは物体の軸（axis_yaw。掴むときは物体の向き、置くときは掴んだときの手先の向き）で決まり、
    180° 回しても同じなので、基準に近い方（grasp_yaw）。隣の物体は考えない（選べる向きがない）。
    円柱：基準から ±90° を yaw_step 刻みで試し、手が隣の物体から safe_clearance 以上離れる向きのうち、
    基準に最も近いもの。どれも離れなければ、隙間が最も大きい向き（#13：指の方向に背の高い物体があると下降できなかった）。
    """
    if shape != "cylinder":
        return grasp_yaw(shape, axis_yaw, reference_yaw)
    n = round((math.pi / 2) / params.yaw_step)
    candidates = [reference_yaw + s * k * params.yaw_step for k in range(n + 1) for s in ((1,) if k == 0 else (1, -1))]

    def clearance(yaw: float) -> float:
        return min((hand_clearance(center, yaw, o, params) for o in neighbors), default=math.inf)

    for yaw in candidates:  # 基準に近い順
        if clearance(yaw) >= params.safe_clearance:
            return yaw
    return max(candidates, key=clearance)


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


def move_toward(current: Sequence[float], target: Sequence[float], max_dist: float) -> tuple[float, float, float]:
    """current から target へ、最大 max_dist だけ近づいた点（手先の目標を少しずつ動かして速さを抑えるため）。"""
    d = [t - c for c, t in zip(current, target, strict=True)]
    n = math.sqrt(sum(x * x for x in d))
    if n <= max_dist:
        return tuple(float(t) for t in target)
    s = max_dist / n
    return tuple(float(c + s * x) for c, x in zip(current, d, strict=True))


def _above(p: Sequence[float], dz: float) -> tuple[float, float, float]:
    return (p[0], p[1], p[2] + dz)


def pick_stages(center: Sequence[float], yaw: float, base: Pose2D, params: GraspParams) -> list[Stage]:
    """掴む：真上 → 下降 → 閉じる → 持ち上げる → 待機姿勢。center は物体の中心（＝掴む位置）。

    持ち上げてからは手先の速さを carry_speed に抑える。
    """
    t = params.timeouts
    p = tuple(center)
    v = params.carry_speed
    return [
        Stage("approach", _above(p, params.approach_height), yaw, Gripper.OPEN, False, True, t["approach"]),
        Stage("descend", p, yaw, Gripper.OPEN, False, False, t["descend"]),
        Stage("grasp", p, yaw, Gripper.CLOSE, True, False, t["grasp"]),
        Stage("lift", _above(p, params.lift_height), yaw, Gripper.CLOSE, False, False, t["lift"], max_speed=v),
        Stage(
            "standby", standby_grasp_position(base), base.yaw, Gripper.CLOSE, False, False, t["standby"], max_speed=v
        ),
    ]


def place_stages(center: Sequence[float], yaw: float, base: Pose2D, params: GraspParams) -> list[Stage]:
    """置く：真上 → 下降 → 開く（落ち着くまで待つ）→ 少し上げる → 退避 → 待機姿勢。

    center は置いたときの物体の中心。手先は yaw のまま回さない。
    """
    t = params.timeouts
    p = _above(center, params.place_clearance)
    v = params.carry_speed  # 指を開くまでは物を持っているので、手先の速さを抑える
    return [
        Stage(
            "approach",
            _above(p, params.approach_height),
            yaw,
            Gripper.CLOSE,
            False,
            False,
            t["place_approach"],
            max_speed=v,
        ),
        Stage("descend", p, yaw, Gripper.CLOSE, False, False, t["place_descend"], max_speed=v),
        Stage("release", p, yaw, Gripper.OPEN, True, False, t["release"], min_duration=params.release_settle),
        Stage("clear", _above(p, params.clear_height), yaw, Gripper.OPEN, False, False, t["retreat"]),
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
        min_duration = max(stage.min_duration, p.gripper_dwell if stage.wait_gripper else 0.0)
        complete = complete and t - self._stage_started >= min_duration
        if complete and stage.wait_gripper:
            complete = gripper_done
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
