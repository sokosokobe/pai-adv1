"""Issue #4 把持と配置の検証用の判定・条件生成（Isaac Sim に依存しない）。

単位は SI（m, rad）。表示用の角度だけ deg。クォータニオンは (w, x, y, z)。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def sample_initial_offsets(
    n: int, *, seed: int, max_offset: float = 0.03, max_yaw_deg: float = 30.0
) -> list[tuple[float, float, float]]:
    """物体の初期位置のずらし方 (dx [m], dy [m], dyaw [rad]) を n 個作る。

    1 個目は基準位置（ずらさない）。残りは一様乱数で、シードを固定すれば毎回同じになる。
    """
    rng = np.random.default_rng(seed)
    offsets = [(0.0, 0.0, 0.0)]
    for _ in range(n - 1):
        dx, dy = rng.uniform(-max_offset, max_offset, size=2)
        dyaw = rng.uniform(-math.radians(max_yaw_deg), math.radians(max_yaw_deg))
        offsets.append((float(dx), float(dy), float(dyaw)))
    return offsets


def tilt_deg_from_quat_wxyz(q_wxyz: Sequence[float]) -> float:
    """物体の z 軸と鉛直上向きのなす角 [deg]（向き（yaw）には依存しない）。"""
    w, x, y, z = q_wxyz
    n2 = w * w + x * x + y * y + z * z
    # 回転行列の (2, 2) 成分 = 物体の z 軸のワールド z 成分
    zz = 1.0 - 2.0 * (x * x + y * y) / n2
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


@dataclass(frozen=True)
class TrialObservation:
    """1 回の把持・配置の試行で観測したこと。"""

    controller_done: bool  # PickPlaceController が DONE まで進んだ
    moved_before_grasp_m: float  # 指を閉じ始める前に物体が動いた距離（手が当たって押した）
    max_lift_m: float  # 初期の高さからの物体の最大の上昇量
    dropped: bool  # 一度持ち上がった後、配置の前に落ちた
    place_xy_err_m: float  # 最終位置と配置目標の水平距離
    tilt_deg: float  # 最終の傾き


def classify_trial(
    o: TrialObservation,
    *,
    place_tol: float = 0.02,
    tilt_tol_deg: float = 10.0,
    lift_min: float = 0.03,
    pre_grasp_move_tol: float = 0.01,
) -> str:
    """試行結果を分類する。原因に近いものから順に判定する。

    Returns:
        "success"、"collision_before_grasp"（掴む前に物体に当たった）、"grasp_miss"（持ち上がらない）、
        "dropped"（運ぶ途中で落ちた）、"tipped"（倒れた）、"placement_off"（置き場所がずれた）、
        "controller_failure"（物体は問題ないがコントローラが失敗・時間切れ）
    """
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
