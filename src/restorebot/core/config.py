"""設定値（設計書 14 章）。型付き・既定値付きの dataclass と、その既定のインスタンス。

物理の設定値（関節の摩擦、指のドライブ）と Isaac Sim のパスは sim/ の定数に置く。場面の定義は core/scene.py。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from restorebot.core.types import Pose2D


@dataclass(frozen=True)
class RobotGeometry:
    """台車の外形（計算用）と、家具に寄せるときの隙間 [m]。"""

    half_length: float = 0.47  # 前後の半分（#2 の実測 0.466m）
    half_width: float = 0.40  # 左右の半分（#10 の深度からの実測 0.39m）
    furniture_gap: float = 0.10  # 台車の前面と家具の縁の隙間（#3：机は台車の前面から 10cm 程度に寄せる）

    @property
    def center_to_edge(self) -> float:
        """家具の正面に止めたときの、台車の中心から家具の縁までの距離。"""
        return self.half_length + self.furniture_gap


@dataclass(frozen=True)
class ReachConfig:
    """アームが楽に届く範囲（台車の中心からの水平距離と、正面からの角度）。#3 の到達範囲の調査から。"""

    min_distance: float = 0.55
    max_distance: float = 0.85
    max_angle: float = math.radians(30.0)


@dataclass(frozen=True)
class ScatterConfig:
    """散らかし方（設計書 5.2）。"""

    min_moved: int = 1  # 動かす物体の数（設計書 18 章 Q10）
    max_moved: int = 3
    # 物体の外接円どうしの隙間（いつもの位置・今の位置の両方。戻し先がふさがらない）。
    # 全開の指の外側は物体の縁から約 3.5cm 出る見込み（開き 8cm、指の厚さ約 2cm と仮定）で、
    # 配置の誤差（約 1cm）を足して 5cm。
    # 10cm では、M1 の止め位置から届く机の範囲にリモコンの動かす先がなかった（#11）
    clearance: float = 0.05
    max_tries: int = 1000  # 1 個の動かす先を探す回数の上限
    max_attempts: int = 50  # 動かす先が見つからないときに、動かす物体を選び直す回数の上限


ROBOT = RobotGeometry()
REACH = ReachConfig()
SCATTER = ScatterConfig()

PLACE_MARGIN = 0.02  # 物体の外形と天板の縁の余白 [m]
# 作業範囲の奥行き：家具の正面に止めた台車から楽に届く、手前の縁からの距離（0.85 − 0.57 = 0.28m）
WORK_DEPTH = REACH.max_distance - ROBOT.center_to_edge

# M1 の止め位置：机（中心 (0.8, 0.85)、奥行き 0.6m）の手前の縁から台車の中心まで 0.57m、机の方（+y）を向く。
# x = 0.75 は、机の 3 個のいつもの位置が楽に届く範囲に入るように選んだ（設計書 5.2）
M1_BASE_POSE = Pose2D(0.75, 0.85 - 0.3 - ROBOT.center_to_edge, math.pi / 2)
