"""モジュール間でやり取りする型と、sim が実装する Protocol（設計書 3.3）。

単位は SI（m, rad, s）。ワールド座標は Z 軸が上向き。
クォータニオンは Isaac Sim の並び (w, x, y, z) を使い、変数名に `_wxyz` を付ける。
SciPy などの (x, y, z, w) との変換は、このモジュールの関数だけで行う（設計書 4 章）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Pose2D:
    """台車の姿勢（ワールド座標）。x, y [m]、yaw [rad]。台車の仮想関節の値と同じ。"""

    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class ObjectState:
    """物体の状態（ワールド座標）。中心の位置 [m] と z 軸まわりの向き yaw [rad]。"""

    key: str  # "cup" など（SceneSpec の ObjectSpec.key）
    position: tuple[float, float, float]
    yaw: float
    surface: str | None  # 置かれている面（"desk" など）。床・不明なら None


@dataclass(frozen=True)
class Observation:
    """天井カメラの 1 枚。rgb: (H, W, 3) uint8、depth: (H, W) float32 [m]（distance_to_image_plane）。"""

    rgb: np.ndarray
    depth: np.ndarray
    time_s: float


@dataclass(frozen=True)
class MoveInstruction:
    """VLM の応答を、物体と 3 次元の目標に直したもの（設計書 6.3）。"""

    key: str
    current_px: tuple[float, float]
    target_px: tuple[float, float]
    target_position: tuple[float, float, float]  # 戻し先（物体の中心）[m]
    target_surface: str


@dataclass(frozen=True)
class TidyTask:
    """1 個の物体を「今の位置で掴み、戻し先に置く」こと。"""

    key: str
    pick: ObjectState  # 掴む物体（M1〜M2 は真値）
    place_position: tuple[float, float, float]
    # 置いたときの物体の向き [rad]。物体の向きは戻さないので、掴んだときの物体の向きのまま（設計書 18 章 Q3）。
    # 手先の向きは、物体の形に合わせて sim の掴む処理が決める
    place_yaw: float
    place_surface: str


class Status(Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class StepResult:
    status: Status
    reason: str | None = None  # 失敗の理由（記録用）


# ---- sim が実装する Protocol ----
class WorldState(Protocol):
    """物体・ロボットの位置。M1〜M2 は真値、推定（余裕があれば）は同じインターフェースで差し替える。"""

    def object_state(self, key: str) -> ObjectState: ...
    def robot_pose(self) -> Pose2D: ...


class CeilingCamera(Protocol):
    def capture(self) -> Observation: ...  # 描画が落ち着くまで進めてから撮る（設計書 5.4）


class Action(Protocol):
    """時間のかかる動作（移動・掴む・置く・姿勢の変更）。全体の制御が物理の 1 ステップごとに step() を呼ぶ。"""

    name: str

    def start(self) -> None: ...
    def step(self, dt: float) -> StepResult: ...


class Robot(Protocol):
    def navigate(self, path: list[Pose2D]) -> Action: ...  # 経路に沿って台車を動かす（設計書 8.3）
    def pick(self, obj: ObjectState) -> Action: ...  # 設計書 9.2
    def place(self, position: tuple[float, float, float], yaw: float) -> Action: ...
    def arm_to(self, posture: str) -> Action: ...  # "standby" | "folded"（設計書 9.3）
    def holding(self) -> str | None: ...  # 掴んでいる物体（真値で判定）


class Simulation(Protocol):
    def step(self) -> float: ...  # 物理を 1 ステップ進め、経過時間 [s] を返す


# ---- クォータニオンの並びの変換（ライブラリの境界で使う）----
def quat_wxyz_to_xyzw(q_wxyz: Sequence[float]) -> tuple[float, float, float, float]:
    """Isaac Sim の (w, x, y, z) → SciPy などの (x, y, z, w)。"""
    w, x, y, z = q_wxyz
    return (x, y, z, w)


def quat_xyzw_to_wxyz(q_xyzw: Sequence[float]) -> tuple[float, float, float, float]:
    """SciPy などの (x, y, z, w) → Isaac Sim の (w, x, y, z)。"""
    x, y, z, w = q_xyzw
    return (w, x, y, z)


def quat_wxyz_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    """z 軸まわりに yaw [rad] だけ回す回転 (w, x, y, z)。"""
    return (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))


def yaw_from_quat_wxyz(q_wxyz: Sequence[float]) -> float:
    """回転 (w, x, y, z) の z 軸まわりの角度 [rad]（-π〜π）。正規化されていなくてもよい。"""
    w, x, y, z = q_wxyz
    # atan2 の 2 つの引数がどちらもノルムの 2 乗に比例するので、正規化しなくても角度は同じ
    return math.atan2(2 * (w * z + x * y), w * w + x * x - y * y - z * z)
