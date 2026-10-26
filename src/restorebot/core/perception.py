"""今の写真（天井カメラの RGB＋深度）と VLM の答えから、物体の位置と向きを推定する（設計書 7.3、#23）。

estimate_object：
    1. VLM が答えた「今の位置」の画素のまわりの深度から天板を決め、物体の上面の高さ（天板＋物体の高さ）を求める
    2. その画素の近くで、深度が上面の高さに近い画素を、VLM の点に最も近い画素からつながった塊として切り出す
       （隣に同じ高さの物体があっても混ざらない。散らかし方の隙間 5cm は机の上で約 19px）
    3. 位置：塊の画素を上面の高さで逆投影した点の重心（VLM の点が数 px ずれても補正される）
    4. 向き：箱だけ。塊の長い方向を主成分分析で求める（180° の違いは区別しない）。円柱は推定しない
EstimatedWorld：WorldState の推定版。台車の位置はオドメトリ（台車の関節の値）をそのまま使う。
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from restorebot.core.camera import DownCamera
from restorebot.core.scene import SceneSpec
from restorebot.core.types import ObjectState, Observation, Pose2D, WorldState
from restorebot.core.vlm.parse import PredictedMove, normalize_name


@dataclass(frozen=True)
class Estimate:
    key: str
    position: tuple[float, float, float]  # 物体の中心（ワールド座標）
    yaw: float | None  # 箱の向き（-π/2〜π/2。180° の違いは区別しない）。円柱は None
    surface: str
    n_pixels: int  # 上面の画素の数


def _median_depth(depth: np.ndarray, u: int, v: int, half: int = 2) -> float | None:
    patch = depth[max(v - half, 0) : v + half + 1, max(u - half, 0) : u + half + 1]
    values = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(values)) if values.size else None


def estimate_object(
    key: str,
    px: Sequence[float],
    obs: Observation,
    scene: SceneSpec,
    camera: DownCamera | None = None,
    *,
    height_tol: float = 0.01,
    search_scale: float = 2.5,
    min_pixels: int = 30,
) -> Estimate | None:
    """物体 key の今の位置・向きを、VLM の点 px と今の深度から推定する。上面が見つからなければ None。"""
    cam = camera or scene.camera
    spec = scene.object(key)
    u, v = int(math.floor(px[0])), int(math.floor(px[1]))
    h, w = obs.depth.shape
    if not (0 <= u < w and 0 <= v < h):
        return None
    d = _median_depth(obs.depth, u, v)
    if d is None:
        return None
    x, y, _ = cam.unproject((px[0], px[1]), d)
    surface = scene.surface_at((x, y))
    if surface is None:
        return None
    top_d = cam.position[2] - (surface.z + spec.height)  # 上面の深度（distance_to_image_plane）
    # 探す範囲：物体の外接円の search_scale 倍（画素）
    r = int(math.ceil(search_scale * spec.footprint_radius * cam.fx / top_d))
    i0, i1, j0, j1 = max(v - r, 0), min(v + r + 1, h), max(u - r, 0), min(u + r + 1, w)
    win = obs.depth[i0:i1, j0:j1]
    mask = np.isfinite(win) & (np.abs(win - top_d) < height_tol)
    if not mask.any():
        return None
    # VLM の点に最も近い上面の画素から、つながった塊を広げる（4 近傍）
    ii, jj = np.nonzero(mask)
    k = int(np.argmin((ii + i0 - v) ** 2 + (jj + j0 - u) ** 2))
    seed = (int(ii[k]), int(jj[k]))
    seen = np.zeros_like(mask)
    seen[seed] = True
    queue, pixels = deque([seed]), []
    while queue:
        a, b = queue.popleft()
        pixels.append((a, b))
        for na, nb in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
            if 0 <= na < mask.shape[0] and 0 <= nb < mask.shape[1] and mask[na, nb] and not seen[na, nb]:
                seen[na, nb] = True
                queue.append((na, nb))
    if len(pixels) < min_pixels:
        return None
    p = np.asarray(pixels, dtype=float)
    # 画素の中心を上面の高さで逆投影する
    xs = cam.position[0] + (p[:, 1] + j0 + 0.5 - cam.cx) * top_d / cam.fx
    ys = cam.position[1] - (p[:, 0] + i0 + 0.5 - cam.cy) * top_d / cam.fy
    cx, cy = float(xs.mean()), float(ys.mean())
    yaw = None
    if spec.shape == "cube":
        cov = np.cov(np.vstack([xs - cx, ys - cy]))
        evals, evecs = np.linalg.eigh(cov)
        major = evecs[:, int(np.argmax(evals))]
        yaw = math.atan2(major[1], major[0])
        if spec.size[1] > spec.size[0]:  # 物体の x 方向が短い辺なら 90° 回す
            yaw += math.pi / 2
        yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    return Estimate(key, (cx, cy, surface.z + spec.height / 2), yaw, surface.key, len(pixels))


class EstimatedWorld:
    """WorldState の推定版。

    - VLM が答えた物体：今の写真から推定した位置・向き（update）
    - 置くのに成功した物体：置いた位置（note_placed）
    - それ以外：いつもの位置にあるとみなす
    - 推定に失敗した物体：位置が分からない（surface が None。全体の制御はそのタスクを飛ばす）
    - 台車の位置：オドメトリ（odometry.robot_pose()。台車の関節の値）
    """

    def __init__(self, scene: SceneSpec, odometry: WorldState, camera: DownCamera | None = None) -> None:
        self.scene = scene
        self.odometry = odometry
        self.camera = camera or scene.camera
        self.states: dict[str, ObjectState] = {}
        self.failed: dict[str, str] = {}
        self.estimates: dict[str, Estimate] = {}
        self._by_name = {normalize_name(o.name): o.key for o in scene.objects}

    def update(self, obs: Observation, moves: Sequence[PredictedMove]) -> dict[str, str]:
        """VLM が答えた物体の位置・向きを、今の写真から推定し直す。推定に失敗した物体と理由を返す。"""
        failures = {}
        for m in moves:
            key = self._by_name.get(normalize_name(m.name))
            if key is None:
                continue
            est = estimate_object(key, m.current_px, obs, self.scene, self.camera)
            if est is None:
                failures[key] = f"今の写真で {key} の上面が見つからない（点 {tuple(round(c) for c in m.current_px)}）"
                self.failed[key] = failures[key]
                self.states.pop(key, None)
                continue
            self.failed.pop(key, None)
            self.estimates[key] = est
            self.states[key] = ObjectState(key, est.position, est.yaw or 0.0, est.surface)
        return failures

    def note_placed(self, key: str, position: Sequence[float], yaw: float) -> None:
        """置くのに成功した物体は、置いた位置にあるとみなす。"""
        s = self.scene.surface_at(position[:2])
        self.failed.pop(key, None)
        self.states[key] = ObjectState(key, tuple(position), yaw, None if s is None else s.key)

    def object_state(self, key: str) -> ObjectState:
        if key in self.failed:
            home = self.scene.object_center(key, self.scene.tidy[key])
            return ObjectState(key, home, 0.0, None)
        if key in self.states:
            return self.states[key]
        p = self.scene.tidy[key]
        return ObjectState(key, self.scene.object_center(key, p), p.yaw, p.surface)

    def robot_pose(self) -> Pose2D:
        return self.odometry.robot_pose()
