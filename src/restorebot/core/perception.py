"""今の写真（天井カメラの RGB＋深度）と VLM の答えから、物体の位置と向きを推定する（設計書 7.3、#23）。

estimate_object：
    1. VLM が答えた「今の位置」の画素のまわりの深度から天板を決め、物体の上面の高さ（天板＋物体の高さ）を求める
    2. その画素の近くで、深度が上面の高さに近い画素を、VLM の点に最も近い画素からつながった塊として切り出す
       （隣に同じ高さの物体があっても混ざらない。散らかし方の隙間 5cm は机の上で約 19px）
    3. 位置：塊の画素を上面の高さで逆投影した点の重心（VLM の点が数 px ずれても補正される）
    4. 向き：箱だけ。塊の長い方向を主成分分析で求める（180° の違いは区別しない）。円柱は推定しない
estimate_robot_pose（#26）：今の写真から台車の位置と向きを推定する（オドメトリとの差を確かめるため）
    1. いつもの写真（ロボットなし）より手前にある画素のうち、台車の上面の高さにある画素の、いちばん大きな塊
       （アームの像が上面を横切って分かれた塊も、台車の大きさの範囲にあればまとめる。離れた雑音は除く）
    2. 塊の凸包に、面積が最小の長方形を当てはめる（回転キャリパー法）。撮影姿勢のアームは上面の内側を隠すだけなので、
       凸包は台車の外形のまま（重心と主成分分析では、アームに隠された分だけ位置が 3cm・向きが 40° ずれた）
    3. 長方形の中心が位置、長い辺の方向が向き。前後は対称なので、180° の違いはオドメトリの向きに近い方
    4. 長方形の大きさが台車の上面と合わなければ（家具に隠れた・写真の外にはみ出た）推定できないとする
EstimatedWorld：WorldState の推定版。台車の位置はオドメトリ（台車の関節の値）をそのまま使う。
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from restorebot.core.camera import DownCamera
from restorebot.core.config import RobotGeometry
from restorebot.core.navigation import wrap_angle
from restorebot.core.occlusion import robot_mask
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


@dataclass(frozen=True)
class RobotPoseEstimate:
    pose: Pose2D | None  # 推定できなければ None
    size: tuple[float, float] | None  # 当てはめた長方形の (長い辺, 短い辺) [m]
    n_pixels: int  # 上面の塊の画素の数
    reason: str | None = None  # 推定できなかった理由


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """mask の塊（8 近傍でつながった画素）。それぞれ (行, 列) の配列。大きい順。"""
    seen = np.zeros_like(mask)
    h, w = mask.shape
    blobs = []
    for seed in zip(*np.nonzero(mask), strict=True):
        if seen[seed]:
            continue
        seen[seed] = True
        queue, pixels = deque([seed]), []
        while queue:
            a, b = queue.popleft()
            pixels.append((a, b))
            for na in (a - 1, a, a + 1):
                for nb in (b - 1, b, b + 1):
                    if 0 <= na < h and 0 <= nb < w and mask[na, nb] and not seen[na, nb]:
                        seen[na, nb] = True
                        queue.append((na, nb))
        blobs.append(np.asarray(pixels))
    return sorted(blobs, key=len, reverse=True)


def _robot_blob(mask: np.ndarray, radius_px: float, min_part: int = 50) -> np.ndarray:
    """台車の上面の画素：いちばん大きな塊と、その重心から radius_px 以内に重心がある塊（min_part 画素以上）。

    アームの像が上面を横切ると、上面が 2 つの塊に分かれることがある（視差で、写真の中心から離れるほど像がずれる）。
    """
    blobs = _components(mask)
    out = np.zeros_like(mask)
    if not blobs:
        return out
    center = blobs[0].mean(axis=0)
    for blob in blobs:
        if blob is blobs[0] or (len(blob) >= min_part and np.hypot(*(blob.mean(axis=0) - center)) <= radius_px):
            out[blob[:, 0], blob[:, 1]] = True
    return out


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """2 次元の点の凸包の頂点（反時計回り。Andrew の単調連鎖法）。"""
    pts = sorted({(float(x), float(y)) for x, y in points})
    if len(pts) < 3:
        return np.asarray(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    upper: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def min_area_rect(points: np.ndarray) -> tuple[tuple[float, float], float, tuple[float, float]]:
    """点を囲む面積が最小の長方形：(中心, 長い辺の向き [rad]（-π/2〜π/2）, (長い辺, 短い辺))。"""
    hull = _convex_hull(points)
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        a = math.atan2(e[1], e[0])
        c, s = math.cos(a), math.sin(a)
        q = hull @ np.array([[c, -s], [s, c]])  # 辺の方向を軸にした座標
        lo, hi = q.min(axis=0), q.max(axis=0)
        area = float(np.prod(hi - lo))
        if best is None or area < best[0]:
            best = (area, a, (lo + hi) / 2, hi - lo)
    _, a, mid, (sa, sb) = best
    c, s = math.cos(a), math.sin(a)
    center = (float(mid[0] * c - mid[1] * s), float(mid[0] * s + mid[1] * c))
    if sb > sa:  # 長い辺の向きにそろえる
        a, sa, sb = a + math.pi / 2, sb, sa
    a = (a + math.pi / 2) % math.pi - math.pi / 2
    return center, a, (float(sa), float(sb))


def estimate_robot_pose(
    obs: Observation,
    reference_depth: np.ndarray,
    odometry: Pose2D,
    camera: DownCamera,
    geometry: RobotGeometry | None = None,
    *,
    size_tol: float = 0.05,
    min_pixels: int = 2000,
) -> RobotPoseEstimate:
    """今の写真から台車の位置と向きを推定する。180° の違いはオドメトリ odometry の向きに近い方を選ぶ。"""
    g = geometry or RobotGeometry()
    z = camera.position[2] - obs.depth
    mask = robot_mask(obs.depth, reference_depth) & (z >= g.top_z_min) & (z <= g.top_z_max)
    if not mask.any():
        return RobotPoseEstimate(None, None, 0, "台車の上面の高さの画素がない")
    # 塊をまとめる範囲：台車の外接円の半径（上面の高さで画素に直す）
    radius_px = g.circumradius * camera.fx / (camera.position[2] - (g.top_z_min + g.top_z_max) / 2)
    blob = _robot_blob(mask, radius_px)
    n = int(blob.sum())
    if n < min_pixels:
        return RobotPoseEstimate(None, None, n, f"台車の上面の画素が少ない（{n} 画素）")
    # 塊の外形：行ごと・列ごとの端の画素の 4 隅（画素の中心は +0.5。端の画素の外側の縁まで含める）
    rows, cols = np.nonzero(blob)
    edge = []
    for idx, other in ((rows, cols), (cols, rows)):
        order = np.lexsort((other, idx))
        a, b = idx[order], other[order]
        first = np.r_[True, a[1:] != a[:-1]]
        last = np.r_[a[1:] != a[:-1], True]
        sel = first | last
        pairs = np.stack([b[sel], a[sel]], axis=1) if idx is rows else np.stack([a[sel], b[sel]], axis=1)
        edge.append(pairs)  # (列, 行)
    px = np.unique(np.concatenate(edge), axis=0).astype(float)
    corners = np.concatenate([px + np.array([du, dv]) for du in (0.0, 1.0) for dv in (0.0, 1.0)])
    # 上面は水平なので、画素の座標から上面の高さの平面へ逆投影する（深度は塊の中央値）
    d = float(np.median(obs.depth[blob]))
    xy = np.stack(
        [
            camera.position[0] + (corners[:, 0] - camera.cx) * d / camera.fx,
            camera.position[1] - (corners[:, 1] - camera.cy) * d / camera.fy,
        ],
        axis=1,
    )
    (cx, cy), axis, size = min_area_rect(xy)
    if abs(size[0] - g.top_length) > size_tol or abs(size[1] - g.top_width) > size_tol:
        reason = f"長方形の大きさ {size[0]:.2f} × {size[1]:.2f}m が台車の上面と合わない（隠れている・写真の外）"
        return RobotPoseEstimate(None, size, n, reason)
    yaw = axis if abs(wrap_angle(axis - odometry.yaw)) <= math.pi / 2 else wrap_angle(axis + math.pi)
    return RobotPoseEstimate(Pose2D(cx, cy, yaw), size, n)


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
