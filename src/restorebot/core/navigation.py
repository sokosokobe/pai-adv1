"""台車の位置と移動の計算（設計書 8 章）。Isaac Sim に依存しない。

- 楽に届く範囲、家具の正面の姿勢、台車の外形と家具・壁との距離
- 止め位置（approach_pose）：物体の正面で、家具の方を向き、隙間 10cm
- 経路計画（plan_route）：格子地図＋A*（障害物を外接円の半径＋余白だけ膨らませる）。止め位置は家具に近く
  膨張の内側にあるので、出発・到着の前後に「向きを変えずにまっすぐ下がる／寄せる」区間を付ける
- 経路追従（PathFollower）：経由点に向かう比例制御＋飽和の速度指令（ワールド座標）
- 撮影時の後退（retreat_pose）
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from restorebot.core.config import NAV, REACH, ROBOT, NavConfig, ReachConfig, RobotGeometry
from restorebot.core.types import Pose2D

if TYPE_CHECKING:
    from restorebot.core.scene import SceneSpec, Surface

SIDES = ("-x", "+x", "-y", "+y")
Point = tuple[float, float]


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


# ------------------------------------------------------------
# 台車の外形と、家具・壁との距離
# ------------------------------------------------------------
def robot_corners(pose: Pose2D, robot: RobotGeometry = ROBOT) -> list[Point]:
    """台車を上から見た長方形の四隅（ワールド座標、反時計回り）。"""
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    hl, hw = robot.half_length, robot.half_width
    return [(pose.x + c * a - s * b, pose.y + s * a + c * b) for a, b in ((hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw))]


def _rect_corners(center: Sequence[float], size: Sequence[float]) -> list[Point]:
    cx, cy = center
    hx, hy = size[0] / 2, size[1] / 2
    return [(cx + hx, cy + hy), (cx - hx, cy + hy), (cx - hx, cy - hy), (cx + hx, cy - hy)]


def _point_segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = b[0] - a[0], b[1] - a[1]
    n2 = ax * ax + ay * ay
    t = 0.0 if n2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / n2))
    return math.hypot(p[0] - (a[0] + t * ax), p[1] - (a[1] + t * ay))


def _overlap_depth(a: list[Point], b: list[Point]) -> float:
    """2 つの凸多角形の、分離軸の定理による重なりの深さ（重ならなければ 0）。"""
    depth = math.inf
    for poly in (a, b):
        for i in range(len(poly)):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
            nx, ny = y2 - y1, x1 - x2
            n = math.hypot(nx, ny)
            nx, ny = nx / n, ny / n
            pa = [x * nx + y * ny for x, y in a]
            pb = [x * nx + y * ny for x, y in b]
            overlap = min(max(pa), max(pb)) - max(min(pa), min(pb))
            if overlap <= 0:
                return 0.0
            depth = min(depth, overlap)
    return depth


def polygon_distance(a: list[Point], b: list[Point]) -> float:
    """2 つの凸多角形の距離。重なっていれば重なりの深さを負の値で返す。"""
    depth = _overlap_depth(a, b)
    if depth > 0:
        return -depth
    d = math.inf
    for p, q in ((a, b), (b, a)):
        for pt in p:
            for i in range(len(q)):
                d = min(d, _point_segment_distance(pt, q[i], q[(i + 1) % len(q)]))
    return d


def robot_clearance(pose: Pose2D, scene: SceneSpec, robot: RobotGeometry = ROBOT) -> float:
    """台車の外形と、家具・壁の最小の距離 [m]。重なる・部屋からはみ出すときは負。"""
    poly = robot_corners(pose, robot)
    hx, hy = scene.room_half
    d = min(min(hx - abs(x), hy - abs(y)) for x, y in poly)
    for s in scene.surfaces.values():
        d = min(d, polygon_distance(poly, _rect_corners(s.center, s.size)))
    return d


# ------------------------------------------------------------
# 止め位置
# ------------------------------------------------------------
def approach_pose(
    target_xy: Sequence[float],
    surface_key: str,
    scene: SceneSpec,
    start: Pose2D,
    robot: RobotGeometry = ROBOT,
    reach: ReachConfig = REACH,
    nav: NavConfig = NAV,
) -> Pose2D | None:
    """物体 target_xy（家具 surface_key の上）を掴む・置くための台車の止め位置（設計書 8.1）。

    家具の 4 辺それぞれで、家具の方を向いて隙間 furniture_gap で止まり、横方向の位置を物体にそろえる。
    壁・ほかの家具に当たる（隙間が furniture_gap 未満）か物体が届かないときは、横に lateral_step ずつずらす
    （最大 max_lateral_shift）。条件を満たす候補のうち、start から最も近いもの。なければ None（届かない）。
    """
    surface = scene.surfaces[surface_key]
    n = round(nav.max_lateral_shift / nav.lateral_step)
    best, best_d = None, math.inf
    for side in SIDES:
        ideal = target_xy[1] if side in ("-x", "+x") else target_xy[0]
        found = None
        for k in range(n + 1):
            for sign in (1,) if k == 0 else (1, -1):
                pose = facing_pose(surface, side, ideal + sign * k * nav.lateral_step, robot)
                ok = robot_clearance(pose, scene, robot) >= robot.furniture_gap - 1e-9
                if ok and in_reach(pose, target_xy, reach):
                    found = pose
                    break
            if found is not None:
                break
        if found is not None:
            d = math.hypot(found.x - start.x, found.y - start.y)
            if d < best_d:
                best, best_d = found, d
    return best


# ------------------------------------------------------------
# 格子地図と A*
# ------------------------------------------------------------
@dataclass(frozen=True)
class GridMap:
    resolution: float  # [m]
    origin: tuple[float, float]  # 格子 (0, 0) の角のワールド座標
    occupied: np.ndarray  # (rows, cols) bool。膨張後。row が y、col が x

    def cell(self, xy: Sequence[float]) -> tuple[int, int]:
        return (
            int(math.floor((xy[1] - self.origin[1]) / self.resolution)),
            int(math.floor((xy[0] - self.origin[0]) / self.resolution)),
        )

    def center(self, cell: tuple[int, int]) -> Point:
        i, j = cell
        return (self.origin[0] + (j + 0.5) * self.resolution, self.origin[1] + (i + 0.5) * self.resolution)

    def occupied_at(self, xy: Sequence[float]) -> bool:
        i, j = self.cell(xy)
        rows, cols = self.occupied.shape
        return not (0 <= i < rows and 0 <= j < cols) or bool(self.occupied[i, j])


def build_grid(scene: SceneSpec, inflation: float, resolution: float = NAV.resolution) -> GridMap:
    """壁と家具を inflation [m] だけ膨らませた格子地図（台車の中心が入ってはいけない所が True）。"""
    hx, hy = scene.room_half
    cols, rows = round(2 * hx / resolution), round(2 * hy / resolution)
    xs = -hx + (np.arange(cols) + 0.5) * resolution
    ys = -hy + (np.arange(rows) + 0.5) * resolution
    x, y = np.meshgrid(xs, ys)
    occ = (np.abs(x) > hx - inflation) | (np.abs(y) > hy - inflation)
    for s in scene.surfaces.values():
        dx = np.maximum(np.abs(x - s.center[0]) - s.size[0] / 2, 0.0)
        dy = np.maximum(np.abs(y - s.center[1]) - s.size[1] / 2, 0.0)
        occ |= np.hypot(dx, dy) < inflation
    return GridMap(resolution, (-hx, -hy), occ)


def _segment_free(grid: GridMap, a: Point, b: Point) -> bool:
    n = max(int(math.ceil(math.dist(a, b) / (grid.resolution / 2))), 1)
    return all(not grid.occupied_at((a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)) for k in range(n + 1))


def plan_path(grid: GridMap, start: Sequence[float], goal: Sequence[float]) -> list[Point] | None:
    """8 近傍の A* で start から goal への経路（ワールド座標の点の列）。見通しの効く点を飛ばして間引く。

    start・goal が障害物（膨張後）の中にある、または経路がなければ None。
    """
    start, goal = (float(start[0]), float(start[1])), (float(goal[0]), float(goal[1]))
    if grid.occupied_at(start) or grid.occupied_at(goal):
        return None
    s, g = grid.cell(start), grid.cell(goal)
    rows, cols = grid.occupied.shape

    def h(c):
        di, dj = abs(c[0] - g[0]), abs(c[1] - g[1])
        return (di + dj) + (math.sqrt(2) - 2) * min(di, dj)

    open_heap = [(h(s), 0.0, s)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {s: 0.0}
    while open_heap:
        _, c_cost, cur = heapq.heappop(open_heap)
        if cur == g:
            break
        if c_cost > cost.get(cur, math.inf):
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == dj == 0:
                    continue
                nb = (cur[0] + di, cur[1] + dj)
                if not (0 <= nb[0] < rows and 0 <= nb[1] < cols) or grid.occupied[nb]:
                    continue
                nc = c_cost + (math.sqrt(2) if di and dj else 1.0)
                if nc < cost.get(nb, math.inf):
                    cost[nb] = nc
                    came[nb] = cur
                    heapq.heappush(open_heap, (nc + h(nb), nc, nb))
    if g not in cost:
        return None
    cells = [g]
    while cells[-1] != s:
        cells.append(came[cells[-1]])
    points = [start] + [grid.center(c) for c in reversed(cells[1:-1])] + [goal]
    # 見通しの効く点を飛ばす
    out = [points[0]]
    i = 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not _segment_free(grid, points[i], points[j]):
            j -= 1
        out.append(points[j])
        i = j
    return out


# ------------------------------------------------------------
# 経路（まっすぐ下がる → A* → まっすぐ寄せる）
# ------------------------------------------------------------
def _back(pose: Pose2D, d: float) -> Pose2D:
    return Pose2D(pose.x - d * math.cos(pose.yaw), pose.y - d * math.sin(pose.yaw), pose.yaw)


def _escape(pose: Pose2D, grid: GridMap, scene: SceneSpec, robot: RobotGeometry, nav: NavConfig) -> Pose2D | None:
    """pose が膨張の内側なら、向きを変えずにまっすぐ動いて膨張の外に出た点。

    後ろ → 斜め後ろ（左・右）→ 前 → 左 → 右の順に試す（#17）。止め位置では後ろ。家具の端の近くで壁の膨張にも
    入っているときは斜め後ろ（机の右端の物体を掴んだ後に、まっすぐ下がっても東の壁の膨張から出られなかった）。
    撮影で下がった後に壁を背にしているときは前。
    途中で家具・壁との距離が min_gap 未満になる方向は使わない。どの方向でも出られなければ None。
    """
    if not grid.occupied_at((pose.x, pose.y)):
        return pose
    for angle in (math.pi, 3 * math.pi / 4, -3 * math.pi / 4, 0.0, math.pi / 2, -math.pi / 2):
        ux, uy = math.cos(pose.yaw + angle), math.sin(pose.yaw + angle)
        d = 0.0
        while d < nav.max_backoff:
            d = min(d + nav.backoff_step, nav.max_backoff)
            p = Pose2D(pose.x + d * ux, pose.y + d * uy, pose.yaw)
            if robot_clearance(p, scene, robot) < nav.min_gap:
                break
            if not grid.occupied_at((p.x, p.y)):
                return p
    return None


def plan_route(
    scene: SceneSpec,
    grid: GridMap,
    start: Pose2D,
    goal: Pose2D,
    robot: RobotGeometry = ROBOT,
    nav: NavConfig = NAV,
) -> list[Pose2D] | None:
    """start から goal への経由点の列（start は含まない。最後は goal）。経路がなければ None。

    1. start が膨張の内側なら、向きを変えずにまっすぐ下がる（経由点の向きは start の向き）
    2. A* の経路（経由点の向きは goal の向き。膨張の外なので、どの向きでも家具・壁に当たらない）
    3. goal が膨張の内側なら、goal から下がった「寄せの開始点」から、goal の向きのまままっすぐ寄せる
    """
    s = _escape(start, grid, scene, robot, nav)
    g = _escape(goal, grid, scene, robot, nav)
    if s is None or g is None:
        return None
    mid = plan_path(grid, (s.x, s.y), (g.x, g.y))
    if mid is None:
        return None
    route: list[Pose2D] = []
    if s != start:
        route.append(s)
    route += [Pose2D(x, y, goal.yaw) for x, y in mid[1:-1]]
    if g != goal:
        route.append(g)
    route.append(goal)
    return route


def retreat_pose(
    current: Pose2D, distance: float, scene: SceneSpec, robot: RobotGeometry = ROBOT, nav: NavConfig = NAV
) -> Pose2D | None:
    """向きを変えずに distance だけまっすぐ下がった姿勢（写真でロボットが天板を隠したとき）。

    下がった姿勢で家具・壁との距離が min_gap 未満なら None。
    """
    p = _back(current, distance)
    return p if robot_clearance(p, scene, robot) >= nav.min_gap else None


# ------------------------------------------------------------
# 経路追従
# ------------------------------------------------------------
class PathFollower:
    """経由点に向かう比例制御＋飽和の速度指令（設計書 8.3。#2 の velocity_command）。

    途中の経由点は、位置 waypoint_tol・向き waypoint_yaw_tol 以内で次へ進む（向きもそろえてから進むので、
    寄せの開始点で向きがそろってから家具に寄せる）。最後の経由点は goal_tol（位置・向き）以内で止まる。
    """

    def __init__(
        self,
        path: Sequence[Pose2D],
        v_max: float = 0.5,
        w_max: float = 1.0,
        k_lin: float = 2.0,
        k_ang: float = 2.0,
        waypoint_tol: float = 0.05,
        goal_tol: tuple[float, float] = (0.01, 0.017),
        waypoint_yaw_tol: float = 0.1,
    ) -> None:
        if not path:
            raise ValueError("経路が空です")
        self.path = list(path)
        self.v_max, self.w_max, self.k_lin, self.k_ang = v_max, w_max, k_lin, k_ang
        self.waypoint_tol, self.goal_tol, self.waypoint_yaw_tol = waypoint_tol, goal_tol, waypoint_yaw_tol
        self.index = 0

    def command(self, pose: Pose2D) -> tuple[float, float, float] | None:
        """今の姿勢 → 速度指令 (vx, vy, wz)（ワールド座標）。到着したら None。"""
        while True:
            wp = self.path[self.index]
            dx, dy = wp.x - pose.x, wp.y - pose.y
            dist, dyaw = math.hypot(dx, dy), wrap_angle(wp.yaw - pose.yaw)
            if self.index == len(self.path) - 1:
                if dist <= self.goal_tol[0] and abs(dyaw) <= self.goal_tol[1]:
                    return None
                break
            if dist <= self.waypoint_tol and abs(dyaw) <= self.waypoint_yaw_tol:
                self.index += 1
                continue
            break
        vx, vy = self.k_lin * dx, self.k_lin * dy
        v = math.hypot(vx, vy)
        if v > self.v_max:
            vx, vy = vx * self.v_max / v, vy * self.v_max / v
        wz = max(-self.w_max, min(self.w_max, self.k_ang * dyaw))
        return (vx, vy, wz)
