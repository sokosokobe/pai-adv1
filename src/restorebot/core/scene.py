"""場面の定義（部屋・置き場所・物体・いつもの配置・天井カメラ）と散らかし方（設計書 5 章）。

画像の生成・VLM・移動・評価で、ここの同じ定義を使う。#9 の scripts/experiments/vlm_common.py を書き直したもの。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from restorebot.core.camera import DownCamera
from restorebot.core.config import PLACE_MARGIN, SCATTER, WORK_DEPTH, ScatterConfig
from restorebot.core.navigation import in_reach
from restorebot.core.types import Pose2D

SIDES = ("-x", "+x", "-y", "+y")


@dataclass(frozen=True)
class Rect:
    """水平な長方形（中心と大きさ）。"""

    center: tuple[float, float]
    size: tuple[float, float]


@dataclass(frozen=True)
class Surface(Rect):
    """物体を置ける水平な面（家具の天板）。z は面の高さ。front は台車が向かう手前の辺（"-x"・"+x"・"-y"・"+y"）。"""

    key: str = ""
    label: str = ""
    z: float = 0.0
    front: str = "-y"

    def __post_init__(self) -> None:
        if self.front not in SIDES:
            raise ValueError(f"front は {SIDES} のどれか: {self.front}")

    def contains(self, xy: Sequence[float], r: float, margin: float = PLACE_MARGIN) -> bool:
        """半径 r の物体を、縁から margin 以上離して置けるか。"""
        hx, hy = self.size[0] / 2 - r - margin, self.size[1] / 2 - r - margin
        return abs(xy[0] - self.center[0]) <= hx and abs(xy[1] - self.center[1]) <= hy

    def depth_from_front(self, xy: Sequence[float]) -> float:
        """点 xy の、手前の縁から奥へ向かう距離 [m]。"""
        cx, cy = self.center
        hw, hh = self.size[0] / 2, self.size[1] / 2
        return {
            "-x": xy[0] - (cx - hw),
            "+x": (cx + hw) - xy[0],
            "-y": xy[1] - (cy - hh),
            "+y": (cy + hh) - xy[1],
        }[self.front]

    def in_work_area(self, xy: Sequence[float], r: float, depth: float = WORK_DEPTH) -> bool:
        """作業範囲：半径 r の物体を縁から PLACE_MARGIN 以上離して置け、中心が手前の縁から depth 以内にあるか。"""
        return self.contains(xy, r) and self.depth_from_front(xy) <= depth + 1e-9

    def work_rect(self, depth: float = WORK_DEPTH) -> Rect:
        """作業範囲を含む長方形（手前の縁から depth までの帯）。"""
        cx, cy = self.center
        w, h = self.size
        if self.front in ("-x", "+x"):
            d = min(depth, w)
            sign = -1 if self.front == "-x" else 1
            return Rect(center=(cx + sign * (w - d) / 2, cy), size=(d, h))
        d = min(depth, h)
        sign = -1 if self.front == "-y" else 1
        return Rect(center=(cx, cy + sign * (h - d) / 2), size=(w, d))


@dataclass(frozen=True)
class ObjectSpec:
    """物体。size は形ごとに cube: (sx, sy, sz)、cylinder: (半径, 高さ)。"""

    key: str
    name: str  # 表示名（VLM のプロンプトに渡す名前）
    shape: str
    size: tuple[float, ...]
    color: tuple[float, float, float]  # RGB（0〜1）
    mass: float  # [kg]

    @property
    def grasp_width(self) -> float:
        """上から掴むときの指の間の幅（箱は短い辺、円柱は直径）。"""
        if self.shape == "cube":
            return min(self.size[0], self.size[1])
        return 2 * self.size[0]

    @property
    def height(self) -> float:
        if self.shape == "cube":
            return self.size[2]
        return self.size[1]

    @property
    def footprint_radius(self) -> float:
        """上から見たときの外接円の半径（向きによらず重ならないようにするため）。"""
        if self.shape == "cube":
            return math.hypot(self.size[0], self.size[1]) / 2
        return self.size[0]


@dataclass(frozen=True)
class Placement:
    """物体の置き方：面、面の上の位置 (x, y) [m]、向き yaw [rad]。"""

    surface: str
    x: float
    y: float
    yaw: float = 0.0


@dataclass(frozen=True)
class SceneSpec:
    key: str
    room_half: tuple[float, float]  # 部屋の内側の半幅 (x, y) [m]
    surfaces: dict[str, Surface] = field(hash=False)
    objects: tuple[ObjectSpec, ...]
    tidy: dict[str, Placement] = field(hash=False)  # いつもの配置
    camera: DownCamera
    marker: Rect | None = None  # 画像の向きを確かめる目印（床の左上 = -x, +y の隅）

    def __post_init__(self) -> None:
        keys = [o.key for o in self.objects]
        if len(set(keys)) != len(keys):
            raise ValueError(f"物体の key が重複しています: {keys}")
        if set(self.tidy) != set(keys):
            raise ValueError(f"いつもの配置の物体が物体の一覧と一致しません: {sorted(self.tidy)} / {sorted(keys)}")
        for key, p in self.tidy.items():
            if p.surface not in self.surfaces:
                raise ValueError(f"{key} のいつもの配置の面 {p.surface} がありません")

    def object(self, key: str) -> ObjectSpec:
        for o in self.objects:
            if o.key == key:
                return o
        raise KeyError(key)

    def object_center(self, key: str, placement: Placement) -> tuple[float, float, float]:
        """物体の中心のワールド座標（面の上に置いたとき）。"""
        return (placement.x, placement.y, self.surfaces[placement.surface].z + self.object(key).height / 2)


# #9 の自宅風の部屋：床 4.0 × 3.0m。置き場所は机（天板 0.70m）と低い棚（天板 0.45m）だけ（床には置かない）。
# 物体は #4 で確認した把持の範囲（掴む幅 4〜5cm、高さ 10cm 以下、0.5kg 以下）に収める。
# 机の 3 個のいつもの位置は、作業範囲の中で、M1 の止め位置から楽に届く所に置く（設計書 5.2。#9 からの変更）
_HOME_DESK = Surface(center=(0.8, 0.85), size=(1.2, 0.6), key="desk", label="机の上", z=0.70, front="-y")
_HOME_SIDE = Surface(center=(-1.5, -0.3), size=(0.4, 0.8), key="side_table", label="低い棚の上", z=0.45, front="+x")
HOME_V2 = SceneSpec(
    key="home_v2",
    room_half=(2.0, 1.5),
    surfaces={s.key: s for s in (_HOME_DESK, _HOME_SIDE)},
    objects=(
        ObjectSpec("cup", "赤いコップ", "cylinder", (0.025, 0.09), (0.85, 0.1, 0.1), mass=0.2),
        ObjectSpec("pen_stand", "青いペン立て", "cylinder", (0.0225, 0.10), (0.1, 0.3, 0.9), mass=0.15),
        ObjectSpec("spray", "緑のスプレー缶", "cylinder", (0.02, 0.10), (0.1, 0.65, 0.2), mass=0.3),
        # リモコンは高さ 2.5cm では指先が机に当たって掴めなかった（#9：0/3）。4cm で 3/3 成功したので 4cm にする
        ObjectSpec("remote", "黒いリモコン", "cube", (0.15, 0.045, 0.04), (0.05, 0.05, 0.05), mass=0.15),
        ObjectSpec("small_box", "黄色い小箱", "cube", (0.06, 0.04, 0.05), (0.95, 0.8, 0.1), mass=0.1),
    ),
    tidy={
        "cup": Placement("desk", 0.45, 0.72),
        "pen_stand": Placement("desk", 0.75, 0.75),
        "remote": Placement("desk", 1.05, 0.72),
        "spray": Placement("side_table", -1.5, -0.10),
        "small_box": Placement("side_table", -1.5, -0.50),
    },
    # 部屋全体が写るよう水平画角 82°、1280 × 960（#9）
    camera=DownCamera(position=(0.0, 0.0, 2.6), width=1280, height=960, focal_length=12.0, horizontal_aperture=20.955),
    marker=Rect(center=(-1.8, 1.3), size=(0.15, 0.15)),
)


# ------------------------------------------------------------
# 散らかし方
# ------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    """散らかった状態。placements は全物体の置き方、moved は動かした物体（SceneSpec.objects の順）。"""

    seed: int
    moved: tuple[str, ...]
    placements: dict[str, Placement] = field(hash=False)


def _gap(scene: SceneSpec, key_a: str, a: Placement, key_b: str, b: Placement) -> float:
    """2 つの物体の外接円どうしの隙間 [m]。"""
    ra, rb = scene.object(key_a).footprint_radius, scene.object(key_b).footprint_radius
    return math.hypot(a.x - b.x, a.y - b.y) - ra - rb


def generate_scenario(
    scene: SceneSpec,
    seed: int,
    *,
    surfaces: Sequence[str] | None = None,
    reach_from: Pose2D | None = None,
    config: ScatterConfig = SCATTER,
) -> Scenario:
    """「いつもの配置」から 1〜3 個を動かした状態を作る。同じシードなら毎回同じになる。

    - 動かす物体は、いつもの位置が surfaces（省略時はすべての面）の上にあるもの。動かす先も surfaces の作業範囲の中
    - reach_from を与えると、動かす物体のいつもの位置と動かす先を、その台車の位置から楽に届く範囲に限る（M1）
    - 動かす先は、どの物体のいつもの位置・今の位置からも、外接円どうしの隙間を config.clearance 以上あける
      （戻し先がふさがらない。設計書 5.2）
    - 動かす先が見つからない物体があれば、動かす物体の選び直しからやり直す（config.max_attempts 回まで）
    """
    allowed = tuple(scene.surfaces) if surfaces is None else tuple(surfaces)
    rng = np.random.default_rng(seed)
    eligible = [
        o.key
        for o in scene.objects
        if scene.tidy[o.key].surface in allowed
        and (reach_from is None or in_reach(reach_from, (scene.tidy[o.key].x, scene.tidy[o.key].y)))
    ]
    max_moved = min(config.max_moved, len(eligible))
    if max_moved < config.min_moved:
        raise ValueError(f"動かせる物体が {len(eligible)} 個しかありません（{config.min_moved} 個以上が必要）")
    for _ in range(config.max_attempts):
        k = int(rng.integers(config.min_moved, max_moved + 1))
        chosen = set(rng.choice(eligible, size=k, replace=False).tolist())
        moved = tuple(o.key for o in scene.objects if o.key in chosen)
        placements = dict(scene.tidy)
        for key in moved:
            p = _sample_destination(scene, key, placements, rng, allowed=allowed, reach_from=reach_from, config=config)
            if p is None:
                break
            placements[key] = p
        else:
            return Scenario(seed=seed, moved=moved, placements=placements)
    raise RuntimeError(f"散らかし方が見つかりませんでした（seed={seed}）")


def _sample_destination(
    scene: SceneSpec,
    key: str,
    placements: dict[str, Placement],
    rng: np.random.Generator,
    *,
    allowed: Sequence[str],
    reach_from: Pose2D | None,
    config: ScatterConfig,
) -> Placement | None:
    """物体 key の動かす先を、制約を満たすまで乱数で探す。config.max_tries 回で見つからなければ None。"""
    r = scene.object(key).footprint_radius
    others = list(scene.tidy.items()) + [(k, p) for k, p in placements.items() if k != key]
    for _ in range(config.max_tries):
        surface = scene.surfaces[allowed[int(rng.integers(len(allowed)))]]
        band = surface.work_rect()
        x = float(rng.uniform(band.center[0] - band.size[0] / 2, band.center[0] + band.size[0] / 2))
        y = float(rng.uniform(band.center[1] - band.size[1] / 2, band.center[1] + band.size[1] / 2))
        candidate = Placement(surface.key, x, y, float(rng.uniform(-math.pi, math.pi)))
        if not surface.in_work_area((x, y), r):
            continue
        if reach_from is not None and not in_reach(reach_from, (x, y)):
            continue
        if any(p.surface == surface.key and _gap(scene, key, candidate, k, p) < config.clearance for k, p in others):
            continue
        return candidate
    return None
