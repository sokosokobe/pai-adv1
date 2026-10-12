"""Issue #5 天井カメラと VLM による差分検出の検証用ロジック（Isaac Sim に依存しない）。

- 部屋・置き場所・物体の定義と、「いつもの状態」「散らかった状態」の配置の生成（シード固定）
- 真下向きの天井カメラの投影（ワールド座標 → 画像上の座標）と正解データ
- VLM のインターフェース、プロンプト、応答（JSON）の解析、正解との照合・集計、費用の計算

単位は SI（m, rad）。ワールド座標は Z 軸が上向き。
画像の座標は左上が原点で、u（x）が右、v（y）が下向きのピクセル。カメラは真下を向き、
画像の右がワールドの +x、画像の上がワールドの +y になる（vlm_scene.py で画像を保存して確認する）。
"""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

# ------------------------------------------------------------
# 部屋・置き場所・物体
# ------------------------------------------------------------
ROOM_HALF = (1.6, 1.2)  # 部屋の内側の半幅 (x, y) [m]。床は 3.2m × 2.4m
PLACE_MARGIN = 0.02  # 置き場所の縁からの余白 [m]
OBJECT_GAP = 0.03  # 物体どうしの最小の隙間 [m]


@dataclass(frozen=True)
class Rect:
    """水平な長方形（中心と大きさ）。"""

    center: tuple[float, float]
    size: tuple[float, float]

    def overlaps_circle(self, xy: Sequence[float], r: float) -> bool:
        """半径 r の円と重なるか。"""
        hx, hy = self.size[0] / 2, self.size[1] / 2
        dx = max(abs(xy[0] - self.center[0]) - hx, 0.0)
        dy = max(abs(xy[1] - self.center[1]) - hy, 0.0)
        return dx * dx + dy * dy < r * r


@dataclass(frozen=True)
class Surface(Rect):
    """物体を置ける水平な面。z は面の高さ。exclusions は面の中で置けない所（床の上の家具など）。"""

    key: str = ""
    label: str = ""
    z: float = 0.0
    exclusions: tuple[Rect, ...] = ()

    def contains(self, xy: Sequence[float], r: float, margin: float = PLACE_MARGIN) -> bool:
        """半径 r の物体を、縁から margin 以上離して置けるか。"""
        hx, hy = self.size[0] / 2 - r - margin, self.size[1] / 2 - r - margin
        if abs(xy[0] - self.center[0]) > hx or abs(xy[1] - self.center[1]) > hy:
            return False
        return not any(ex.overlaps_circle(xy, r + margin) for ex in self.exclusions)


_DESK = Surface(center=(0.75, 0.7), size=(1.2, 0.6), key="desk", label="机の上", z=0.72)
_SHELF = Surface(center=(-1.1, 0.0), size=(0.4, 1.0), key="shelf", label="棚の上", z=0.8)
# 画像の向きを確かめる目印（床の左上 = -x, +y の隅に置く平たい板）。画像でも左上に写るはず
ORIENTATION_MARKER = Rect(center=(-1.45, 1.05), size=(0.15, 0.15))
_FLOOR = Surface(
    center=(0.0, 0.0),
    size=(2 * ROOM_HALF[0], 2 * ROOM_HALF[1]),
    key="floor",
    label="床",
    z=0.0,
    exclusions=(Rect(_DESK.center, _DESK.size), Rect(_SHELF.center, _SHELF.size), ORIENTATION_MARKER),
)
SURFACES: dict[str, Surface] = {s.key: s for s in (_DESK, _SHELF, _FLOOR)}
SURFACE_WEIGHTS = {"floor": 0.6, "desk": 0.2, "shelf": 0.2}  # 動かす先の面を選ぶ確率


@dataclass(frozen=True)
class ObjectSpec:
    """物体。size は形ごとに cube: (sx, sy, sz)、cylinder / cone: (半径, 高さ)、sphere: (半径,)。"""

    key: str
    name: str  # 表示名（プロンプトに渡す名前）
    shape: str
    size: tuple[float, ...]
    color: tuple[float, float, float]  # RGB（0〜1）
    mass: float = 0.1  # [kg]（#9 から使う。#5 の物体は描画だけなので既定値）

    @property
    def grasp_width(self) -> float:
        """上から掴むときの指の間の幅（箱は短い辺、それ以外は直径）。"""
        if self.shape == "cube":
            return min(self.size[0], self.size[1])
        return 2 * self.size[0]

    @property
    def height(self) -> float:
        if self.shape == "cube":
            return self.size[2]
        if self.shape == "sphere":
            return 2 * self.size[0]
        return self.size[1]

    @property
    def footprint_radius(self) -> float:
        """上から見たときの外接円の半径（向きによらず重ならないようにするため）。"""
        if self.shape == "cube":
            return math.hypot(self.size[0], self.size[1]) / 2
        return self.size[0]


OBJECTS: tuple[ObjectSpec, ...] = (
    ObjectSpec("mug", "赤いマグカップ", "cylinder", (0.045, 0.10), (0.85, 0.1, 0.1)),
    ObjectSpec("box", "青い箱", "cube", (0.18, 0.12, 0.08), (0.1, 0.3, 0.9)),
    ObjectSpec("ball", "緑のボール", "sphere", (0.06,), (0.1, 0.7, 0.2)),
    ObjectSpec("book", "黄色い本", "cube", (0.24, 0.17, 0.035), (0.95, 0.85, 0.1)),
    ObjectSpec("cone", "紫のコーン", "cone", (0.06, 0.14), (0.6, 0.2, 0.8)),
)


@dataclass(frozen=True)
class Placement:
    """物体の置き方：面、面の上の位置 (x, y) [m]、向き yaw [rad]。"""

    surface: str
    x: float
    y: float
    yaw: float = 0.0


TIDY_LAYOUT: dict[str, Placement] = {
    "mug": Placement("desk", 0.40, 0.80),
    "book": Placement("desk", 0.80, 0.65),
    "box": Placement("desk", 1.15, 0.80),
    "ball": Placement("shelf", -1.1, 0.30),
    "cone": Placement("shelf", -1.1, -0.25),
}


def object_center(
    spec: ObjectSpec, placement: Placement, *, scene: SceneSpec | None = None
) -> tuple[float, float, float]:
    """物体の中心のワールド座標（面の上に置いたとき）。scene を省略すると #5 の部屋（ROOM_V1）。"""
    surfaces = SURFACES if scene is None else scene.surfaces
    return (placement.x, placement.y, surfaces[placement.surface].z + spec.height / 2)


@dataclass(frozen=True)
class Scenario:
    """散らかった状態。placements は全物体の置き方、moved は動かした物体（OBJECTS の順）。"""

    index: int
    moved: tuple[str, ...]
    placements: dict[str, Placement] = field(hash=False)


def generate_scenarios(
    n: int,
    *,
    seed: int,
    min_moved: int = 1,
    max_moved: int = 3,
    min_shift: float = 0.4,
    max_shift: float | None = None,
    same_surface: bool = False,
    max_tries: int = 1000,
    scene: SceneSpec | None = None,
) -> list[Scenario]:
    """「いつもの状態」から 1〜3 個を別の場所へ動かした状態を n 個作る。シードを固定すれば毎回同じになる。

    既定は、置き場所（面）を選び直して min_shift [m] 以上離れた所へ動かす。
    same_surface=True のときは同じ面の上で、元の位置から min_shift〜max_shift [m] だけずらす（小さな移動）。
    """
    if same_surface and max_shift is None:
        raise ValueError("same_surface=True のときは max_shift を指定してください")
    objects = OBJECTS if scene is None else scene.objects
    tidy_layout = TIDY_LAYOUT if scene is None else scene.tidy
    surfaces = SURFACES if scene is None else scene.surfaces
    surface_weights = SURFACE_WEIGHTS if scene is None else scene.surface_weights
    rng = np.random.default_rng(seed)
    specs = {o.key: o for o in objects}
    keys = [o.key for o in objects]
    surface_keys = list(surface_weights)
    weights = np.array([surface_weights[k] for k in surface_keys])
    scenarios = []
    for index in range(n):
        k = int(rng.integers(min_moved, max_moved + 1))
        chosen = set(rng.choice(keys, size=k, replace=False).tolist())
        moved = tuple(key for key in keys if key in chosen)
        placements = dict(tidy_layout)
        for key in moved:
            spec, tidy = specs[key], tidy_layout[key]
            for _ in range(max_tries):
                if same_surface:
                    surface = surfaces[tidy.surface]
                    r = float(rng.uniform(min_shift, max_shift))
                    a = float(rng.uniform(-math.pi, math.pi))
                    x, y = tidy.x + r * math.cos(a), tidy.y + r * math.sin(a)
                else:
                    surface = surfaces[surface_keys[int(rng.choice(len(surface_keys), p=weights / weights.sum()))]]
                    x = float(
                        rng.uniform(surface.center[0] - surface.size[0] / 2, surface.center[0] + surface.size[0] / 2)
                    )
                    y = float(
                        rng.uniform(surface.center[1] - surface.size[1] / 2, surface.center[1] + surface.size[1] / 2)
                    )
                yaw = float(rng.uniform(-math.pi, math.pi))
                if not surface.contains((x, y), spec.footprint_radius):
                    continue
                shift = math.hypot(x - tidy.x, y - tidy.y)
                if shift < min_shift or (max_shift is not None and shift > max_shift):
                    continue
                if any(
                    p.surface == surface.key
                    and math.hypot(x - p.x, y - p.y)
                    < spec.footprint_radius + specs[other].footprint_radius + OBJECT_GAP
                    for other, p in placements.items()
                    if other != key
                ):
                    continue
                placements[key] = Placement(surface.key, x, y, yaw)
                break
            else:
                raise RuntimeError(f"{key} の置き場所が見つかりませんでした（シナリオ {index}）")
        scenarios.append(Scenario(index=index, moved=moved, placements=placements))
    return scenarios


# ------------------------------------------------------------
# 天井カメラ（真下向き）
# ------------------------------------------------------------
@dataclass(frozen=True)
class DownCamera:
    """真下を向いたピンホールカメラ。画像の右 = ワールド +x、画像の上 = ワールド +y。

    focal_length と horizontal_aperture は同じ単位（USD の Camera と同じく mm など）。画素は正方形。
    """

    position: tuple[float, float, float]
    width: int
    height: int
    focal_length: float
    horizontal_aperture: float

    @property
    def fx(self) -> float:
        return self.focal_length / self.horizontal_aperture * self.width

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return self.width / 2

    @property
    def cy(self) -> float:
        return self.height / 2

    def project(self, point: Sequence[float]) -> tuple[float, float]:
        """ワールド座標の点 → 画像上の座標 (u, v) [px]。"""
        x0, y0, z0 = self.position
        depth = z0 - point[2]
        if depth <= 0:
            raise ValueError("カメラより上の点は投影できません")
        return (self.cx + self.fx * (point[0] - x0) / depth, self.cy - self.fy * (point[1] - y0) / depth)

    def meters_per_pixel(self, z: float) -> float:
        """高さ z の水平面上で、1 ピクセルが何 m に当たるか。"""
        return (self.position[2] - z) / self.fx


# 天井（高さ 2.6m）の中央に付けるカメラ。床の約 4.0m × 3.0m が写る（床の上で約 3.9mm/px）
CEILING_CAMERA = DownCamera(
    position=(0.0, 0.0, 2.6), width=1024, height=768, focal_length=13.5, horizontal_aperture=20.955
)


# ------------------------------------------------------------
# 場面の定義（部屋・置き場所・物体・いつもの配置・天井カメラ）
# ------------------------------------------------------------
@dataclass(frozen=True)
class SceneSpec:
    key: str
    room_half: tuple[float, float]  # 部屋の内側の半幅 (x, y) [m]
    surfaces: dict[str, Surface] = field(hash=False)
    surface_weights: dict[str, float] = field(hash=False)  # 動かす先の面を選ぶ確率
    objects: tuple[ObjectSpec, ...] = ()
    tidy: dict[str, Placement] = field(default_factory=dict, hash=False)
    camera: DownCamera | None = None
    marker: Rect | None = None  # 画像の向きを確かめる目印（床の左上の隅）


# #5 の部屋（これまでの定数そのもの。#5 の結果を再現できるように残す）
ROOM_V1 = SceneSpec(
    key="room_v1",
    room_half=ROOM_HALF,
    surfaces=SURFACES,
    surface_weights=SURFACE_WEIGHTS,
    objects=OBJECTS,
    tidy=TIDY_LAYOUT,
    camera=CEILING_CAMERA,
    marker=ORIENTATION_MARKER,
)

# #9 の自宅風の部屋：床 4.0 × 3.0m。置き場所は机（天板 0.70m）と低い棚（天板 0.45m）だけ（床には置かない）。
# 物体は #4 で確認した把持の範囲（掴む幅 4〜5cm、高さ 10cm 以下、0.5kg 以下）に収める
_HOME_HALF = (2.0, 1.5)
_HOME_DESK = Surface(center=(0.8, 0.85), size=(1.2, 0.6), key="desk", label="机の上", z=0.70)
_HOME_SIDE = Surface(center=(-1.5, -0.3), size=(0.4, 0.8), key="side_table", label="低い棚の上", z=0.45)
HOME_V2 = SceneSpec(
    key="home_v2",
    room_half=_HOME_HALF,
    surfaces={s.key: s for s in (_HOME_DESK, _HOME_SIDE)},
    surface_weights={"desk": 0.5, "side_table": 0.5},
    objects=(
        ObjectSpec("cup", "赤いコップ", "cylinder", (0.025, 0.09), (0.85, 0.1, 0.1), mass=0.2),
        ObjectSpec("pen_stand", "青いペン立て", "cylinder", (0.0225, 0.10), (0.1, 0.3, 0.9), mass=0.15),
        ObjectSpec("spray", "緑のスプレー缶", "cylinder", (0.02, 0.10), (0.1, 0.65, 0.2), mass=0.3),
        # リモコンは高さ 2.5cm では指先が机に当たって掴めなかった（0/3）。4cm で 3/3 成功したので 4cm にする
        ObjectSpec("remote", "黒いリモコン", "cube", (0.15, 0.045, 0.04), (0.05, 0.05, 0.05), mass=0.15),
        ObjectSpec("small_box", "黄色い小箱", "cube", (0.06, 0.04, 0.05), (0.95, 0.8, 0.1), mass=0.1),
    ),
    tidy={
        "cup": Placement("desk", 0.40, 0.95),
        "pen_stand": Placement("desk", 0.70, 1.00),
        "remote": Placement("desk", 1.05, 0.80),
        "spray": Placement("side_table", -1.5, -0.10),
        "small_box": Placement("side_table", -1.5, -0.50),
    },
    # 部屋全体が写るよう、#5 より画角を広げ（水平 82°）、物体が小さく写る分だけ解像度を上げる
    camera=DownCamera(position=(0.0, 0.0, 2.6), width=1280, height=960, focal_length=12.0, horizontal_aperture=20.955),
    marker=Rect(center=(-1.8, 1.3), size=(0.15, 0.15)),
)
SCENES = {s.key: s for s in (ROOM_V1, HOME_V2)}


# ------------------------------------------------------------
# ロボット（今の画像にだけ写す。いつもの写真を撮るときはロボットは部屋にいない想定）
# ------------------------------------------------------------
ROBOT_HALF_SIZE = (0.47, 0.40)  # 台車の前後・左右の半分の長さ [m]（前後は #2 の実測 0.466m、左右は推定）
ROBOT_GAP = 0.10  # 台車の前面と家具の隙間 [m]（#3：机は台車の前面から 10cm 程度に寄せる）


def robot_rect(pose: Sequence[float]) -> Rect:
    """台車を上から見たときの外接長方形（軸に平行）。"""
    c, s = abs(math.cos(pose[2])), abs(math.sin(pose[2]))
    hx, hy = ROBOT_HALF_SIZE
    return Rect(center=(pose[0], pose[1]), size=(2 * (hx * c + hy * s), 2 * (hx * s + hy * c)))


def rects_overlap(a: Rect, b: Rect) -> bool:
    return (
        abs(a.center[0] - b.center[0]) < (a.size[0] + b.size[0]) / 2
        and abs(a.center[1] - b.center[1]) < (a.size[1] + b.size[1]) / 2
    )


def robot_faces(pose: Sequence[float], furniture: Rect) -> bool:
    """台車の前方（前面の少し先）が家具の上にあるか（＝家具の正面で、家具の方を向いているか）。"""
    d = ROBOT_HALF_SIZE[0] + ROBOT_GAP + 0.05
    x, y = pose[0] + d * math.cos(pose[2]), pose[1] + d * math.sin(pose[2])
    hx, hy = furniture.size[0] / 2, furniture.size[1] / 2
    return abs(x - furniture.center[0]) <= hx and abs(y - furniture.center[1]) <= hy


def sample_robot_poses(
    n: int, *, seed: int, scene: SceneSpec, max_tries: int = 1000
) -> list[tuple[float, float, float]]:
    """家具のどれかの正面に、家具の方を向いて止まった台車の姿勢 (x, y, yaw) を n 個作る（作業中の想定）。"""
    rng = np.random.default_rng(seed)
    furniture = list(scene.surfaces.values())
    hx_room, hy_room = scene.room_half
    d = ROBOT_HALF_SIZE[0] + ROBOT_GAP
    poses = []
    for _ in range(n):
        for _ in range(max_tries):
            f = furniture[int(rng.integers(len(furniture)))]
            side = int(rng.integers(4))  # 0: 家具の -x 側、1: +x 側、2: -y 側、3: +y 側
            fx, fy, fw, fh = f.center[0], f.center[1], f.size[0], f.size[1]
            if side in (0, 1):
                sign = -1 if side == 0 else 1
                x, y = fx + sign * (fw / 2 + d), fy + float(rng.uniform(-fh / 2, fh / 2))
                yaw = 0.0 if side == 0 else math.pi
            else:
                sign = -1 if side == 2 else 1
                x, y = fx + float(rng.uniform(-fw / 2, fw / 2)), fy + sign * (fh / 2 + d)
                yaw = math.pi / 2 if side == 2 else -math.pi / 2
            r = robot_rect((x, y, yaw))
            inside = abs(x) + r.size[0] / 2 <= hx_room and abs(y) + r.size[1] / 2 <= hy_room
            if inside and not any(rects_overlap(r, Rect(g.center, g.size)) for g in furniture):
                poses.append((x, y, yaw))
                break
        else:
            raise RuntimeError("ロボットの置き場所が見つかりませんでした")
    return poses


# ------------------------------------------------------------
# 正解データ
# ------------------------------------------------------------
@dataclass(frozen=True)
class ScenarioTruth:
    moved: tuple[str, ...]
    names: dict[str, str]  # key → 表示名
    current_px: dict[str, tuple[float, float]]  # 散らかった画像での全物体の中心
    target_px: dict[str, tuple[float, float]]  # 動いた物体の戻し先（いつもの画像での中心）


def scenario_truth(scenario: Scenario, camera: DownCamera, *, scene: SceneSpec | None = None) -> ScenarioTruth:
    objects = OBJECTS if scene is None else scene.objects
    tidy = TIDY_LAYOUT if scene is None else scene.tidy
    specs = {o.key: o for o in objects}
    return ScenarioTruth(
        moved=scenario.moved,
        names={o.key: o.name for o in objects},
        current_px={k: camera.project(object_center(specs[k], p, scene=scene)) for k, p in scenario.placements.items()},
        target_px={k: camera.project(object_center(specs[k], tidy[k], scene=scene)) for k in scenario.moved},
    )


# ------------------------------------------------------------
# 座標の表し方（モデルごとに得意な形式が違うので、プロンプトで指定し、受け取ったらピクセルにそろえる）
# ------------------------------------------------------------
COORD_FORMATS = ("pixel_xy", "norm1000_xy", "norm1000_yx")


def to_pixels(point: Sequence[float], coord_format: str, width: int, height: int) -> tuple[float, float]:
    """VLM が返した点 → 画像上の座標 (u, v) [px]。"""
    a, b = float(point[0]), float(point[1])
    if coord_format == "pixel_xy":
        return (a, b)
    if coord_format == "norm1000_xy":
        return (a / 1000 * width, b / 1000 * height)
    if coord_format == "norm1000_yx":
        return (b / 1000 * width, a / 1000 * height)
    raise ValueError(f"未知の座標の形式です: {coord_format}")


def _coord_description(coord_format: str, width: int, height: int) -> str:
    if coord_format == "pixel_xy":
        return (
            f"Points are [x, y] in pixels of the {width}x{height} image: origin at the top-left corner, "
            "x increases to the right, y increases downward."
        )
    if coord_format == "norm1000_xy":
        return (
            "Points are [x, y] normalized to 0-1000: [0, 0] is the top-left corner and [1000, 1000] is the "
            "bottom-right corner of the image."
        )
    if coord_format == "norm1000_yx":
        return (
            "Points are [y, x] normalized to 0-1000: [0, 0] is the top-left corner and [1000, 1000] is the "
            "bottom-right corner of the image."
        )
    raise ValueError(f"未知の座標の形式です: {coord_format}")


# ------------------------------------------------------------
# プロンプトと JSON の形式
# ------------------------------------------------------------
RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "moved_objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "current_location": {"type": "string"},
                    "current_point": {"type": "array", "items": {"type": "number"}},
                    "target_location": {"type": "string"},
                    "target_point": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["name", "current_location", "current_point", "target_location", "target_point"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["moved_objects"],
    "additionalProperties": False,
}


ROBOT_NOTE = (
    "A mobile robot (a black wheeled base with a white robot arm) may appear in the CURRENT image. "
    "It is not a household object: never list it, even though it is absent from the REFERENCE image."
)


def build_prompt(
    *, coord_format: str, width: int, height: int, object_names: Sequence[str] | None, robot_note: bool = False
) -> str:
    """差分検出のプロンプト。画像は 1 枚目に「いつもの状態」、2 枚目に「今の状態」を渡す前提。"""
    if object_names:
        names = "\n".join(f"- {n}" for n in object_names)
        name_rule = "The room contains these movable objects. Use exactly one of these names for `name`:\n" + names
    else:
        name_rule = "Use a short Japanese name for `name`, e.g. color + kind of object."
    robot = ROBOT_NOTE + "\n" if robot_note else ""
    return f"""You are given two top-down images of the same room, taken by the same fixed ceiling camera.
Image 1 is the REFERENCE: the usual, tidy state of the room.
Image 2 is the CURRENT state. Some objects have been moved from their usual places.

Task: list every object whose position in the CURRENT image differs from its position in the REFERENCE image.
Do not list objects that stayed in place. Furniture (desk, shelf) and walls never move.
{robot}
{name_rule}

For each moved object, report:
- current_location: where it is now (short Japanese description, e.g. 床の右下, 机の上の左端)
- current_point: the center of the object in the CURRENT image
- target_location: where it should be returned to (short Japanese description)
- target_point: the center of the object in the REFERENCE image (its usual place)

{_coord_description(coord_format, width, height)}

Respond with JSON only, in this format:
{{"moved_objects": [{{"name": "...", "current_location": "...", "current_point": [0, 0], \
"target_location": "...", "target_point": [0, 0]}}]}}
"""


@dataclass(frozen=True)
class PredictedMove:
    name: str
    current_location: str
    current_px: tuple[float, float]
    target_location: str
    target_px: tuple[float, float]


def _extract_json(text: str) -> str:
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("応答に JSON がありません")
    return text[start : end + 1]


def _point(value, label: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2 or not all(isinstance(v, int | float) for v in value):
        raise ValueError(f"{label} は 2 つの数値の配列である必要があります: {value!r}")
    return (float(value[0]), float(value[1]))


def parse_prediction(text: str, *, coord_format: str, width: int, height: int) -> list[PredictedMove]:
    """VLM の応答（JSON）を解析し、座標をピクセルにそろえる。形式が正しくなければ ValueError。"""
    data = json.loads(_extract_json(text))
    if not isinstance(data, dict) or not isinstance(data.get("moved_objects"), list):
        raise ValueError("moved_objects（配列）がありません")
    moves = []
    for item in data["moved_objects"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError(f"name がありません: {item!r}")
        moves.append(
            PredictedMove(
                name=item["name"],
                current_location=str(item.get("current_location", "")),
                current_px=to_pixels(_point(item.get("current_point"), "current_point"), coord_format, width, height),
                target_location=str(item.get("target_location", "")),
                target_px=to_pixels(_point(item.get("target_point"), "target_point"), coord_format, width, height),
            )
        )
    return moves


# ------------------------------------------------------------
# 正解との照合と集計
# ------------------------------------------------------------
@dataclass(frozen=True)
class Match:
    key: str
    current_err_px: float
    target_err_px: float
    target_ok: bool


@dataclass(frozen=True)
class ScenarioScore:
    tp: int
    fp: int
    fn: int
    target_ok: int
    matches: tuple[Match, ...]
    parse_failed: bool = False


def _normalize_name(name: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", name)).lower()


def evaluate(
    preds: Sequence[PredictedMove] | None,
    truth: ScenarioTruth,
    *,
    use_names: bool,
    match_tol_px: float = 40.0,
    target_tol_px: float = 40.0,
) -> ScenarioScore:
    """予測と正解を照合する。preds が None のときは応答の解析に失敗したものとして扱う。

    use_names=True：名前で照合する（物体名のリストを渡した場合）。
    use_names=False：今の画像での位置が最も近い物体と照合する（match_tol_px 以内）。名前の表記の揺れを避けるため。
    同じ物体に 2 回以上照合された予測と、動いていない物体・どの物体にも当たらない予測は誤検出とする。
    """
    if preds is None:
        return ScenarioScore(tp=0, fp=0, fn=len(truth.moved), target_ok=0, matches=(), parse_failed=True)
    assigned: dict[int, str] = {}
    if use_names:
        by_name = {_normalize_name(n): k for k, n in truth.names.items()}
        for i, p in enumerate(preds):
            key = by_name.get(_normalize_name(p.name))
            if key is not None and key not in assigned.values():
                assigned[i] = key
    else:
        pairs = sorted(
            (math.dist(p.current_px, px), i, key)
            for i, p in enumerate(preds)
            for key, px in truth.current_px.items()
            if math.dist(p.current_px, px) <= match_tol_px
        )
        for _, i, key in pairs:
            if i not in assigned and key not in assigned.values():
                assigned[i] = key
    matches = []
    for i, key in assigned.items():
        if key not in truth.moved:
            continue
        p = preds[i]
        target_err = math.dist(p.target_px, truth.target_px[key])
        matches.append(
            Match(
                key=key,
                current_err_px=math.dist(p.current_px, truth.current_px[key]),
                target_err_px=target_err,
                target_ok=target_err <= target_tol_px,
            )
        )
    tp = len(matches)
    return ScenarioScore(
        tp=tp,
        fp=len(preds) - tp,
        fn=len(truth.moved) - tp,
        target_ok=sum(m.target_ok for m in matches),
        matches=tuple(matches),
    )


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def aggregate(scores: Sequence[ScenarioScore]) -> dict:
    """複数のシナリオの結果をまとめる。分母が 0 の率は None。"""
    tp = sum(s.tp for s in scores)
    fp = sum(s.fp for s in scores)
    fn = sum(s.fn for s in scores)
    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else None
    matches = [m for s in scores for m in s.matches]
    return {
        "n": len(scores),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_rate": _ratio(sum(s.fp == 0 and s.fn == 0 and not s.parse_failed for s in scores), len(scores)),
        "parse_failure_rate": _ratio(sum(s.parse_failed for s in scores), len(scores)),
        "target_ok_rate": _ratio(sum(s.target_ok for s in scores), tp),
        "current_err_px_mean": _ratio(sum(m.current_err_px for m in matches), len(matches)),
        "target_err_px_mean": _ratio(sum(m.target_err_px for m in matches), len(matches)),
        "target_err_px_max": max((m.target_err_px for m in matches), default=None),
    }


def cost_usd(input_tokens: int, output_tokens: int, *, price_in_per_mtok: float, price_out_per_mtok: float) -> float:
    return (input_tokens * price_in_per_mtok + output_tokens * price_out_per_mtok) / 1_000_000


def draw_overlay(image, truth: ScenarioTruth, preds: Sequence[PredictedMove] | None = None):
    """確認用の画像（PIL.Image）を作る。

    正解：動いた物体の今の位置（赤い円）と戻し先（緑の円）を線で結ぶ。動いていない物体は白い小さな円。
    予測：今の位置（橙の×）と戻し先（水色の×）を点線風の線で結ぶ。
    """
    from PIL import ImageDraw

    out = image.convert("RGB").copy()
    d = ImageDraw.Draw(out)

    def circle(px, r, color, width=3):
        d.ellipse([px[0] - r, px[1] - r, px[0] + r, px[1] + r], outline=color, width=width)

    def cross(px, r, color):
        d.line([px[0] - r, px[1] - r, px[0] + r, px[1] + r], fill=color, width=3)
        d.line([px[0] - r, px[1] + r, px[0] + r, px[1] - r], fill=color, width=3)

    for key, px in truth.current_px.items():
        if key not in truth.moved:
            circle(px, 6, (255, 255, 255), width=2)
    for key in truth.moved:
        d.line([*truth.current_px[key], *truth.target_px[key]], fill=(255, 60, 60), width=2)
        circle(truth.current_px[key], 14, (255, 40, 40))
        circle(truth.target_px[key], 14, (40, 220, 40))
        d.text((truth.current_px[key][0] + 16, truth.current_px[key][1] - 8), key, fill=(255, 255, 255))
    for p in preds or []:
        d.line([*p.current_px, *p.target_px], fill=(0, 200, 255), width=1)
        cross(p.current_px, 9, (255, 150, 0))
        cross(p.target_px, 9, (0, 200, 255))
    return out


# ------------------------------------------------------------
# VLM のインターフェース
# ------------------------------------------------------------
@dataclass(frozen=True)
class VlmResult:
    text: str
    input_tokens: int
    output_tokens: int  # 思考のトークンを含む（課金される出力の合計）
    latency_s: float


class VlmClient(Protocol):
    name: str
    coord_format: str
    price_in_per_mtok: float
    price_out_per_mtok: float

    def detect(self, before_png: bytes, after_png: bytes, prompt: str) -> VlmResult:
        """1 枚目に「いつもの状態」、2 枚目に「今の状態」の画像を渡して差分を問い合わせる。"""
        ...


class DummyVlmClient:
    """単体テスト用。決まった文字列を返し、呼び出しを記録する。"""

    def __init__(self, text: str, *, name: str = "dummy", coord_format: str = "pixel_xy") -> None:
        self.text = text
        self.name = name
        self.coord_format = coord_format
        self.price_in_per_mtok = 0.0
        self.price_out_per_mtok = 0.0
        self.calls: list[tuple[str, int]] = []

    def detect(self, before_png: bytes, after_png: bytes, prompt: str) -> VlmResult:
        t0 = time.perf_counter()
        self.calls.append((prompt, 2))
        return VlmResult(text=self.text, input_tokens=0, output_tokens=0, latency_s=time.perf_counter() - t0)
