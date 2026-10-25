"""restorebot.core.tasks（VLM の予測 → 目標 → 運ぶタスクの列）の単体テスト。

いつもの写真の深度は、HOME_V2 の天板と物体（いつもの配置）から幾何計算で作る（Isaac Sim 不要）。
"""

import math

import numpy as np
import pytest

from restorebot.core.scene import HOME_V2, Placement, SceneSpec
from restorebot.core.tasks import plan_tasks, to_instructions
from restorebot.core.types import MoveInstruction, ObjectState, Observation, Pose2D
from restorebot.core.vlm.parse import PredictedMove

CAM = HOME_V2.camera
NAMES = {o.key: o.name for o in HOME_V2.objects}


def render_depth(scene: SceneSpec, placements: dict[str, Placement]) -> np.ndarray:
    """真下向きのカメラの distance_to_image_plane を、床・天板・物体の上面から作る（低い面から順に上書き）。"""
    cam = scene.camera
    cz = cam.position[2]
    v, u = np.mgrid[0 : cam.height, 0 : cam.width].astype(np.float64) + 0.5  # 画素の中心

    def plane_xy(z):
        d = cz - z
        return cam.position[0] + (u - cam.cx) * d / cam.fx, cam.position[1] - (v - cam.cy) * d / cam.fy

    depth = np.full((cam.height, cam.width), cz, dtype=np.float32)  # 床（z = 0）
    for s in sorted(scene.surfaces.values(), key=lambda s: s.z):
        x, y = plane_xy(s.z)
        inside = (np.abs(x - s.center[0]) <= s.size[0] / 2) & (np.abs(y - s.center[1]) <= s.size[1] / 2)
        depth[inside] = cz - s.z
    for key, p in placements.items():
        spec = scene.object(key)
        top = scene.surfaces[p.surface].z + spec.height
        x, y = plane_xy(top)
        dx, dy = x - p.x, y - p.y
        if spec.shape == "cube":
            c, s_ = math.cos(p.yaw), math.sin(p.yaw)
            lx, ly = c * dx + s_ * dy, -s_ * dx + c * dy
            inside = (np.abs(lx) <= spec.size[0] / 2) & (np.abs(ly) <= spec.size[1] / 2)
        else:
            inside = dx * dx + dy * dy <= spec.size[0] ** 2
        depth[inside] = cz - top
    return depth


REFERENCE = Observation(
    rgb=np.zeros((CAM.height, CAM.width, 3), np.uint8), depth=render_depth(HOME_V2, HOME_V2.tidy), time_s=0.0
)


def _center_px(key: str, placement: Placement | None = None) -> tuple[float, float]:
    """物体の中心（高さの半分）を投影した画素。VLM が答える「物体の中心」はこの点に当たる（#9 の正解、#12 で確認）。"""
    return CAM.project(HOME_V2.object_center(key, placement or HOME_V2.tidy[key]))


def _top_px(key: str) -> tuple[float, float]:
    """物体の上面の中心を投影した画素（物体の中心の投影とは数 px〜10px ずれる）。"""
    x, y, z = HOME_V2.object_center(key, HOME_V2.tidy[key])
    return CAM.project((x, y, z + HOME_V2.object(key).height / 2))


def _pred(key: str, target_px, current_px=(10.0, 10.0), name: str | None = None) -> PredictedMove:
    return PredictedMove(name or NAMES[key], "今", current_px, "戻し先", target_px)


# ---- to_instructions ----


@pytest.mark.parametrize("key", list(HOME_V2.tidy))
def test_target_position_recovered_from_reference_depth(key):
    (ins,), dropped = to_instructions([_pred(key, _center_px(key))], HOME_V2, REFERENCE)
    assert dropped == []
    assert ins.key == key and ins.target_surface == HOME_V2.tidy[key].surface
    assert ins.target_position == pytest.approx(HOME_V2.object_center(key, HOME_V2.tidy[key]), abs=1e-3)


@pytest.mark.parametrize("key", list(HOME_V2.tidy))
def test_depth_is_used_only_to_find_the_surface(key):
    # 深度の画素に物体が写っていなくても（天板だけ）、物体の中心の高さで逆投影するので位置は同じ
    empty = Observation(rgb=REFERENCE.rgb, depth=render_depth(HOME_V2, {}), time_s=0.0)
    (ins,), _ = to_instructions([_pred(key, _center_px(key))], HOME_V2, empty)
    assert ins.target_position == pytest.approx(HOME_V2.object_center(key, HOME_V2.tidy[key]), abs=1e-3)


def test_top_center_pixel_would_be_off_by_centimeters():
    # 物体の上面の中心の投影は、物体の中心の投影と数 px ずれる（#12 で見つけた 1.5〜1.8cm の誤差の原因）
    assert math.dist(_top_px("spray"), _center_px("spray")) > 5.0


def test_target_two_pixels_off_stays_within_one_centimeter():
    u, v = _center_px("pen_stand")
    (ins,), _ = to_instructions([_pred("pen_stand", (u + 2, v - 2))], HOME_V2, REFERENCE)
    home = HOME_V2.tidy["pen_stand"]
    assert ins.target_position[:2] == pytest.approx((home.x, home.y), abs=0.01)


def test_name_matching_ignores_spacing_and_width():
    (ins,), _ = to_instructions([_pred("cup", _center_px("cup"), name="赤い　コップ")], HOME_V2, REFERENCE)
    assert ins.key == "cup"


def test_unknown_and_duplicate_names_are_dropped():
    preds = [
        _pred("cup", _center_px("cup")),
        _pred("cup", _center_px("cup")),  # 2 回目の同じ名前
        _pred("cup", _center_px("cup"), name="緑のボール"),  # 一覧にない名前
    ]
    instructions, dropped = to_instructions(preds, HOME_V2, REFERENCE)
    assert [i.key for i in instructions] == ["cup"]
    assert len(dropped) == 2
    assert any("緑のボール" in d for d in dropped)


@pytest.mark.parametrize(
    ("target_xyz", "reason"),
    [
        ((0.0, 0.0, 0.0), "天板"),  # 床
        ((0.8, 1.05, 0.70), "作業範囲"),  # 机の奥（手前の縁から 50cm）
    ],
)
def test_targets_off_surface_or_outside_work_area_are_dropped(target_xyz, reason):
    instructions, dropped = to_instructions([_pred("cup", CAM.project(target_xyz))], HOME_V2, REFERENCE)
    assert instructions == []
    assert len(dropped) == 1 and reason in dropped[0]


def test_target_outside_image_is_dropped():
    instructions, dropped = to_instructions([_pred("cup", (-5.0, 100.0))], HOME_V2, REFERENCE)
    assert instructions == [] and "画像" in dropped[0]


def test_invalid_depth_is_dropped():
    depth = REFERENCE.depth.copy()
    u, v = _center_px("cup")
    depth[int(v) - 5 : int(v) + 6, int(u) - 5 : int(u) + 6] = np.nan
    ref = Observation(rgb=REFERENCE.rgb, depth=depth, time_s=0.0)
    instructions, dropped = to_instructions([_pred("cup", (u, v))], HOME_V2, ref)
    assert instructions == [] and "深度" in dropped[0]


def test_instruction_keeps_pixels_for_the_log():
    (ins,), _ = to_instructions([_pred("cup", _center_px("cup"), current_px=(900.0, 300.0))], HOME_V2, REFERENCE)
    assert ins.current_px == (900.0, 300.0)
    assert ins.target_px == pytest.approx(_center_px("cup"))


# ---- plan_tasks ----


class FakeWorld:
    def __init__(self, states: dict[str, ObjectState]):
        self.states = states

    def object_state(self, key: str) -> ObjectState:
        return self.states[key]

    def robot_pose(self) -> Pose2D:
        return Pose2D(0.0, 0.0, 0.0)


def _ins(key: str, xy) -> MoveInstruction:
    p = Placement(HOME_V2.tidy[key].surface, *xy)
    return MoveInstruction(key, (0.0, 0.0), (0.0, 0.0), HOME_V2.object_center(key, p), p.surface)


def _state(key: str, xy, yaw=0.0) -> ObjectState:
    p = Placement(HOME_V2.tidy[key].surface, *xy)
    return ObjectState(key, HOME_V2.object_center(key, p), yaw, p.surface)


def test_plan_tasks_orders_by_nearest_from_current_position():
    world = FakeWorld(
        {
            "cup": _state("cup", (0.40, 0.60)),
            "pen_stand": _state("pen_stand", (1.10, 0.60)),
            "spray": _state("spray", (-1.45, -0.60)),
        }
    )
    instructions = [
        _ins("spray", (-1.5, -0.10)),
        _ins("pen_stand", (0.75, 0.75)),
        _ins("cup", (0.45, 0.72)),
    ]
    # 机の右の前にいる → ペン立て → （置いた位置 (0.75, 0.75) から）コップ → スプレー缶
    tasks = plan_tasks(instructions, FakeWorld(world.states), HOME_V2, Pose2D(1.1, 0.0, math.pi / 2))
    assert [t.key for t in tasks] == ["pen_stand", "cup", "spray"]


def test_plan_tasks_fills_pick_state_and_keeps_object_yaw():
    world = FakeWorld({"remote": _state("remote", (0.9, 0.62), yaw=0.7)})
    (task,) = plan_tasks([_ins("remote", (1.05, 0.72))], world, HOME_V2, Pose2D(0.75, -0.02, math.pi / 2))
    assert task.pick == world.states["remote"]
    assert task.place_position == pytest.approx(HOME_V2.object_center("remote", HOME_V2.tidy["remote"]))
    assert task.place_yaw == pytest.approx(0.7)  # 物体の向きは戻さない（掴んだときのまま）
    assert task.place_surface == "desk"


def test_plan_tasks_empty():
    assert plan_tasks([], FakeWorld({}), HOME_V2, Pose2D(0.0, 0.0, 0.0)) == []
