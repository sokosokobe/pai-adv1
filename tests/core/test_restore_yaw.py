"""箱の向きをいつもの向きに戻す（#25）の単体テスト：戻し先の向きの推定、place_yaw の決め方、置くときの手先の向き。"""

import math

import numpy as np
import pytest

from restorebot.core.grasp import box_place_hand_yaw
from restorebot.core.navigation import wrap_angle
from restorebot.core.scene import HOME_V2, Placement
from restorebot.core.tasks import plan_tasks, to_instructions
from restorebot.core.types import MoveInstruction, ObjectState, Observation, Pose2D
from restorebot.core.vlm.parse import PredictedMove
from tests.core.test_tasks import FakeWorld, render_depth

CAM = HOME_V2.camera


def _yaw_err_deg(a, b):
    d = (a - b) % math.pi
    return math.degrees(min(d, math.pi - d))


def _reference(placements):
    return Observation(np.zeros((CAM.height, CAM.width, 3), np.uint8), render_depth(HOME_V2, placements), 0.0)


def _pred(key, placement):
    return PredictedMove(
        HOME_V2.object(key).name, "", (10.0, 10.0), "", CAM.project(HOME_V2.object_center(key, placement))
    )


# ---- 戻し先の向き ----


@pytest.mark.parametrize("home_yaw", [0.0, 0.5, -1.2, 1.5])
def test_target_yaw_of_box_is_estimated_from_reference(home_yaw):
    home = Placement("desk", 1.05, 0.72, home_yaw)
    (ins,), dropped = to_instructions([_pred("remote", home)], HOME_V2, _reference({"remote": home}))
    assert dropped == []
    assert ins.target_yaw is not None and _yaw_err_deg(ins.target_yaw, home_yaw) < 3.0


def test_cylinder_has_no_target_yaw():
    home = HOME_V2.tidy["cup"]
    (ins,), _ = to_instructions([_pred("cup", home)], HOME_V2, _reference({"cup": home}))
    assert ins.target_yaw is None


def test_target_yaw_none_when_box_not_visible_in_reference():
    # いつもの写真に箱が写っていない（深度に上面がない）→ 向きは分からない。位置は今までどおり求める
    home = HOME_V2.tidy["remote"]
    (ins,), _ = to_instructions([_pred("remote", home)], HOME_V2, _reference({}))
    assert ins.target_yaw is None
    assert ins.target_position[:2] == pytest.approx((home.x, home.y), abs=0.01)


# ---- place_yaw の決め方 ----


def _state(key, xy, yaw):
    p = Placement(HOME_V2.tidy[key].surface, *xy)
    return ObjectState(key, HOME_V2.object_center(key, p), yaw, p.surface)


def _ins(key, target_yaw):
    p = HOME_V2.tidy[key]
    return MoveInstruction(key, (0.0, 0.0), (0.0, 0.0), HOME_V2.object_center(key, p), p.surface, target_yaw)


def test_box_place_yaw_is_target_yaw():
    world = FakeWorld({"remote": _state("remote", (0.9, 0.62), 0.7)})
    (task,) = plan_tasks([_ins("remote", 0.1)], world, HOME_V2, Pose2D(0.75, -0.02, math.pi / 2))
    assert task.place_yaw == pytest.approx(0.1)


def test_cylinder_and_unknown_target_keep_object_yaw():
    world = FakeWorld({"cup": _state("cup", (0.6, 0.62), 0.4), "remote": _state("remote", (0.9, 0.62), 0.7)})
    tasks = plan_tasks([_ins("cup", None), _ins("remote", None)], world, HOME_V2, Pose2D(0.75, -0.02, math.pi / 2))
    assert {t.key: t.place_yaw for t in tasks} == pytest.approx({"cup": 0.4, "remote": 0.7})


# ---- 置くときの手先の向き ----


@pytest.mark.parametrize("delta", [0.0, 0.3, -0.8, 1.4, 2.5])
def test_hand_rotates_by_object_yaw_difference(delta):
    grasp_hand, grasp_obj = 1.2, 0.5
    ref = 1.5
    hand = box_place_hand_yaw(grasp_hand, grasp_obj, grasp_obj + delta, ref)
    # 物体は手と一緒に回るので、手先を delta 回せば物体も delta 回る（箱は 180° の違いを同じとみなす）
    assert _yaw_err_deg(hand - grasp_hand, delta) < 1e-6
    assert abs(wrap_angle(hand - ref)) <= math.pi / 2 + 1e-9  # 手首を回しすぎない


def test_hand_picks_the_equivalent_closest_to_reference():
    # 180° 違う 2 通りのうち、台車から置く位置を見た方向に近い方
    hand = box_place_hand_yaw(0.0, 0.0, math.pi / 2 + 0.1, math.pi / 2)
    assert hand == pytest.approx(math.pi / 2 + 0.1)
    hand = box_place_hand_yaw(0.0, 0.0, math.pi / 2 + 0.1, -math.pi / 2)
    assert hand == pytest.approx(-math.pi / 2 + 0.1)
