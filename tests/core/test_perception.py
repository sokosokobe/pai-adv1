"""restorebot.core.perception（今の写真から物体の位置と向きを推定する。#23）の単体テスト。

深度画像は tests/core/test_tasks.render_depth（天板と物体の上面から幾何計算で作る）を使う。
"""

import math

import numpy as np
import pytest

from restorebot.core.perception import EstimatedWorld, estimate_object
from restorebot.core.scene import HOME_V2, Placement
from restorebot.core.types import Observation, Pose2D
from restorebot.core.vlm.parse import PredictedMove
from tests.core.test_tasks import render_depth

CAM = HOME_V2.camera


def _obs(placements):
    depth = render_depth(HOME_V2, placements)
    return Observation(np.zeros((CAM.height, CAM.width, 3), np.uint8), depth, 0.0)


def _px(key, p):
    return CAM.project(HOME_V2.object_center(key, p))


def _yaw_err_deg(a, b):
    """180° の違いは同じとみなした角度の差 [deg]。"""
    d = (a - b) % math.pi
    return math.degrees(min(d, math.pi - d))


# ---- 1 個の物体 ----


@pytest.mark.parametrize(
    ("key", "placement"),
    [
        ("cup", Placement("desk", 0.62, 0.66, 0.0)),
        ("pen_stand", Placement("side_table", -1.45, -0.20, 0.0)),
        ("remote", Placement("desk", 0.95, 0.70, 0.7)),
        ("remote", Placement("side_table", -1.47, -0.45, -1.2)),
        ("small_box", Placement("desk", 0.40, 0.62, 2.0)),
        ("small_box", Placement("side_table", -1.42, -0.05, 0.3)),
    ],
)
def test_estimate_position_and_box_yaw(key, placement):
    obs = _obs({key: placement})
    est = estimate_object(key, _px(key, placement), obs, HOME_V2)
    assert est is not None
    assert est.surface == placement.surface
    truth = HOME_V2.object_center(key, placement)
    assert math.dist(est.position, truth) < 0.01
    if HOME_V2.object(key).shape == "cube":
        assert _yaw_err_deg(est.yaw, placement.yaw) < 5.0
    else:
        assert est.yaw is None  # 円柱は向きを推定しない
    assert est.n_pixels > 50


def test_vlm_point_a_few_pixels_off_is_corrected_by_centroid():
    p = Placement("desk", 0.95, 0.70, 0.4)
    u, v = _px("remote", p)
    est = estimate_object("remote", (u + 6, v - 4), _obs({"remote": p}), HOME_V2)
    assert math.dist(est.position[:2], (p.x, p.y)) < 0.01


def test_same_height_neighbor_is_not_merged():
    # ペン立てとスプレー缶は同じ高さ 10cm。外接円どうしの隙間 5cm で隣り合っても、塊は分かれる
    pen = Placement("desk", 0.70, 0.66, 0.0)
    gap = 0.0225 + 0.02 + 0.05
    spray = Placement("desk", 0.70 + gap, 0.66, 0.0)
    est = estimate_object("pen_stand", _px("pen_stand", pen), _obs({"pen_stand": pen, "spray": spray}), HOME_V2)
    assert math.dist(est.position[:2], (pen.x, pen.y)) < 0.01


def test_pointing_at_empty_surface_fails():
    obs = _obs({})
    assert estimate_object("cup", CAM.project((0.8, 0.7, 0.70)), obs, HOME_V2) is None
    assert estimate_object("cup", (-10.0, 5.0), obs, HOME_V2) is None  # 画像の外


def test_occluded_top_face_fails_when_too_few_pixels():
    p = Placement("desk", 0.62, 0.66, 0.0)
    obs = _obs({"cup": p})
    u, v = _px("cup", p)
    depth = obs.depth.copy()
    depth[int(v) - 30 : int(v) + 30, int(u) - 30 : int(u) + 30] = 1.0  # 上面をロボットが隠す
    assert estimate_object("cup", (u, v), Observation(obs.rgb, depth, 0.0), HOME_V2) is None


# ---- 推定版の世界 ----


class Odometry:
    def robot_pose(self):
        return Pose2D(0.0, -0.55, math.pi / 2)


def _pred(key, px):
    return PredictedMove(HOME_V2.object(key).name, "", px, "", (0.0, 0.0))


def test_estimated_world_update_and_fallbacks():
    remote = Placement("desk", 0.95, 0.70, 0.7)
    obs = _obs({"remote": remote})
    world = EstimatedWorld(HOME_V2, Odometry())
    failures = world.update(obs, [_pred("remote", _px("remote", remote)), _pred("cup", CAM.project((0.8, 0.7, 0.7)))])
    s = world.object_state("remote")
    assert math.dist(s.position[:2], (remote.x, remote.y)) < 0.01 and s.surface == "desk"
    assert _yaw_err_deg(s.yaw, remote.yaw) < 5.0
    # 推定に失敗した物体は「位置が分からない」（surface が None）→ タスクを飛ばす
    assert "cup" in failures and world.object_state("cup").surface is None
    # VLM が答えなかった物体は、いつもの位置にあるとみなす
    home = HOME_V2.tidy["spray"]
    assert world.object_state("spray").position == pytest.approx(HOME_V2.object_center("spray", home))
    # 台車の位置はオドメトリ（真値）のまま
    assert world.robot_pose() == Odometry().robot_pose()


def test_estimated_world_note_placed():
    world = EstimatedWorld(HOME_V2, Odometry())
    target = HOME_V2.object_center("remote", HOME_V2.tidy["remote"])
    world.note_placed("remote", target, 0.3)
    s = world.object_state("remote")
    assert s.position == pytest.approx(target) and s.yaw == pytest.approx(0.3) and s.surface == "desk"


def test_estimated_world_names_with_spacing_are_matched():
    remote = Placement("desk", 0.95, 0.70, 0.7)
    world = EstimatedWorld(HOME_V2, Odometry())
    pred = PredictedMove("黒い リモコン", "", _px("remote", remote), "", (0.0, 0.0))
    assert world.update(_obs({"remote": remote}), [pred]) == {}
    assert world.object_state("remote").surface == "desk"
