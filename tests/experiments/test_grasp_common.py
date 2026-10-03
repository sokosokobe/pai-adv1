"""scripts/experiments/grasp_common.py（把持と配置の検証用の判定）の単体テスト。Isaac Sim 不要。"""

import math

import pytest

from experiments.grasp_common import (
    TrialObservation,
    classify_trial,
    sample_initial_offsets,
    tilt_deg_from_quat_wxyz,
)

# ---- 初期位置のずらし方（シード固定）----


def test_offsets_are_reproducible_with_seed():
    assert sample_initial_offsets(10, seed=0) == sample_initial_offsets(10, seed=0)
    assert sample_initial_offsets(10, seed=0) != sample_initial_offsets(10, seed=1)


def test_offsets_stay_within_limits():
    offsets = sample_initial_offsets(200, seed=3, max_offset=0.03, max_yaw_deg=30.0)
    assert len(offsets) == 200
    for dx, dy, dyaw in offsets:
        assert abs(dx) <= 0.03 and abs(dy) <= 0.03
        assert abs(dyaw) <= math.radians(30.0) + 1e-12


def test_first_offset_is_nominal():
    # 1回目は基準位置（ずらさない）で試し、連番画像の代表にする
    assert sample_initial_offsets(5, seed=0)[0] == (0.0, 0.0, 0.0)


# ---- 傾き（物体の z 軸と鉛直の角度）----


def test_tilt_upright_is_zero():
    assert tilt_deg_from_quat_wxyz((1.0, 0.0, 0.0, 0.0)) == pytest.approx(0.0)


def test_tilt_ignores_yaw():
    yaw = 0.7
    assert tilt_deg_from_quat_wxyz((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))) == pytest.approx(0.0, abs=1e-9)


def test_tilt_lying_on_side_is_90():
    a = math.pi / 2  # x 軸まわり 90°
    assert tilt_deg_from_quat_wxyz((math.cos(a / 2), math.sin(a / 2), 0.0, 0.0)) == pytest.approx(90.0)


# ---- 試行結果の分類 ----


def obs(**kwargs) -> TrialObservation:
    base = {
        "controller_done": True,
        "moved_before_grasp_m": 0.0,
        "max_lift_m": 0.25,
        "dropped": False,
        "place_xy_err_m": 0.005,
        "tilt_deg": 1.0,
    }
    base.update(kwargs)
    return TrialObservation(**base)


def test_success():
    assert classify_trial(obs()) == "success"


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"moved_before_grasp_m": 0.03, "max_lift_m": 0.0, "controller_done": False}, "collision_before_grasp"),
        ({"max_lift_m": 0.005, "controller_done": False}, "grasp_miss"),
        ({"dropped": True, "place_xy_err_m": 0.2}, "dropped"),
        ({"tilt_deg": 45.0}, "tipped"),
        ({"place_xy_err_m": 0.05}, "placement_off"),
        ({"controller_done": False}, "controller_failure"),
    ],
)
def test_failure_classes(kwargs, expected):
    assert classify_trial(obs(**kwargs)) == expected


def test_boundaries_are_inclusive():
    assert classify_trial(obs(place_xy_err_m=0.02, tilt_deg=10.0, moved_before_grasp_m=0.01)) == "success"
