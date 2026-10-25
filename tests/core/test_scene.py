"""restorebot.core.scene（場面の定義、作業範囲、散らかし方）の単体テスト。"""

import math

import pytest

from restorebot.core.config import M1_BASE_POSE, PLACE_MARGIN, SCATTER, WORK_DEPTH
from restorebot.core.navigation import in_reach
from restorebot.core.scene import HOME_V2, Surface, generate_scenario

DESK = HOME_V2.surfaces["desk"]
SIDE = HOME_V2.surfaces["side_table"]
SPECS = {o.key: o for o in HOME_V2.objects}


# ---- 作業範囲（手前の縁から WORK_DEPTH 以内、縁から PLACE_MARGIN 以上内側）----


def test_work_depth_is_derived_from_reach_and_gap():
    # 楽に届く上限 0.85m − 台車の中心から家具の縁まで 0.57m
    assert pytest.approx(0.28) == WORK_DEPTH


@pytest.mark.parametrize(
    ("surface", "xy", "expected"),
    [
        (DESK, (0.8, 0.55), 0.0),  # 机の手前（-y）の縁
        (DESK, (0.8, 0.75), 0.20),
        (DESK, (0.8, 1.15), 0.60),  # 奥の縁
        (SIDE, (-1.3, -0.3), 0.0),  # 低い棚の手前（+x）の縁
        (SIDE, (-1.5, -0.3), 0.20),
    ],
)
def test_depth_from_front(surface, xy, expected):
    assert surface.depth_from_front(xy) == pytest.approx(expected)


@pytest.mark.parametrize("front", ["-x", "+x", "-y", "+y"])
def test_depth_from_front_for_each_side(front):
    s = Surface(center=(0.0, 0.0), size=(1.0, 0.6), key="t", z=0.5, front=front)
    edge = {"-x": (-0.5, 0.0), "+x": (0.5, 0.0), "-y": (0.0, -0.3), "+y": (0.0, 0.3)}[front]
    assert s.depth_from_front(edge) == pytest.approx(0.0)
    assert s.depth_from_front((0.0, 0.0)) == pytest.approx(0.5 if front in ("-x", "+x") else 0.3)


def test_invalid_front_is_rejected():
    with pytest.raises(ValueError):
        Surface(center=(0.0, 0.0), size=(1.0, 1.0), front="north")


def test_in_work_area_boundaries():
    r = 0.025
    edge_y = DESK.center[1] - DESK.size[1] / 2
    # 奥行き：中心が手前の縁から WORK_DEPTH まで
    assert DESK.in_work_area((0.8, edge_y + WORK_DEPTH - 1e-6), r)
    assert not DESK.in_work_area((0.8, edge_y + WORK_DEPTH + 1e-3), r)
    # 手前：外形が縁から PLACE_MARGIN 以上内側
    assert DESK.in_work_area((0.8, edge_y + PLACE_MARGIN + r + 1e-6), r)
    assert not DESK.in_work_area((0.8, edge_y + PLACE_MARGIN + r - 1e-3), r)
    # 左右の縁
    left = DESK.center[0] - DESK.size[0] / 2
    assert not DESK.in_work_area((left + r, edge_y + 0.1), r)
    # 天板の外
    assert not DESK.in_work_area((0.0, 0.0), r)


def test_home_layout_is_inside_work_area():
    for key, p in HOME_V2.tidy.items():
        spec = SPECS[key]
        assert HOME_V2.surfaces[p.surface].in_work_area((p.x, p.y), spec.footprint_radius), key


def test_home_layout_keeps_clearance():
    items = list(HOME_V2.tidy.items())
    for i, (ka, pa) in enumerate(items):
        for kb, pb in items[i + 1 :]:
            if pa.surface != pb.surface:
                continue
            gap = math.hypot(pa.x - pb.x, pa.y - pb.y) - SPECS[ka].footprint_radius - SPECS[kb].footprint_radius
            assert gap >= SCATTER.clearance, (ka, kb)


def test_object_center_is_on_surface():
    x, y, z = HOME_V2.object_center("cup", HOME_V2.tidy["cup"])
    assert (x, y) == (0.45, 0.72)
    assert z == pytest.approx(DESK.z + SPECS["cup"].height / 2)


def test_scene_rejects_tidy_on_unknown_surface():
    from dataclasses import replace

    from restorebot.core.scene import Placement

    with pytest.raises(ValueError):
        replace(HOME_V2, tidy={**HOME_V2.tidy, "cup": Placement("floor", 0.0, 0.0)})


# ---- 散らかし方 ----


def _check_scenario(scenario, *, surfaces, reach_from):
    """散らかし方の制約をすべて確かめる。"""
    assert SCATTER.min_moved <= len(scenario.moved) <= SCATTER.max_moved
    assert set(scenario.placements) == set(HOME_V2.tidy)
    for key, p in scenario.placements.items():
        if key not in scenario.moved:
            assert p == HOME_V2.tidy[key], key  # 動かさない物体はいつもの位置のまま
            continue
        spec = SPECS[key]
        surface = HOME_V2.surfaces[p.surface]
        assert p.surface in surfaces
        assert surface.in_work_area((p.x, p.y), spec.footprint_radius), key
        if reach_from is not None:
            assert in_reach(reach_from, (p.x, p.y)), key
            assert in_reach(reach_from, (HOME_V2.tidy[key].x, HOME_V2.tidy[key].y)), key
        # どの物体のいつもの位置からも、隙間 clearance 以上離れている（戻し先がふさがらない）
        for other, home in HOME_V2.tidy.items():
            if home.surface != p.surface:
                continue
            gap = math.hypot(p.x - home.x, p.y - home.y) - spec.footprint_radius - SPECS[other].footprint_radius
            assert gap >= SCATTER.clearance - 1e-9, (key, "home", other)
        # ほかの物体の今の位置からも離れている
        for other, q in scenario.placements.items():
            if other == key or q.surface != p.surface:
                continue
            gap = math.hypot(p.x - q.x, p.y - q.y) - spec.footprint_radius - SPECS[other].footprint_radius
            assert gap >= SCATTER.clearance - 1e-9, (key, "now", other)


def test_same_seed_gives_same_scenario():
    a = generate_scenario(HOME_V2, seed=3)
    b = generate_scenario(HOME_V2, seed=3)
    assert a == b


def test_different_seeds_give_different_scenarios():
    scenarios = {tuple(sorted(generate_scenario(HOME_V2, seed=s).placements.items())) for s in range(10)}
    assert len(scenarios) > 1


@pytest.mark.parametrize("seed", range(100))
def test_scenario_satisfies_constraints_m2(seed):
    scenario = generate_scenario(HOME_V2, seed=seed)
    _check_scenario(scenario, surfaces=set(HOME_V2.surfaces), reach_from=None)


@pytest.mark.parametrize("seed", range(100))
def test_scenario_satisfies_constraints_m1(seed):
    # M1：机の上だけ、M1 の止め位置から届く範囲だけ
    scenario = generate_scenario(HOME_V2, seed=seed, surfaces=("desk",), reach_from=M1_BASE_POSE)
    _check_scenario(scenario, surfaces={"desk"}, reach_from=M1_BASE_POSE)
    assert all(HOME_V2.tidy[k].surface == "desk" for k in scenario.moved)


def test_every_desk_object_can_be_moved_in_m1():
    # 隙間が大きすぎると、M1 の狭い範囲ではリモコンの動かす先がなくなる（#11 で 10cm のときに起きた）
    moved = set()
    for seed in range(100):
        moved.update(generate_scenario(HOME_V2, seed=seed, surfaces=("desk",), reach_from=M1_BASE_POSE).moved)
    assert moved == {k for k, p in HOME_V2.tidy.items() if p.surface == "desk"}


def test_every_object_can_be_moved_in_m2():
    moved = set()
    for seed in range(100):
        moved.update(generate_scenario(HOME_V2, seed=seed).moved)
    assert moved == set(HOME_V2.tidy)


def test_scenario_records_seed():
    assert generate_scenario(HOME_V2, seed=42).seed == 42
