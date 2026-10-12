"""scripts/experiments/vlm_common.py（VLM による差分検出の検証用ロジック）の単体テスト。Isaac Sim・API 不要。"""

import json
import math

import pytest

from experiments.vlm_common import (
    CEILING_CAMERA,
    HOME_V2,
    OBJECTS,
    ROBOT_NOTE,
    ROOM_HALF,
    ROOM_V1,
    SURFACES,
    TIDY_LAYOUT,
    DownCamera,
    DummyVlmClient,
    PredictedMove,
    Rect,
    aggregate,
    build_prompt,
    cost_usd,
    draw_overlay,
    evaluate,
    generate_scenarios,
    object_center,
    parse_prediction,
    rects_overlap,
    robot_faces,
    robot_rect,
    sample_robot_poses,
    scenario_truth,
    to_pixels,
)

CAM = DownCamera(position=(0.0, 0.0, 2.5), width=1024, height=768, focal_length=18.0, horizontal_aperture=36.0)


# ---- カメラの投影（真下向き、画像の右 = ワールド +x、画像の上 = ワールド +y）----


def test_camera_intrinsics_from_focal_length_and_aperture():
    assert CAM.fx == pytest.approx(18.0 / 36.0 * 1024)
    assert CAM.fy == pytest.approx(CAM.fx)  # 正方画素
    assert (CAM.cx, CAM.cy) == (512.0, 384.0)


def test_point_below_camera_projects_to_image_center():
    assert CAM.project((0.0, 0.0, 0.0)) == pytest.approx((512.0, 384.0))


def test_projection_axes_follow_world_axes():
    u, v = CAM.project((1.0, 0.0, 0.0))
    assert u > 512.0 and v == pytest.approx(384.0)  # +x は右
    u, v = CAM.project((0.0, 1.0, 0.0))
    assert u == pytest.approx(512.0) and v < 384.0  # +y は上（v が小さい）
    # 1m 先、高さ 0 → 距離 2.5m: u = cx + fx * 1 / 2.5
    assert CAM.project((1.0, 0.0, 0.0))[0] == pytest.approx(512.0 + CAM.fx / 2.5)


def test_higher_points_project_farther_from_center():
    # 机の上（カメラに近い）の点は、同じ水平位置でも画像の端寄りに写る
    u_floor, _ = CAM.project((1.0, 0.0, 0.0))
    u_desk, _ = CAM.project((1.0, 0.0, 0.7))
    assert u_desk > u_floor


def test_meters_per_pixel():
    assert CAM.meters_per_pixel(0.0) == pytest.approx(2.5 / CAM.fx)


def test_ceiling_camera_sees_whole_floor_and_all_placements():
    cam = CEILING_CAMERA
    hx, hy = ROOM_HALF
    for corner in [(-hx, -hy, 0.0), (hx, hy, 0.0), (-hx, hy, 0.0), (hx, -hy, 0.0)]:
        u, v = cam.project(corner)
        assert 0 <= u <= cam.width and 0 <= v <= cam.height
    specs = {o.key: o for o in OBJECTS}
    for sc in generate_scenarios(50, seed=6):
        for key, p in sc.placements.items():
            u, v = cam.project(object_center(specs[key], p))
            assert 0 <= u <= cam.width and 0 <= v <= cam.height


# ---- 座標の表し方の変換 ----


def test_to_pixels_from_each_format():
    assert to_pixels((100.0, 200.0), "pixel_xy", 1024, 768) == pytest.approx((100.0, 200.0))
    assert to_pixels((500.0, 250.0), "norm1000_xy", 1024, 768) == pytest.approx((512.0, 192.0))
    # Gemini Robotics-ER は [y, x] の順
    assert to_pixels((250.0, 500.0), "norm1000_yx", 1024, 768) == pytest.approx((512.0, 192.0))


def test_to_pixels_rejects_unknown_format():
    with pytest.raises(ValueError):
        to_pixels((0.0, 0.0), "norm1_xy", 1024, 768)


# ---- 物体と置き場所 ----


def test_tidy_layout_places_every_object_inside_its_surface():
    assert set(TIDY_LAYOUT) == {o.key for o in OBJECTS}
    for obj in OBJECTS:
        placement = TIDY_LAYOUT[obj.key]
        surface = SURFACES[placement.surface]
        assert surface.contains((placement.x, placement.y), obj.footprint_radius)


def test_object_center_sits_on_surface():
    obj = OBJECTS[0]
    placement = TIDY_LAYOUT[obj.key]
    x, y, z = object_center(obj, placement)
    assert (x, y) == (placement.x, placement.y)
    assert z == pytest.approx(SURFACES[placement.surface].z + obj.height / 2)


# ---- 散らかった状態の生成（シード固定）----


def test_scenarios_are_reproducible_with_seed():
    assert generate_scenarios(10, seed=0) == generate_scenarios(10, seed=0)
    assert generate_scenarios(10, seed=0) != generate_scenarios(10, seed=1)


def test_scenarios_move_one_to_three_objects_far_enough():
    specs = {o.key: o for o in OBJECTS}
    for sc in generate_scenarios(50, seed=3, min_shift=0.4):
        assert 1 <= len(sc.moved) <= 3
        for key, placement in sc.placements.items():
            tidy = TIDY_LAYOUT[key]
            shift = math.hypot(placement.x - tidy.x, placement.y - tidy.y)
            if key in sc.moved:
                assert shift >= 0.4 or placement.surface != tidy.surface
            else:
                assert placement == tidy
            assert SURFACES[placement.surface].contains((placement.x, placement.y), specs[key].footprint_radius)


def test_small_move_scenarios_stay_on_same_surface_within_shift_range():
    specs = {o.key: o for o in OBJECTS}
    scenarios = generate_scenarios(50, seed=7, min_shift=0.10, max_shift=0.25, same_surface=True)
    for sc in scenarios:
        for key in sc.moved:
            tidy, p = TIDY_LAYOUT[key], sc.placements[key]
            assert p.surface == tidy.surface
            assert 0.10 <= math.hypot(p.x - tidy.x, p.y - tidy.y) <= 0.25
            assert SURFACES[p.surface].contains((p.x, p.y), specs[key].footprint_radius)
    assert scenarios != generate_scenarios(50, seed=7)


def test_scenarios_have_no_overlapping_objects():
    specs = {o.key: o for o in OBJECTS}
    for sc in generate_scenarios(50, seed=4):
        items = list(sc.placements.items())
        for i, (ka, pa) in enumerate(items):
            for kb, pb in items[i + 1 :]:
                if pa.surface != pb.surface:
                    continue
                gap = math.hypot(pa.x - pb.x, pa.y - pb.y)
                assert gap >= specs[ka].footprint_radius + specs[kb].footprint_radius


def test_floor_placements_avoid_furniture():
    specs = {o.key: o for o in OBJECTS}
    furniture = [s for s in SURFACES.values() if s.key != "floor"]
    for sc in generate_scenarios(50, seed=5):
        for key, p in sc.placements.items():
            if p.surface != "floor":
                continue
            for f in furniture:
                assert not f.overlaps_circle((p.x, p.y), specs[key].footprint_radius)


# ---- 正解データ ----


def test_scenario_truth_projects_current_and_target_positions():
    sc = generate_scenarios(1, seed=0)[0]
    truth = scenario_truth(sc, CAM)
    specs = {o.key: o for o in OBJECTS}
    assert set(truth.moved) == set(sc.moved)
    for key in sc.moved:
        cur = CAM.project(object_center(specs[key], sc.placements[key]))
        tgt = CAM.project(object_center(specs[key], TIDY_LAYOUT[key]))
        assert truth.current_px[key] == pytest.approx(cur)
        assert truth.target_px[key] == pytest.approx(tgt)
    assert set(truth.current_px) == {o.key for o in OBJECTS}


# ---- プロンプト ----


def test_prompt_mentions_names_only_when_given():
    names = [o.name for o in OBJECTS]
    with_names = build_prompt(coord_format="pixel_xy", width=1024, height=768, object_names=names)
    without = build_prompt(coord_format="pixel_xy", width=1024, height=768, object_names=None)
    assert all(n in with_names for n in names)
    assert not any(n in without for n in names)


def test_prompt_describes_coordinate_format():
    assert "1024" in build_prompt(coord_format="pixel_xy", width=1024, height=768, object_names=None)
    assert "[y, x]" in build_prompt(coord_format="norm1000_yx", width=1024, height=768, object_names=None)
    assert "[x, y]" in build_prompt(coord_format="norm1000_xy", width=1024, height=768, object_names=None)


# ---- 応答の解析 ----


def _response(moves) -> str:
    return json.dumps({"moved_objects": moves}, ensure_ascii=False)


def _move(name, cur, tgt) -> dict:
    return {
        "name": name,
        "current_location": "床",
        "current_point": list(cur),
        "target_location": "机の上",
        "target_point": list(tgt),
    }


def test_parse_prediction_converts_points_to_pixels():
    text = _response([_move("赤いマグカップ", (250, 500), (500, 250))])
    (move,) = parse_prediction(text, coord_format="norm1000_yx", width=1024, height=768)
    assert move.name == "赤いマグカップ"
    assert move.current_px == pytest.approx((512.0, 192.0))
    assert move.target_px == pytest.approx((256.0, 384.0))


def test_parse_prediction_accepts_code_fence_and_surrounding_text():
    text = "結果です。\n```json\n" + _response([_move("a", (1, 2), (3, 4))]) + "\n```\n以上"
    assert len(parse_prediction(text, coord_format="pixel_xy", width=1024, height=768)) == 1


def test_parse_prediction_accepts_empty_list():
    assert parse_prediction(_response([]), coord_format="pixel_xy", width=1024, height=768) == []


@pytest.mark.parametrize(
    "text",
    [
        "動いた物体はありません",  # JSON がない
        '{"moved_objects": [{"name": "a"}]}',  # 必須の項目がない
        '{"moved_objects": [{"name": "a", "current_point": [1], "target_point": [1, 2]}]}',  # 座標の要素数
        '{"objects": []}',  # キーが違う
        '{"moved_objects": [',  # 壊れている
    ],
)
def test_parse_prediction_rejects_invalid_responses(text):
    with pytest.raises(ValueError):
        parse_prediction(text, coord_format="pixel_xy", width=1024, height=768)


# ---- 評価 ----


def _truth():
    sc = generate_scenarios(1, seed=0)[0]
    return sc, scenario_truth(sc, CAM)


def _pred_for(truth, key, *, name=None, cur_offset=(0.0, 0.0), tgt_offset=(0.0, 0.0)) -> PredictedMove:
    cu, cv = truth.current_px[key]
    tu, tv = truth.target_px.get(key, truth.current_px[key])
    return PredictedMove(
        name=name if name is not None else truth.names[key],
        current_location="",
        current_px=(cu + cur_offset[0], cv + cur_offset[1]),
        target_location="",
        target_px=(tu + tgt_offset[0], tv + tgt_offset[1]),
    )


@pytest.mark.parametrize("use_names", [True, False])
def test_evaluate_perfect_prediction(use_names):
    sc, truth = _truth()
    preds = [_pred_for(truth, k) for k in sc.moved]
    s = evaluate(preds, truth, use_names=use_names)
    assert (s.tp, s.fp, s.fn) == (len(sc.moved), 0, 0)
    assert s.target_ok == len(sc.moved)
    assert all(m.current_err_px == pytest.approx(0.0) for m in s.matches)


@pytest.mark.parametrize("use_names", [True, False])
def test_evaluate_counts_miss_and_false_positive(use_names):
    sc, truth = _truth()
    still = next(o.key for o in OBJECTS if o.key not in sc.moved)
    preds = [_pred_for(truth, k) for k in sc.moved[1:]] + [_pred_for(truth, still)]
    s = evaluate(preds, truth, use_names=use_names)
    assert (s.tp, s.fp, s.fn) == (len(sc.moved) - 1, 1, 1)


def test_evaluate_by_name_ignores_spacing_and_case():
    sc, truth = _truth()
    key = sc.moved[0]
    preds = [_pred_for(truth, key, name=" " + truth.names[key].upper() + " ")]
    assert evaluate(preds, truth, use_names=True).tp == 1


def test_evaluate_without_names_matches_by_position():
    sc, truth = _truth()
    key = sc.moved[0]
    near = _pred_for(truth, key, name="なにか", cur_offset=(10.0, 0.0))
    s = evaluate([near], truth, use_names=False, match_tol_px=40.0)
    assert s.tp == 1 and s.matches[0].current_err_px == pytest.approx(10.0)
    far = _pred_for(truth, key, name="なにか", cur_offset=(500.0, 500.0))
    s = evaluate([far], truth, use_names=False, match_tol_px=40.0)
    assert (s.tp, s.fp) == (0, 1)


def test_evaluate_duplicate_prediction_is_false_positive():
    sc, truth = _truth()
    key = sc.moved[0]
    preds = [_pred_for(truth, key), _pred_for(truth, key)]
    for use_names in (True, False):
        s = evaluate(preds, truth, use_names=use_names)
        assert (s.tp, s.fp) == (1, 1)


def test_evaluate_target_tolerance():
    sc, truth = _truth()
    key = sc.moved[0]
    off = _pred_for(truth, key, tgt_offset=(60.0, 80.0))
    s = evaluate([off], truth, use_names=True, target_tol_px=50.0)
    assert s.tp == 1 and s.target_ok == 0
    assert s.matches[0].target_err_px == pytest.approx(100.0)


def test_evaluate_failed_response_misses_everything():
    sc, truth = _truth()
    s = evaluate(None, truth, use_names=True)
    assert (s.tp, s.fp, s.fn) == (0, 0, len(sc.moved)) and s.parse_failed


def test_aggregate_rates():
    sc, truth = _truth()
    perfect = evaluate([_pred_for(truth, k) for k in sc.moved], truth, use_names=True)
    failed = evaluate(None, truth, use_names=True)
    agg = aggregate([perfect, failed])
    n = len(sc.moved)
    assert agg["precision"] == pytest.approx(1.0)
    assert agg["recall"] == pytest.approx(n / (2 * n))
    assert agg["exact_match_rate"] == pytest.approx(0.5)
    assert agg["parse_failure_rate"] == pytest.approx(0.5)
    assert agg["target_ok_rate"] == pytest.approx(1.0)


def test_aggregate_handles_no_predictions():
    _, truth = _truth()
    agg = aggregate([evaluate([], truth, use_names=True)])
    assert agg["precision"] is None and agg["recall"] == 0.0


# ---- 費用・ダミーの VLM ----


def test_cost_usd():
    assert cost_usd(3000, 1500, price_in_per_mtok=2.0, price_out_per_mtok=10.0) == pytest.approx(0.021)


def test_draw_overlay_keeps_image_size():
    from PIL import Image

    sc, truth = _truth()
    img = Image.new("RGB", (1024, 768))
    out = draw_overlay(img, truth, [_pred_for(truth, sc.moved[0])])
    assert out.size == (1024, 768)
    assert out.getpixel((0, 0)) == (0, 0, 0) and out.getbbox() is not None


def test_dummy_client_returns_canned_text():
    client = DummyVlmClient(text=_response([]))
    result = client.detect(b"before", b"after", "prompt")
    assert result.text == _response([]) and result.latency_s >= 0.0
    assert client.calls == [("prompt", 2)]


# ---- 場面の定義（#9：自宅風の部屋と、掴める物体の組）----


def test_room_v1_scene_matches_module_constants():
    # #5 の場面は、これまでの定数と同じ内容（#5 の結果を再現できる）
    assert ROOM_V1.objects == OBJECTS
    assert ROOM_V1.tidy == TIDY_LAYOUT
    assert ROOM_V1.surfaces == SURFACES
    assert ROOM_V1.camera == CEILING_CAMERA
    assert generate_scenarios(10, seed=0) == generate_scenarios(10, seed=0, scene=ROOM_V1)


def test_home_v2_objects_fit_grasp_range_of_issue4():
    # #4 で確認した把持の範囲：掴む幅 4〜5cm、高さ 10cm 以下、0.5kg 以下
    assert len(HOME_V2.objects) == 5
    for o in HOME_V2.objects:
        assert 0.04 <= o.grasp_width <= 0.05, o.key
        assert o.height <= 0.10, o.key
        assert 0 < o.mass <= 0.5, o.key
    assert len({o.color for o in HOME_V2.objects}) == 5  # 色がすべて違う


def test_home_v2_surfaces_are_within_reach_height_and_not_floor():
    # #3 の楽に届く範囲：手先の高さ 0.4〜0.9m（物体の中心の高さで判定）
    assert "floor" not in HOME_V2.surfaces
    for o in HOME_V2.objects:
        for s in HOME_V2.surfaces.values():
            assert 0.4 <= s.z + o.height / 2 <= 0.9, (o.key, s.key)


def test_home_v2_tidy_layout_and_scenarios_are_valid_and_visible():
    scene = HOME_V2
    specs = {o.key: o for o in scene.objects}
    assert set(scene.tidy) == set(specs)
    for sc in generate_scenarios(50, seed=8, min_shift=0.3, scene=scene):
        assert 1 <= len(sc.moved) <= 3
        for key, p in sc.placements.items():
            assert p.surface in scene.surfaces
            assert scene.surfaces[p.surface].contains((p.x, p.y), specs[key].footprint_radius)
            u, v = scene.camera.project(object_center(specs[key], p, scene=scene))
            assert 0 <= u <= scene.camera.width and 0 <= v <= scene.camera.height


def test_home_v2_scenario_truth_uses_scene_objects():
    sc = generate_scenarios(1, seed=0, min_shift=0.3, scene=HOME_V2)[0]
    truth = scenario_truth(sc, HOME_V2.camera, scene=HOME_V2)
    assert set(truth.names) == {o.key for o in HOME_V2.objects}
    assert set(truth.target_px) == set(sc.moved)


# ---- ロボットの置き方（今の画像にだけ写す。#9 のリスク #3 の確認）----


def test_robot_poses_face_furniture_and_stay_in_room_without_overlap():
    scene = HOME_V2
    furniture = [Rect(s.center, s.size) for s in scene.surfaces.values()]
    poses = sample_robot_poses(30, seed=0, scene=scene)
    assert poses == sample_robot_poses(30, seed=0, scene=scene)
    for pose in poses:
        r = robot_rect(pose)
        hx, hy = scene.room_half
        assert abs(r.center[0]) + r.size[0] / 2 <= hx and abs(r.center[1]) + r.size[1] / 2 <= hy
        for f in furniture:
            assert not rects_overlap(r, f)
        # どれかの家具の正面にいて、家具の方を向いている
        assert any(robot_faces(pose, f) for f in furniture)


def test_robot_rect_rotates_with_yaw():
    r0 = robot_rect((0.0, 0.0, 0.0))
    r90 = robot_rect((0.0, 0.0, math.pi / 2))
    assert r0.size == pytest.approx((r90.size[1], r90.size[0]))
    assert r0.size[0] > r0.size[1]  # 台車は前後に長い


def test_rects_overlap():
    a = Rect(center=(0.0, 0.0), size=(1.0, 1.0))
    assert rects_overlap(a, Rect(center=(0.9, 0.0), size=(1.0, 1.0)))
    assert not rects_overlap(a, Rect(center=(1.1, 0.0), size=(1.0, 1.0)))


def test_prompt_robot_note_is_optional_and_does_not_change_default():
    kwargs = {"coord_format": "pixel_xy", "width": 1280, "height": 960, "object_names": None}
    default = build_prompt(**kwargs)
    with_note = build_prompt(**kwargs, robot_note=True)
    assert "robot" not in default.lower()
    assert "robot" in with_note.lower()
    assert with_note.replace(ROBOT_NOTE + "\n", "") == default
