"""restorebot.core.grasp（掴む・置くの段階の列、段階を進める判定、手先の向き、成功の判定）の単体テスト。"""

import math

import pytest

from restorebot.core.grasp import (
    GRASP_OFFSET_Z,
    GraspParams,
    Gripper,
    StageSequencer,
    TrialObservation,
    classify_trial,
    grasp_yaw,
    pick_stages,
    place_stages,
    standby_grasp_position,
)
from restorebot.core.navigation import wrap_angle
from restorebot.core.types import Pose2D, Status

BASE = Pose2D(0.75, -0.02, math.pi / 2)
PARAMS = GraspParams()


# ---- 手先の向き ----


@pytest.mark.parametrize("object_yaw", [0.0, 0.5, 1.5, 2.5, -2.5, math.pi, -math.pi / 2])
def test_box_grasp_yaw_matches_object_axis_and_stays_near_base(object_yaw):
    yaw = grasp_yaw("cube", object_yaw, BASE.yaw)
    # 箱は物体の軸に合わせる（180° 回しても同じ掴み方）
    assert math.sin(yaw - object_yaw) == pytest.approx(0.0, abs=1e-9)
    # 台車の向きから ±90° 以内（手首を大きく回さない）
    assert abs(wrap_angle(yaw - BASE.yaw)) <= math.pi / 2 + 1e-9


@pytest.mark.parametrize("object_yaw", [0.0, 1.0, -2.0])
def test_cylinder_grasp_yaw_follows_base(object_yaw):
    assert grasp_yaw("cylinder", object_yaw, BASE.yaw) == pytest.approx(BASE.yaw)


def test_unknown_shape_is_rejected():
    with pytest.raises(ValueError):
        grasp_yaw("sphere", 0.0, 0.0)


# ---- 隣の物体を避ける手先の向き ----


def test_bearing_yaw():
    from restorebot.core.grasp import bearing_yaw

    assert bearing_yaw(BASE, (0.75, 0.70)) == pytest.approx(math.pi / 2)  # 正面
    assert bearing_yaw(BASE, (0.40, 0.62)) == pytest.approx(math.atan2(0.64, -0.35))  # 左の端


def _finger_axis(yaw):
    return (math.sin(yaw), -math.cos(yaw))  # 真下向きの手先の y 方向（指が閉じる方向）


def test_cylinder_without_neighbors_uses_reference():
    from restorebot.core.grasp import choose_grasp_yaw

    assert choose_grasp_yaw("cylinder", 0.0, 1.2, (0.6, 0.7), [], PARAMS) == pytest.approx(1.2)


def test_cylinder_turns_fingers_away_from_tall_neighbor():
    from restorebot.core.grasp import Footprint, choose_grasp_yaw, hand_clearance

    obj = (0.60, 0.70)
    ref = math.pi / 2  # 指は x 方向に閉じる
    fx, fy = _finger_axis(ref)
    neighbor = Footprint((obj[0] + 0.0975 * fx, obj[1] + 0.0975 * fy), 0.0225)  # 指の方向に隙間 5cm（#13 の失敗例）
    assert hand_clearance(obj, ref, neighbor, PARAMS) < PARAMS.safe_clearance  # 基準の向きでは手が当たる
    yaw = choose_grasp_yaw("cylinder", 0.0, ref, obj, [neighbor], PARAMS)
    assert hand_clearance(obj, yaw, neighbor, PARAMS) >= PARAMS.safe_clearance
    assert abs(wrap_angle(yaw - ref)) <= math.pi / 2 + 1e-9  # 手首は ±90° 以内
    # 当たらない向きのうち基準に最も近いもの（15° 刻み）
    step = PARAMS.yaw_step
    closer = [ref + s * k * step for k in range(round(abs(wrap_angle(yaw - ref)) / step)) for s in (1, -1)]
    assert all(hand_clearance(obj, y, neighbor, PARAMS) < PARAMS.safe_clearance for y in closer)


def test_cylinder_picks_best_clearance_when_none_is_safe():
    from restorebot.core.grasp import Footprint, choose_grasp_yaw, hand_clearance

    obj = (0.60, 0.70)
    # 四方を囲まれていて、どの向きでも安全な隙間がとれない → 隙間が最大の向き
    ring = [Footprint((obj[0] + 0.07 * math.cos(a), obj[1] + 0.07 * math.sin(a)), 0.02) for a in (0, 1.6, 3.1, 4.7)]
    yaw = choose_grasp_yaw("cylinder", 0.0, 0.3, obj, ring, PARAMS)
    best = max(min(hand_clearance(obj, 0.3 + k * PARAMS.yaw_step, n, PARAMS) for n in ring) for k in range(-6, 7))
    assert min(hand_clearance(obj, yaw, n, PARAMS) for n in ring) == pytest.approx(best)


def test_box_ignores_neighbors_and_keeps_axis():
    from restorebot.core.grasp import Footprint, choose_grasp_yaw

    obj = (0.60, 0.70)
    neighbor = Footprint((0.70, 0.70), 0.03)
    yaw = choose_grasp_yaw("cube", 0.4, BASE.yaw, obj, [neighbor], PARAMS)
    assert yaw == pytest.approx(grasp_yaw("cube", 0.4, BASE.yaw))  # 箱は物体の軸で決まる


def test_hand_down_quat():
    from restorebot.core.grasp import hand_down_quat_wxyz, quat_angle_wxyz
    from restorebot.core.types import quat_wxyz_from_yaw

    assert hand_down_quat_wxyz(0.0) == pytest.approx((0.0, 1.0, 0.0, 0.0))  # サンプルの真下向き
    # yaw ⊗ (0, 1, 0, 0) と一致する
    w1, x1, y1, z1 = quat_wxyz_from_yaw(0.8)
    expected = (-x1, w1, z1, -y1)  # (w1, x1, y1, z1) ⊗ (0, 1, 0, 0) の展開
    assert hand_down_quat_wxyz(0.8) == pytest.approx(expected)
    # 向きの差：yaw が 0.3 違えば 0.3、q と -q は同じ
    assert quat_angle_wxyz(hand_down_quat_wxyz(0.0), hand_down_quat_wxyz(0.3)) == pytest.approx(0.3)
    q = hand_down_quat_wxyz(1.0)
    assert quat_angle_wxyz(q, tuple(-c for c in q)) == pytest.approx(0.0, abs=1e-6)


# ---- 段階の列 ----


def test_pick_stages():
    center = (0.6, 0.7, 0.745)
    stages = pick_stages(center, 0.3, BASE, PARAMS)
    assert [s.name for s in stages] == ["approach", "descend", "grasp", "lift", "standby"]
    approach, descend, grasp, lift, standby = stages
    assert approach.position == pytest.approx((0.6, 0.7, 0.745 + PARAMS.approach_height))
    assert descend.position == pytest.approx(center) and grasp.position == pytest.approx(center)
    assert lift.position == pytest.approx((0.6, 0.7, 0.745 + PARAMS.lift_height))
    assert standby.position == pytest.approx(standby_grasp_position(BASE))
    assert [s.gripper for s in stages] == [Gripper.OPEN, Gripper.OPEN, Gripper.CLOSE, Gripper.CLOSE, Gripper.CLOSE]
    assert [s.wait_gripper for s in stages] == [False, False, True, False, False]
    # 掴む物体は、下降を始めたら障害物から外す（掴んでいる間も外したまま）
    assert [s.object_is_obstacle for s in stages] == [True, False, False, False, False]
    # 掴むまでは物体の向きに合わせた手先の向き、待機姿勢は台車の向き
    assert all(s.yaw == 0.3 for s in stages[:4]) and standby.yaw == BASE.yaw


def test_place_stages_keep_hand_yaw_and_add_clearance():
    target = (0.45, 0.72, 0.745)
    stages = place_stages(target, 0.3, BASE, PARAMS)
    assert [s.name for s in stages] == ["approach", "descend", "release", "clear", "retreat", "standby"]
    approach, descend, release, clear, retreat, standby = stages
    p = 0.745 + PARAMS.place_clearance
    assert descend.position == pytest.approx((0.45, 0.72, p))
    assert approach.position[2] == pytest.approx(p + PARAMS.approach_height)
    # 指を開いたら、まず少しだけ真上に上げてから退避する（#13：一気に上げるとペン立てが指に引っかかって倒れた）
    assert clear.position == pytest.approx((0.45, 0.72, p + PARAMS.clear_height))
    assert retreat.position[2] == pytest.approx(p + PARAMS.lift_height)
    assert release.min_duration == pytest.approx(PARAMS.release_settle)  # 指を開いた後、物体が落ち着くまで待つ
    assert [s.gripper for s in stages] == [Gripper.CLOSE, Gripper.CLOSE] + [Gripper.OPEN] * 4
    assert [s.wait_gripper for s in stages] == [False, False, True, False, False, False]
    # 置いて退避するまでは障害物から外したまま、待機姿勢へ戻るときに戻す
    assert [s.object_is_obstacle for s in stages] == [False] * 5 + [True]
    assert all(s.yaw == 0.3 for s in stages[:5])  # 置いて退避するまで手先を回さない


def test_carrying_stages_limit_hand_speed():
    # 物を持って動く段階だけ、手先の目標の速さを抑える（#13：速く振ると円柱が手から抜けた）
    pick = pick_stages((0.6, 0.7, 0.745), 0.0, BASE, PARAMS)
    place = place_stages((0.45, 0.72, 0.745), 0.0, BASE, PARAMS)
    carry = PARAMS.carry_speed
    assert [s.max_speed for s in pick] == [None, None, None, carry, carry]
    assert [s.max_speed for s in place] == [carry, carry, None, None, None, None]


def _settle(samples, **kw):
    from restorebot.core.grasp import FingerSettle

    s = FingerSettle(open_positions=(0.04, 0.04), **kw)
    return [s.update(t, p) for t, p in samples]


def test_finger_settle_after_window():
    # 物を挟んで止まった指：位置が window のあいだ tol 以内なら止まったとみなす（速度の値は使わない。#25）
    dt = 1 / 60
    samples = [(i * dt, (0.0205, 0.016)) for i in range(20)]
    done = _settle(samples, window=0.2, tol=5e-4, min_close=0.002)
    assert not any(done[:12])  # 0.2 秒たつまでは判定しない
    assert done[-1]


def test_finger_settle_requires_still_and_closed():
    dt = 1 / 60
    moving = [(i * dt, (0.04 - 0.002 * i, 0.04 - 0.002 * i)) for i in range(20)]  # まだ閉じている途中
    assert not any(_settle(moving, window=0.2, tol=5e-4, min_close=0.002))
    still_open = [(i * dt, (0.0395, 0.0395)) for i in range(30)]  # 開いたまま止まっている：閉じていない
    assert not any(_settle(still_open, window=0.2, tol=5e-4, min_close=0.002))


def test_finger_settle_reset():
    from restorebot.core.grasp import FingerSettle

    s = FingerSettle(open_positions=(0.04, 0.04), window=0.2, tol=5e-4, min_close=0.002)
    for i in range(20):
        s.update(i / 60, (0.02, 0.02))
    assert s.update(20 / 60, (0.02, 0.02))
    s.reset()
    assert not s.update(21 / 60, (0.02, 0.02))  # リセットしたら、また window 待つ


def test_move_toward():
    from restorebot.core.grasp import move_toward

    assert move_toward((0.0, 0.0, 0.0), (0.3, 0.4, 0.0), 0.1) == pytest.approx((0.06, 0.08, 0.0))
    assert move_toward((0.0, 0.0, 0.0), (0.03, 0.04, 0.0), 0.1) == pytest.approx((0.03, 0.04, 0.0))  # 近ければ目標へ
    assert move_toward((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), 0.1) == pytest.approx((1.0, 1.0, 1.0))


def test_stage_min_duration_delays_advance():
    from restorebot.core.grasp import Stage

    stage = Stage("release", (0.0, 0.0, 0.0), 0.0, Gripper.OPEN, True, False, 2.0, min_duration=0.5)
    seq = StageSequencer([stage], GraspParams(stability_duration=0.0, gripper_dwell=0.2))
    seq.start(0.0)
    assert seq.update(0.3, 0.0, 0.0, gripper_done=True) is Status.RUNNING  # 指は開いたが 0.5 秒たっていない
    assert seq.update(0.5, 0.0, 0.0, gripper_done=True) is Status.SUCCEEDED


def test_standby_grasp_position_is_below_hand():
    # 待機姿勢：台車座標で手先 (0.4, 0, 0.95)。指先の間はその 0.1034m 下
    x, y, z = standby_grasp_position(Pose2D(1.0, 2.0, 0.0))
    assert (x, y, z) == pytest.approx((1.4, 2.0, 0.95 - GRASP_OFFSET_Z))
    x, y, _ = standby_grasp_position(Pose2D(0.0, 0.0, math.pi / 2))
    assert (x, y) == pytest.approx((0.0, 0.4))


# ---- 段階を進める判定 ----


def _seq(stages=None, **kw):
    params = GraspParams(**kw)
    stages = stages or pick_stages((0.6, 0.7, 0.745), 0.0, BASE, params)
    seq = StageSequencer(stages, params)
    seq.start(0.0)
    return seq


def test_advances_after_stable_convergence():
    seq = _seq(stability_duration=0.1)
    assert seq.update(0.05, 0.001, 0.0, gripper_done=False) is Status.RUNNING  # 収束し始め
    assert seq.stage.name == "approach"
    assert seq.update(0.10, 0.001, 0.0, gripper_done=False) is Status.RUNNING  # まだ 0.05 秒
    assert seq.update(0.16, 0.001, 0.0, gripper_done=False) is Status.RUNNING
    assert seq.stage.name == "descend"


def test_stability_timer_resets_when_error_grows():
    seq = _seq(stability_duration=0.1)
    seq.update(0.0, 0.001, 0.0, gripper_done=False)
    seq.update(0.05, 0.05, 0.0, gripper_done=False)  # 離れた
    seq.update(0.12, 0.001, 0.0, gripper_done=False)  # 収束し直し（ここから 0.1 秒）
    assert seq.stage.name == "approach"
    seq.update(0.23, 0.001, 0.0, gripper_done=False)
    assert seq.stage.name == "descend"


def test_orientation_error_blocks_convergence():
    seq = _seq(stability_duration=0.0)
    seq.update(0.1, 0.001, 0.5, gripper_done=False)
    assert seq.stage.name == "approach"


def test_grasp_waits_for_gripper_and_dwell():
    seq = _seq(stability_duration=0.0, gripper_dwell=0.2)
    seq.update(0.0, 0.0, 0.0, False)  # approach → descend
    seq.update(0.1, 0.0, 0.0, False)  # descend → grasp（t = 0.1 で開始）
    assert seq.stage.name == "grasp"
    seq.update(0.2, 0.0, 0.0, gripper_done=True)  # 0.1 秒しかたっていない
    assert seq.stage.name == "grasp"
    seq.update(0.35, 0.0, 0.0, gripper_done=False)  # 指がまだ動いている
    assert seq.stage.name == "grasp"
    seq.update(0.4, 0.0, 0.0, gripper_done=True)
    assert seq.stage.name == "lift"


def test_succeeds_after_last_stage():
    seq = _seq(stability_duration=0.0, gripper_dwell=0.0)
    status, t = Status.RUNNING, 0.0
    while status is Status.RUNNING and t < 5.0:
        status = seq.update(t, 0.0, 0.0, gripper_done=True)
        t += 0.1
    assert status is Status.SUCCEEDED
    assert seq.done


def test_timeout_fails_with_reason():
    seq = _seq()
    timeout = seq.stage.timeout
    assert seq.update(timeout - 0.01, 0.2, 0.0, False) is Status.RUNNING
    assert seq.update(timeout + 0.01, 0.2, 0.0, False) is Status.FAILED
    assert "approach" in seq.failure_reason and "0.2" in seq.failure_reason


def test_timeout_is_measured_from_stage_start():
    seq = _seq(stability_duration=0.0)
    seq.update(3.0, 0.0, 0.0, False)  # approach → descend（t = 3.0 で開始）
    timeout = seq.stage.timeout
    assert seq.update(3.0 + timeout - 0.01, 0.2, 0.0, False) is Status.RUNNING


def test_update_before_start_is_an_error():
    seq = StageSequencer(pick_stages((0.6, 0.7, 0.745), 0.0, BASE, PARAMS), PARAMS)
    with pytest.raises(RuntimeError):
        seq.update(0.0, 0.0, 0.0, False)


# ---- 成功の判定（#4 の classify_trial）----


def _obs(**kw):
    base = dict(
        controller_done=True,
        moved_before_grasp_m=0.0,
        max_lift_m=0.10,
        dropped=False,
        place_xy_err_m=0.005,
        tilt_deg=1.0,
    )
    return TrialObservation(**{**base, **kw})


@pytest.mark.parametrize(
    ("kw", "expected"),
    [
        ({}, "success"),
        ({"moved_before_grasp_m": 0.02}, "collision_before_grasp"),
        ({"max_lift_m": 0.01}, "grasp_miss"),
        ({"dropped": True}, "dropped"),
        ({"tilt_deg": 20.0}, "tipped"),
        ({"place_xy_err_m": 0.04}, "placement_off"),
        ({"controller_done": False}, "controller_failure"),
    ],
)
def test_classify_trial(kw, expected):
    assert classify_trial(_obs(**kw)) == expected


def test_classify_uses_3cm_success_criterion():
    # 設計書 13 章：戻し先から水平 3cm 以内（#4 の検証では 2cm だった）
    assert classify_trial(_obs(place_xy_err_m=0.025)) == "success"
