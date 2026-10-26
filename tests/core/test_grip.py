"""指の開きによる、掴めたかどうかの判定（core.grasp.judge_grip・GripMonitor。#27）の単体テスト。

指の開き：左右の指の関節の値の和 [m]（全開 0.08、全閉 0）。
"""

import pytest

from restorebot.core.grasp import GraspParams, GripMonitor, judge_grip

P = GraspParams()
W = 0.045  # 物体の掴む幅（ペン立ての直径・リモコンの短い辺）


@pytest.mark.parametrize("opening", [W, W - 0.0035, W - 0.010, W + 0.005])
def test_grasped(opening):
    # 幅どおり、指が少し食い込む（#25 の実測で約 3.5mm）、少しつぶれ気味・少し広め
    assert judge_grip(opening, W, P) is None


def test_fully_closed_is_empty():
    reason = judge_grip(0.002, W, P)
    assert reason is not None and "閉じきった" in reason


def test_too_wide_is_not_grasped():
    reason = judge_grip(W + 0.02, W, P)
    assert reason is not None and "広い" in reason


def test_too_narrow_is_not_grasped():
    reason = judge_grip(W - 0.025, W, P)  # 物体の端をつまんでいる
    assert reason is not None and "狭い" in reason


def test_thresholds_boundaries():
    # 全閉：grip_closed_max 未満
    assert "閉じきった" in judge_grip(P.grip_closed_max - 1e-4, 0.012, P)
    # 狭すぎる：幅 − grip_under_tol 未満（境界のすぐ内側は掴めた）
    assert judge_grip(W - P.grip_under_tol + 1e-4, W, P) is None
    assert "狭い" in judge_grip(W - P.grip_under_tol - 1e-4, W, P)
    # 広すぎる：幅 ＋ grip_over_tol を超える
    assert judge_grip(W + P.grip_over_tol - 1e-4, W, P) is None
    assert "広い" in judge_grip(W + P.grip_over_tol + 1e-4, W, P)


def test_monitor_ignores_short_glitch():
    m = GripMonitor(W, P)
    dt = 1 / 60
    t = 0.0
    for opening in [0.041] * 10 + [0.0] * 3 + [0.041] * 10:  # 0.05 秒だけ閉じたように見えた
        t += dt
        assert m.update(t, opening) is None


def test_monitor_detects_drop_after_hold_time():
    m = GripMonitor(W, P)
    dt = 1 / 60
    t, results = 0.0, []
    for opening in [0.041] * 10 + [0.0] * 12:  # 運ぶ途中で落とした（指が全閉まで閉じた）
        t += dt
        results.append(m.update(t, opening))
    first = next(i for i, r in enumerate(results) if r is not None)
    assert 10 + 5 <= first <= 10 + 7  # 0.1 秒（6 ステップ）続いたら
    assert "落とした" in results[first] and "閉じきった" in results[first]


def test_monitor_reports_not_grasped_when_never_held():
    m = GripMonitor(W, P)
    results = [m.update((i + 1) / 60, 0.0) for i in range(10)]  # 閉じた指が空振りした（最初から全閉）
    reason = next(r for r in results if r is not None)
    assert "掴めていない" in reason and "落とした" not in reason
