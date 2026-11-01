"""restorebot.core.evaluation（定量評価の判定と集計。設計書 13 章、#18）の単体テスト。"""

import pytest

from restorebot.core.evaluation import evaluate_run, markdown_tables, summarize
from restorebot.core.scene import HOME_V2

SCENE = HOME_V2
NAME = {o.key: o.name for o in SCENE.objects}


def _result(moved, err, yaw=None, **kw):
    return {
        "seed": kw.pop("seed", 100),
        "moved": list(moved),
        "status": kw.pop("status", "succeeded"),
        "home_err_m": err,
        "home_yaw_err_deg": yaw or {},
        "untouched_moved_m": kw.pop("untouched", {"spray": 0.0}),
        "sim_time_s": kw.pop("sim", 60.0),
        "wall_s": kw.pop("wall", 100.0),
        "cost_usd": kw.pop("cost", 0.02),
        "robot_pose_checks": [{"diff_m": 0.001}, {"diff_m": 0.014}, {"diff_m": None}],
        "grip_checks": [{"grip_ok": True, "truth_ok": True}, {"grip_ok": False, "truth_ok": False}],
        **kw,
    }


def _log(predicted, tasks, verify_predicted=None):
    det = {"ok": True, "moves": [{"name": n} for n in predicted]}
    verify = (
        None
        if verify_predicted is None
        else {"detection": {"ok": True, "moves": [{"name": n} for n in verify_predicted]}}
    )
    return {"status": "succeeded", "detection": det, "verify": verify, "tasks": tasks}


def _task(key, status="succeeded", reason=None, round_=1):
    return {"key": key, "status": status, "reason": reason, "round": round_}


def test_task_success_uses_3cm_boundary():
    r = evaluate_run(
        _result(["cup", "remote"], {"cup": 0.0299, "remote": 0.0301}, {"remote": 2.0}),
        _log([NAME["cup"], NAME["remote"]], [_task("cup"), _task("remote")]),
        SCENE,
    )
    ok = {t["key"]: t["ok"] for t in r["tasks"]}
    assert ok == {"cup": True, "remote": False}
    assert r["task_success"] == 1 and not r["episode_success"]
    assert {t["key"]: t["yaw_ok"] for t in r["tasks"]} == {"cup": None, "remote": True}  # 箱だけ向きを判定
    assert r["failures"] == [{"key": "remote", "category": "戻し先から 3cm を超えた"}]


def test_box_yaw_over_10_deg_is_recorded_but_position_decides_success():
    r = evaluate_run(
        _result(["remote"], {"remote": 0.01}, {"remote": 12.0}), _log([NAME["remote"]], [_task("remote")]), SCENE
    )
    assert r["episode_success"] and r["tasks"][0]["yaw_ok"] is False


def test_detection_precision_recall_counts():
    # 動かしたのはコップとリモコン。VLM はコップ（正解）、スプレー缶（動いていない）、一覧にない名前を答え、
    # リモコンを見逃した
    log = _log([NAME["cup"], NAME["spray"], "白いマグカップ"], [_task("cup"), _task("spray")])
    r = evaluate_run(_result(["cup", "remote"], {"cup": 0.01, "remote": 0.5}), log, SCENE)
    assert r["detection"] == {"tp": 1, "fp": 2, "fn": 1}
    assert {"key": "remote", "category": "VLM の見逃し"} in r["failures"]


def test_failure_category_from_last_task_record():
    log = _log(
        [NAME["pen_stand"]],
        [
            _task("pen_stand", "skipped", "lift：掴めていない：指が閉じきった"),
            _task("pen_stand", "skipped", "物体の位置が分からない（天板の上にない）", round_=2),
        ],
        verify_predicted=[NAME["pen_stand"]],
    )
    r = evaluate_run(_result(["pen_stand"], {"pen_stand": 1.2}, status="partial"), log, SCENE)
    assert r["failures"] == [{"key": "pen_stand", "category": "場所が分からなかった"}]
    assert r["retried"] == 1


def test_detected_but_no_task_is_unusable_answer():
    r = evaluate_run(_result(["cup"], {"cup": 0.3}), _log([NAME["cup"]], []), SCENE)
    assert r["failures"] == [{"key": "cup", "category": "VLM の答えが使えなかった"}]


def test_crashed_run():
    r = evaluate_run({"seed": 105, "status": "error", "error": "RuntimeError: x", "moved": ["cup"]}, None, SCENE)
    assert r["status"] == "error" and not r["episode_success"] and r["task_success"] == 0
    assert r["failures"] == [{"key": "cup", "category": "実行エラー"}]


def _two_runs():
    a = evaluate_run(
        _result(["cup", "remote"], {"cup": 0.004, "remote": 0.002}, {"remote": 1.0}, seed=100, cost=0.02),
        _log([NAME["cup"], NAME["remote"]], [_task("cup"), _task("remote")]),
        SCENE,
    )
    b = evaluate_run(
        _result(["spray"], {"spray": 0.5}, seed=101, cost=None, untouched={"cup": 0.003}),
        _log([], []),
        SCENE,
    )
    return [a, b]


def test_summarize():
    s = summarize(_two_runs())
    assert s["episodes"] == 2 and s["episode_success"] == 1
    assert s["tasks"] == 3 and s["task_success"] == 2
    assert s["task_success_rate"] == pytest.approx(2 / 3)
    assert s["err_success_mm"] == {"mean": pytest.approx(3.0), "max": pytest.approx(4.0)}  # 成功したタスクだけ
    assert s["detection"] == {"tp": 2, "fp": 0, "fn": 1, "precision": 1.0, "recall": pytest.approx(2 / 3)}
    assert s["cost_usd"] is None  # 費用が分からない実行がある
    assert s["failures"] == {"VLM の見逃し": 1}
    assert s["box_yaw"] == {"n": 1, "ok": 1, "max_deg": 1.0}
    assert s["untouched_max_mm"] == pytest.approx(3.0)
    assert s["robot_pose_max_cm"] == pytest.approx(1.4)
    assert s["grip_agree"] == "4/4"


def test_markdown_tables_lists_each_episode():
    runs = _two_runs()
    md = markdown_tables(runs, summarize(runs))
    assert "| 100 |" in md and "| 101 |" in md
    assert NAME["spray"] in md and "VLM の見逃し" in md
