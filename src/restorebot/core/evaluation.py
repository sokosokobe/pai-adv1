"""定量評価の判定と集計（設計書 13 章。#18）。Isaac Sim に依存しない。

1 回の実行の結果（run_episode.py の result.json：シミュレータの真値）と記録（log.json：ロボットがしたこと）から、
タスク・エピソードの成否と指標を求め（evaluate_run）、複数の実行を集計する（summarize）。

- タスクの成否：動かした物体が、戻し先から水平 SUCCESS_TOL（3cm）以内（18 章 Q7）
- 箱の向き：いつもの向きとの差（180° の範囲）が YAW_TOL（10°）以内か。成否とは別に記録する（#25）
- エピソードの成否：動かした物体がすべて戻った
- VLM の検出：最初の比較で VLM が挙げた物体と、実際に動かした物体（真値）を比べる。一覧にない名前は誤検出
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Sequence

from restorebot.core.vlm.parse import normalize_name
from restorebot.core.vlm.summary import reason_category

SUCCESS_TOL = 0.03  # [m]
YAW_TOL = 10.0  # [deg]


def _predicted_keys(detection: dict | None, by_name: dict[str, str]) -> list[str | None]:
    """VLM が挙げた物体の key（一覧にない名前は None）。"""
    moves = (detection or {}).get("moves", [])
    return [by_name.get(normalize_name(m["name"])) for m in moves]


def _failure_category(key: str, log: dict, detected: set[str]) -> str:
    """戻らなかった物体 key の失敗の分類。"""
    if key not in detected:
        return "VLM の見逃し"
    records = [t for t in log.get("tasks", []) if t["key"] == key]
    if not records:
        return "VLM の答えが使えなかった"  # 戻し先の深度が無効、など（タスクにならなかった）
    last = records[-1]
    if last["status"] != "succeeded":
        return reason_category(last["status"], last["reason"])
    return "戻し先から 3cm を超えた"  # ロボットは置けたと判断したが、真値では 3cm より遠い


def evaluate_run(result: dict, log: dict | None, scene) -> dict:
    """1 回の実行の指標。log が None（実行が途中で止まった）でも、分かる範囲で返す。"""
    by_name = {normalize_name(o.name): o.key for o in scene.objects}
    moved = list(result.get("moved", []))
    err = result.get("home_err_m", {})
    yaw = result.get("home_yaw_err_deg", {})
    crashed = log is None or result.get("status") == "error"

    tasks = []
    for k in moved:
        e = err.get(k)
        ok = e is not None and e <= SUCCESS_TOL and not crashed
        y = yaw.get(k)
        tasks.append(
            {
                "key": k,
                "err_m": e,
                "ok": ok,
                "yaw_err_deg": y,
                "yaw_ok": None if y is None else y <= YAW_TOL,
            }
        )

    run: dict = {
        "seed": result.get("seed"),
        "status": "error" if crashed else result.get("status"),
        "error": result.get("error"),
        "moved": moved,
        "tasks": tasks,
        "task_success": sum(t["ok"] for t in tasks),
        "episode_success": bool(tasks) and all(t["ok"] for t in tasks),
        "sim_time_s": result.get("sim_time_s"),
        "wall_s": result.get("wall_s"),
        "cost_usd": result.get("cost_usd"),
        "untouched_max_m": max(result.get("untouched_moved_m", {}).values(), default=0.0),
    }
    pose_diffs = [c["diff_m"] for c in result.get("robot_pose_checks", []) if c.get("diff_m") is not None]
    run["robot_pose_max_m"] = max(pose_diffs, default=None)
    grips = result.get("grip_checks", [])
    run["grip_agree"] = [sum(c["grip_ok"] == c["truth_ok"] for c in grips), len(grips)]

    if crashed:
        run["detection"] = None
        run["retried"] = 0
        run["failures"] = [{"key": t["key"], "category": "実行エラー"} for t in tasks if not t["ok"]]
        return run

    predicted = _predicted_keys(log.get("detection"), by_name)
    hits = {k for k in predicted if k is not None}
    run["detection"] = {
        "tp": len(hits & set(moved)),
        "fp": sum(k is None or k not in moved for k in predicted),
        "fn": len(set(moved) - hits),
    }
    verify_det = (log.get("verify") or {}).get("detection")
    detected = hits | {k for k in _predicted_keys(verify_det, by_name) if k is not None}
    run["retried"] = sum(t.get("round", 1) == 2 for t in log.get("tasks", []))
    run["failures"] = [
        {"key": t["key"], "category": _failure_category(t["key"], log, detected)} for t in tasks if not t["ok"]
    ]
    return run


def summarize(runs: Sequence[dict]) -> dict:
    """複数の実行の集計。"""
    tasks = [t for r in runs for t in r["tasks"]]
    ok_err = [t["err_m"] * 1000 for t in tasks if t["ok"]]
    det = [r["detection"] for r in runs if r["detection"] is not None]
    tp, fp, fn = (sum(d[k] for d in det) for k in ("tp", "fp", "fn"))
    yaws = [t["yaw_err_deg"] for t in tasks if t["yaw_err_deg"] is not None and t["ok"]]
    costs = [r["cost_usd"] for r in runs]
    sims = [r["sim_time_s"] for r in runs if r["sim_time_s"] is not None]
    walls = [r["wall_s"] for r in runs if r["wall_s"] is not None]
    poses = [r["robot_pose_max_m"] for r in runs if r["robot_pose_max_m"] is not None]
    return {
        "episodes": len(runs),
        "episode_success": sum(r["episode_success"] for r in runs),
        "tasks": len(tasks),
        "task_success": sum(t["ok"] for t in tasks),
        "task_success_rate": (sum(t["ok"] for t in tasks) / len(tasks)) if tasks else None,
        "err_success_mm": {"mean": statistics.fmean(ok_err), "max": max(ok_err)} if ok_err else None,
        "box_yaw": {"n": len(yaws), "ok": sum(y <= YAW_TOL for y in yaws), "max_deg": max(yaws, default=None)},
        "detection": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        },
        "retried_tasks": sum(r.get("retried", 0) for r in runs),
        "failures": dict(Counter(f["category"] for r in runs for f in r["failures"])),
        "sim_time_s_mean": statistics.fmean(sims) if sims else None,
        "wall_s_mean": statistics.fmean(walls) if walls else None,
        "cost_usd": None if any(c is None for c in costs) else sum(costs),
        "untouched_max_mm": max((r["untouched_max_m"] for r in runs), default=0.0) * 1000,
        "robot_pose_max_cm": max(poses) * 100 if poses else None,
        "grip_agree": f"{sum(r['grip_agree'][0] for r in runs)}/{sum(r['grip_agree'][1] for r in runs)}",
    }


def _fmt(v, spec: str, none: str = "–") -> str:
    return none if v is None else format(v, spec)


def markdown_tables(runs: Sequence[dict], summary: dict, scene=None) -> str:
    """エピソードごとの表と、全体の表（Markdown）。scene を与えると物体を表示名で書く。"""
    if scene is None:
        from restorebot.core.scene import HOME_V2 as scene  # noqa: N811
    name = {o.key: o.name for o in scene.objects}
    lines = [
        "| シード | 動かした物体 | VLM の検出（正・誤・見逃し） | 戻った | 戻し先からの誤差 | 箱の向きの差 | やり直し "
        "| 時間（シミュ／実） | 費用 | 失敗の内訳 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        d = r["detection"]
        det = "–" if d is None else f"{d['tp']}・{d['fp']}・{d['fn']}"
        errs = "、".join(
            f"{name.get(t['key'], t['key'])} {_fmt(None if t['err_m'] is None else t['err_m'] * 1000, '.1f')}mm"
            + ("" if t["ok"] else "（×）")
            for t in r["tasks"]
        )
        yaws = "、".join(f"{t['yaw_err_deg']:.1f}°" for t in r["tasks"] if t["yaw_err_deg"] is not None) or "–"
        fails = "、".join(f"{name.get(f['key'], f['key'])}：{f['category']}" for f in r["failures"]) or "–"
        lines.append(
            f"| {r['seed']} | {'、'.join(name.get(k, k) for k in r['moved'])} | {det} "
            f"| {r['task_success']}/{len(r['tasks'])} | {errs} | {yaws} | {r.get('retried', 0)} "
            f"| {_fmt(r['sim_time_s'], '.0f')}／{_fmt(r['wall_s'], '.0f')} 秒 | {_fmt(r['cost_usd'], '.3f', '不明')} "
            f"| {fails} |"
        )
    s = summary
    d = s["detection"]
    e = s["err_success_mm"] or {}
    y = s["box_yaw"]
    lines += [
        "",
        "| 指標 | 値 |",
        "|---|---|",
        f"| エピソードの成功 | {s['episode_success']}/{s['episodes']} |",
        f"| タスクの成功（戻し先から 3cm 以内） | {s['task_success']}/{s['tasks']}"
        f"（{_fmt(None if s['task_success_rate'] is None else s['task_success_rate'] * 100, '.0f')}%） |",
        f"| 成功したタスクの誤差（平均・最大） | {_fmt(e.get('mean'), '.1f')}mm・{_fmt(e.get('max'), '.1f')}mm |",
        f"| 箱の向き（10° 以内、成功したタスク） | {y['ok']}/{y['n']}（最大 {_fmt(y['max_deg'], '.1f')}°） |",
        f"| VLM の検出（適合率・再現率） | {_fmt(d['precision'], '.2f')}・{_fmt(d['recall'], '.2f')}"
        f"（正 {d['tp']}・誤 {d['fp']}・見逃し {d['fn']}） |",
        f"| やり直したタスク | {s['retried_tasks']} |",
        f"| 所要時間（平均。シミュレーション・実時間） | {_fmt(s['sim_time_s_mean'], '.0f')} 秒・"
        f"{_fmt(s['wall_s_mean'], '.0f')} 秒 |",
        f"| VLM の費用（合計） | {_fmt(s['cost_usd'], '.3f', '不明')} USD |",
        f"| 動かしていない物体の動き（最大） | {s['untouched_max_mm']:.1f}mm |",
        f"| 台車の位置の推定とオドメトリの差（最大） | {_fmt(s['robot_pose_max_cm'], '.1f')}cm |",
        f"| 指の開きの判定と真値の一致 | {s['grip_agree']} |",
        f"| 失敗の内訳 | {'、'.join(f'{k} {v}' for k, v in s['failures'].items()) or 'なし'} |",
    ]
    return "\n".join(lines) + "\n"
