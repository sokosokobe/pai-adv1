"""restorebot.core.vlm.summary（報告文）の単体テスト。実際の API は呼ばない。"""

import json

from restorebot.core.episode import UNKNOWN_POSITION
from restorebot.core.scene import HOME_V2
from restorebot.core.vlm.client import DummyVlmClient, VlmApiError, VlmResult
from restorebot.core.vlm.summary import (
    reason_category,
    report_facts,
    report_prompt,
    template_report,
    write_report,
)

SCENE = HOME_V2


def _move(name, cur, tgt, cur_px=(100.0, 200.0), tgt_px=(300.0, 400.0)):
    return {
        "name": name,
        "current_location": cur,
        "current_px": list(cur_px),
        "target_location": tgt,
        "target_px": list(tgt_px),
    }


def _task(key, status, reason=None, round_=1, attempts=1, duration=10.0):
    return {
        "key": key,
        "place_position": [0.45, 0.72, 0.745],
        "pick_position": [0.3, 0.6, 0.745],
        "status": status,
        "reason": reason,
        "pick_attempts": attempts,
        "final_position": [0.451, 0.721, 0.745],
        "error_m": 0.0012,  # 真値（報告文には入れない）
        "started_s": 2.0,
        "duration_s": duration,
        "round": round_,
        "estimate_error_m": 0.003,
        "estimate_yaw_error_deg": None,
    }


def make_log(tasks, moves, *, status="succeeded", verify_moves=None, dropped=()):
    return {
        "meta": {"seed": 2, "vlm": "dummy"},
        "status": status,
        "reason": None,
        "sim_time_s": 125.0,
        "cost_usd": 0.02,
        "events": [],
        "detection": {"ok": True, "moves": moves, "cost_usd": 0.02, "attempts": []},
        "instructions": [],
        "dropped": list(dropped),
        "tasks": tasks,
        "verify": None
        if verify_moves is None
        else {"detection": {"ok": True, "moves": verify_moves, "cost_usd": 0.0}, "instructions": [], "dropped": []},
        "photos": ["after", "current", "reference"],
    }


CUP, PEN, REMOTE = (SCENE.object(k).name for k in ("cup", "pen_stand", "remote"))


def partial_log():
    moves = [
        _move(CUP, "机の左端", "机の左寄り"),
        _move(PEN, "低い棚の上", "机の中央"),
        _move(REMOTE, "机の右端", "机の右寄り"),
    ]
    tasks = [
        _task("cup", "succeeded"),
        _task("pen_stand", "skipped", "掴み損ね（持ち上がった高さ -0.477m）", attempts=2),
        _task("remote", "failed", "place が 7.0 秒で時間切れ（位置の誤差 0.0200m、向きの誤差 0.0010rad）"),
        _task("remote", "succeeded", round_=2),
        _task("pen_stand", "skipped", UNKNOWN_POSITION, round_=2, attempts=0),
    ]
    verify = [_move(PEN, "床の上", "机の中央"), _move(REMOTE, "机の右寄りの少し手前", "机の右寄り")]
    return make_log(tasks, moves, status="partial", verify_moves=verify, dropped=["謎の物体: 物体の一覧にない名前"])


def test_reason_category():
    assert reason_category("succeeded", None) == "戻した"
    assert reason_category("skipped", UNKNOWN_POSITION) == "場所が分からなかった"
    assert reason_category("skipped", "掴み損ね（持ち上がった高さ 0.001m）") == "掴めなかった"
    assert reason_category("skipped", "grasp が 2.0 秒で時間切れ（位置の誤差 0.0041m）") == "掴めなかった"
    assert reason_category("skipped", "経路がない") == "近づけなかった"
    assert reason_category("skipped", "止め位置がない（届かない）") == "近づけなかった"
    assert reason_category("failed", "スタック（3 秒進まない。(0.10, 0.20) 付近）") == "運べなかった"
    assert reason_category("failed", "place が 7.0 秒で時間切れ") == "置けなかった"


def test_facts_use_final_result_per_object_in_task_order():
    facts = report_facts(partial_log(), SCENE)
    assert [o["name"] for o in facts["objects"]] == [CUP, PEN, REMOTE]
    assert [o["no"] for o in facts["objects"]] == [1, 2, 3]
    cup, pen, remote = facts["objects"]
    assert cup["result"] == "戻した" and not cup["retried"]
    assert remote["result"] == "戻した" and remote["retried"]  # やり直して戻せた
    assert pen["result"] == "場所が分からなかった" and pen["retried"]
    assert pen["from"] == "低い棚の上" and pen["to"] == "机の中央"  # 最初の比較での場所
    assert pen["last_seen"] == "床の上"  # 確認で見えた場所
    # やり直した物は回ごとの結果も（最初は掴めず、やり直しでは場所が分からなかった）
    assert pen["history"] == [{"round": 1, "result": "掴めなかった"}, {"round": 2, "result": "場所が分からなかった"}]
    assert "history" not in cup
    assert facts["outcome"] == "一部の物を戻せなかった"
    assert facts["verify"] == {"remaining": [PEN, REMOTE]}
    assert facts["ignored"] == ["謎の物体: 物体の一覧にない名前"]


def test_facts_have_no_ground_truth():
    text = json.dumps(report_facts(partial_log(), SCENE), ensure_ascii=False)
    for word in ("error_m", "estimate", "final_position", "0.0012", "seed"):
        assert word not in text


def test_prompt_contains_facts_and_rules():
    facts = report_facts(partial_log(), SCENE)
    prompt = report_prompt(facts)
    assert CUP in prompt and PEN in prompt and "場所が分からなかった" in prompt
    assert "記録にないこと" in prompt  # 推測で書かない、という指示
    assert "error_m" not in prompt


def test_template_report_partial():
    text = template_report(report_facts(partial_log(), SCENE))
    assert "3 個" in text
    assert CUP in text and REMOTE in text and PEN in text
    assert "やり直して" in text  # リモコンはやり直して戻した
    assert "場所が分からなかった" in text


def test_template_report_all_returned_and_nothing_moved():
    log = make_log([_task("cup", "succeeded")], [_move(CUP, "机の左端", "机の左寄り")])
    text = template_report(report_facts(log, SCENE))
    assert "すべて" in text and CUP in text
    empty = template_report(report_facts(make_log([], []), SCENE))
    assert "動いている物はありませんでした" in empty


def test_facts_when_vlm_failed():
    log = make_log([], [], status="failed")
    log["detection"]["ok"] = False
    log["reason"] = "VLM の問い合わせに失敗"
    facts = report_facts(log, SCENE)
    assert facts["outcome"] == "写真を比べられなかった"
    assert "比べられません" in template_report(facts)


def test_write_report_uses_writer_and_cost():
    writer = DummyVlmClient(
        [
            VlmResult(
                text=" 3 個のうち 2 個を戻しました。\n",
                input_tokens=10,
                output_tokens=5,
                latency_s=1.0,
                model="m",
                cost_usd=0.004,
            )
        ]
    )
    facts = report_facts(partial_log(), SCENE)
    r = write_report(facts, writer)
    assert r.text == "3 個のうち 2 個を戻しました。"
    assert r.source == "m" and r.cost_usd == 0.004 and r.error is None
    assert writer.prompts == [report_prompt(facts)]


def test_write_report_falls_back_to_template():
    facts = report_facts(partial_log(), SCENE)
    assert write_report(facts, None).source == "template"
    r = write_report(facts, DummyVlmClient([VlmApiError("APIConnectionError")]))
    assert r.source == "template" and r.text == template_report(facts) and "APIConnectionError" in r.error
    refused = VlmResult(
        text="", input_tokens=1, output_tokens=0, latency_s=0.1, model="m", refused=True, cost_usd=0.001
    )
    r = write_report(facts, DummyVlmClient([refused]))
    assert r.source == "template" and r.cost_usd == 0.001 and r.error
