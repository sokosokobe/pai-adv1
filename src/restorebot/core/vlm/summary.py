"""報告文（設計書 11 章。#24）。

1 回の実行の記録（log.json の dict）から、片付けを頼んだ人に向けた短い日本語の報告文を作る。
VLM（文章だけの問い合わせ）に書かせ、API を呼ばないとき・失敗したときは決まった型の文（テンプレート）にする。

報告に入れるのはロボットが知っていることだけ（VLM の答え、タスクの結果）。シミュレータの真値は入れない。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from restorebot.core.vlm.client import VlmApiError, VlmResult

# タスクの失敗の理由（ロボットの内部の言い方）に含まれる言葉
_NEAR_WORDS = ("止め位置がない", "経路がない", "届かない", "スタック")  # 台車で近づけない・運べない
_GRASP_WORDS = ("掴み損ね", "approach", "descend", "grasp", "lift")  # 掴む段階の失敗


class ReportWriter(Protocol):
    name: str

    def write(self, prompt: str) -> VlmResult:
        """文章だけを渡して、文章を書かせる。"""
        ...


@dataclass(frozen=True)
class ReportText:
    text: str
    source: str  # 書いたモデルの名前、またはテンプレート（"template"）
    cost_usd: float | None = None
    error: str | None = None  # VLM に書かせるのに失敗した理由（テンプレートにした場合）


def reason_category(status: str, reason: str | None) -> str:
    """タスクの結果 → 報告の言葉（戻した・掴めなかった・場所が分からなかった など）。"""
    if status == "succeeded":
        return "戻した"
    text = reason or ""
    if "位置が分からない" in text:
        return "場所が分からなかった"
    if "落とした" in text:  # 指の開きで、持ち上げた後・運ぶ途中に落としたと分かった（#27）
        return "落とした"
    if status == "failed":  # 掴んだ後の失敗：運ぶ途中か、置くとき
        return "運べなかった" if any(w in text for w in _NEAR_WORDS) else "置けなかった"
    if any(w in text for w in _NEAR_WORDS):
        return "近づけなかった"
    if any(w in text for w in _GRASP_WORDS):
        return "掴めなかった"
    return "戻せなかった"


def report_facts(log: dict, scene) -> dict:
    """記録 → 報告文の材料（ロボットが知っていることだけ）。物体の番号は、片付けた順（タスクの順）。"""
    names = {o.key: o.name for o in scene.objects}
    first_moves = {m["name"]: m for m in (log.get("detection") or {}).get("moves", [])}
    verify_det = ((log.get("verify") or {}).get("detection")) or {}
    verify_moves = {m["name"]: m for m in verify_det.get("moves", [])}

    order: list[str] = []
    records: dict[str, list[dict]] = {}
    for t in log.get("tasks", []):
        if t["key"] not in records:
            order.append(t["key"])
            records[t["key"]] = []
        records[t["key"]].append(t)

    objects = []
    for no, key in enumerate(order, start=1):
        name = names.get(key, key)
        recs = records[key]
        last = recs[-1]
        move = first_moves.get(name) or verify_moves.get(name) or {}
        obj = {
            "no": no,
            "name": name,
            "from": move.get("current_location", ""),
            "to": move.get("target_location", ""),
            "result": reason_category(last["status"], last["reason"]),
            "detail": last["reason"],
            "retried": any(r.get("round", 1) == 2 for r in recs),
            "pick_attempts": sum(r.get("pick_attempts", 0) for r in recs),
            "duration_s": round(sum(r.get("duration_s", 0.0) for r in recs), 1),
        }
        if len(recs) > 1:  # やり直した物：回ごとの結果（最後の結果だけでは、途中で何が起きたか分からない）
            obj["history"] = [
                {"round": r.get("round", 1), "result": reason_category(r["status"], r["reason"])} for r in recs
            ]
        if name in verify_moves and first_moves.get(name) is not None:
            obj["last_seen"] = verify_moves[name].get("current_location", "")
        objects.append(obj)

    detection_ok = (log.get("detection") or {}).get("ok", False)
    returned = sum(o["result"] == "戻した" for o in objects)
    if not detection_ok:
        outcome = "写真を比べられなかった"
    elif not objects:
        outcome = "動いている物はなかった"
    elif returned == len(objects):
        outcome = "すべて戻した"
    elif returned > 0:
        outcome = "一部の物を戻せなかった"
    else:
        outcome = "戻せなかった"
    facts: dict = {
        "outcome": outcome,
        "found": len(objects),
        "returned": returned,
        "objects": objects,
        "time_min": round(log.get("sim_time_s", 0.0) / 60.0, 1),
    }
    if log.get("verify") is not None and verify_det.get("ok"):
        facts["verify"] = {"remaining": [m["name"] for m in verify_det.get("moves", [])]}
    ignored = list(log.get("dropped", [])) + list((log.get("verify") or {}).get("dropped", []))
    if ignored:
        facts["ignored"] = ignored
    return facts


PROMPT_TEMPLATE = """\
あなたは、部屋を「いつもの状態」に戻すロボットの報告係です。
次の JSON は、今回の片付けの記録です（ロボットが知っていることだけ）。
- objects：VLM が見つけた、いつもの場所から動いていた物。no は片付けた順。from は見つけた場所、to は戻し先
  （どちらも VLM が写真を見て答えた言葉）。result は最終的な結果、detail はロボットの内部の失敗の理由。
  retried は、片付けた後の確認でまだ戻っていなかったため、もう一度やり直したこと。history は回ごとの結果
  （round 1 が最初の片付け、round 2 がやり直し）。last_seen は確認のときに見えた場所
- verify.remaining：片付けた後の確認で、まだいつもの場所にないと判断した物（この後にやり直している）
- ignored：VLM の答えのうち、使えなかったもの

```json
{facts}
```

これをもとに、片付けを頼んだ人（外出先でスマートフォンで読む）への報告文を、日本語で 2〜5 文で書いてください。
- です・ます調。最初の 1 文で全体の結果を伝える
- 戻せなかった物は、どこにあるかと理由を、分かりやすい言葉で伝える。内部の用語や数値（秒、m、rad、段階の名前）は使わない
- やり直して戻せた物があれば、そう書く
- 記録にないことは書かない（原因を推測で付け加えない。お詫びや提案もしない）
- 報告文だけを出力する（見出し・箇条書き・前置きは書かない）
"""


def report_prompt(facts: dict) -> str:
    return PROMPT_TEMPLATE.format(facts=json.dumps(facts, ensure_ascii=False, indent=2))


def template_report(facts: dict) -> str:
    """VLM を使わない報告文（決まった型）。"""
    outcome = facts["outcome"]
    if outcome == "写真を比べられなかった":
        return "いつもの写真と今の写真を比べられませんでした。片付けはしていません。"
    if outcome == "動いている物はなかった":
        return "いつもの写真と比べて、動いている物はありませんでした。"
    objects = facts["objects"]
    done = [o for o in objects if o["result"] == "戻した"]
    sentences = [f"いつもの場所から動いていた物が {facts['found']} 個ありました。"]
    if len(done) == len(objects):
        sentences.append(f"{'、'.join(o['name'] for o in done)}を、すべて元の場所に戻しました。")
    elif done:
        sentences.append(f"{'、'.join(o['name'] for o in done)}は元の場所に戻しました。")
    retried = [o for o in done if o["retried"]]
    if retried:
        sentences.append(f"{'、'.join(o['name'] for o in retried)}は、確認の後にやり直して戻しました。")
    for o in objects:
        if o["result"] != "戻した":
            where = o.get("last_seen") or o["from"]
            place = f"（{where}）" if where else ""
            sentences.append(f"{o['name']}{place}は、{o['result']}ため戻せませんでした。")
    return "".join(sentences)


def write_report(facts: dict, writer: ReportWriter | None = None) -> ReportText:
    """報告文を書く。writer が None・API の失敗・断られた・空のときはテンプレートの文にする。"""
    if writer is None:
        return ReportText(template_report(facts), "template", 0.0)
    try:
        r = writer.write(report_prompt(facts))
    except VlmApiError as e:
        return ReportText(template_report(facts), "template", None, error=str(e))
    text = r.text.strip()
    if r.refused or not text:
        reason = "断られた" if r.refused else "応答が空"
        return ReportText(template_report(facts), "template", r.cost_usd, error=f"{r.model}：{reason}")
    return ReportText(text, r.model, r.cost_usd)
