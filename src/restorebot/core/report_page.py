"""報告の画面（1 ファイルの HTML。設計書 11 章。#24）。

保存済みの 1 回の実行の記録（log.json と写真）から作る。片付けを頼んだ人が、何を見つけて何をしたかを一目で分かるように、
写真（いつもの・片付ける前・片付けた後）に VLM の答え（今の位置 → 戻し先）を番号で重ね、物体ごとの結果を表にする。

画面に出すのはロボットが知っていることだけ。シミュレータの真値（result.json）は、末尾の折りたたみの「開発用」の欄に分ける。
"""

from __future__ import annotations

import base64
import html
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from restorebot.core.vlm.summary import ReportText, report_facts, write_report

# 番号ごとの色（RGB）。灰色の床・茶色の天板の上で見分けやすい色
PALETTE: list[tuple[int, int, int]] = [
    (230, 25, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (60, 180, 75),
    (240, 50, 230),
]
PHOTO_WIDTH = 800  # 埋め込む写真の幅 [px]
RING_RADIUS = 40  # 戻し先の輪の半径（元の画像の px）
CURRENT_RADIUS = 26  # 今の位置の輪の半径（元の画像の px）
LINE_WIDTH = 6

PHOTO_CAPTIONS = {
    "reference": "いつもの写真",
    "current": "片付ける前（見つけた物 → 戻し先）",
    "after": "片付けた後",
    "final": "やり直した後",
}


@dataclass(frozen=True)
class Mark:
    """写真に重ねる印。current_px に番号の丸、target_px に輪、両方あれば矢印。座標は元の画像の (u, v) [px]。"""

    no: int
    current_px: tuple[float, float] | None
    target_px: tuple[float, float] | None


def mark_color(no: int) -> tuple[int, int, int]:
    return PALETTE[(no - 1) % len(PALETTE)]


def draw_marks(rgb: np.ndarray, marks: Sequence[Mark]) -> np.ndarray:
    """rgb（変えない）に印を描いた画像を返す。"""
    img = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=32)
    r = RING_RADIUS
    for m in marks:
        color = mark_color(m.no)
        if m.target_px is not None:
            tx, ty = m.target_px
            draw.ellipse((tx - r, ty - r, tx + r, ty + r), outline=color, width=LINE_WIDTH)
        if m.current_px is not None and m.target_px is not None:
            (cx, cy), (tx, ty) = m.current_px, m.target_px
            d = math.hypot(tx - cx, ty - cy)
            if d > r + CURRENT_RADIUS + 30:
                ux, uy = (tx - cx) / d, (ty - cy) / d
                sx, sy = cx + ux * CURRENT_RADIUS, cy + uy * CURRENT_RADIUS  # 矢印の元は今の位置の輪の縁
                ex, ey = tx - ux * r, ty - uy * r  # 矢印の先は戻し先の輪の縁
                bx, by = ex - ux * 28, ey - uy * 28
                draw.line((sx, sy, bx, by), fill=color, width=LINE_WIDTH)
                draw.polygon([(ex, ey), (bx - uy * 14, by + ux * 14), (bx + uy * 14, by - ux * 14)], fill=color)
        if m.current_px is not None:
            # 今の位置は細い輪で囲み（物体が隠れないように）、番号の丸は左上に添える
            cx, cy = m.current_px
            q = CURRENT_RADIUS
            draw.ellipse((cx - q, cy - q, cx + q, cy + q), outline=color, width=4)
            bx, by = cx - q - 14, cy - q - 14
            draw.ellipse((bx - 22, by - 22, bx + 22, by + 22), fill=color, outline=(255, 255, 255), width=3)
            draw.text((bx, by), str(m.no), fill=(255, 255, 255), font=font, anchor="mm")
        elif m.target_px is not None:  # 今の位置がない（戻し先だけ）：番号は輪の右上
            tx, ty = m.target_px
            draw.text((tx + r, ty - r), str(m.no), fill=color, font=font, anchor="lb", stroke_width=3,
                      stroke_fill=(255, 255, 255))  # fmt: skip
    return np.asarray(img)


def _jpeg_data_uri(rgb: np.ndarray, width: int = PHOTO_WIDTH) -> str:
    img = Image.fromarray(rgb).convert("RGB")
    img = img.resize((width, round(img.height * width / img.width)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def total_cost_usd(log: dict, report: ReportText) -> float | None:
    """VLM の料金の合計（比較・確認・報告文）。分からないものがあれば None。"""
    if log.get("cost_usd") is None or report.cost_usd is None:
        return None
    return log["cost_usd"] + report.cost_usd


def _marks_for(moves: Sequence[dict], numbers: Mapping[str, int], *, arrows: bool) -> list[Mark]:
    marks = []
    for m in moves:
        no = numbers.get(m["name"])
        if no is None:  # 物体の一覧にない名前など：使っていない答えは描かない
            continue
        cur = tuple(m["current_px"]) if arrows and m.get("current_px") else None
        tgt = tuple(m["target_px"]) if m.get("target_px") else None
        marks.append(Mark(no, cur, tgt))
    return marks


def photo_marks(log: dict, numbers: Mapping[str, int]) -> dict[str, list[Mark]]:
    """写真ごとの印。片付ける前は最初の比較の答え（矢印）、片付けた後は戻し先の輪と、確認で残っていた物の矢印。"""
    first = (log.get("detection") or {}).get("moves", [])
    remaining = (((log.get("verify") or {}).get("detection")) or {}).get("moves", [])
    targets = _marks_for(first, numbers, arrows=False)
    left = {m.no for m in _marks_for(remaining, numbers, arrows=True)}
    return {
        "reference": [],
        "current": _marks_for(first, numbers, arrows=True),
        "after": [t for t in targets if t.no not in left] + _marks_for(remaining, numbers, arrows=True),
        "final": targets,
    }


_CSS = """
:root { --bg:#f6f6f4; --card:#ffffff; --text:#1d1d1f; --muted:#6b6b70; --line:#e2e2e0;
  --ok:#1f7a3d; --ok-bg:#e3f4e8; --ng:#b3261e; --ng-bg:#fbe5e3; --warn:#8a5a00; --warn-bg:#fdf0d5; }
@media (prefers-color-scheme: dark) { :root { --bg:#161618; --card:#222225; --text:#ececee; --muted:#a0a0a8;
  --line:#38383c; --ok:#7fd69a; --ok-bg:#1d3a26; --ng:#f2a49d; --ng-bg:#43201d; --warn:#f0c46b; --warn-bg:#3d3015; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
  font-family:"Hiragino Sans","Yu Gothic UI","Meiryo",system-ui,sans-serif; line-height:1.7; }
main { max-width:1080px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:1.5rem; margin:0 0 4px; } h2 { font-size:1.1rem; margin:0 0 12px; }
.meta { color:var(--muted); font-size:.9rem; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px 20px; margin-top:16px; }
.report p { font-size:1.15rem; margin:0 0 6px; } .report .by { color:var(--muted); font-size:.8rem; margin-top:8px; }
.pill { display:inline-block; padding:1px 10px; border-radius:999px; font-size:.85rem; font-weight:600; }
.ok { color:var(--ok); background:var(--ok-bg); } .ng { color:var(--ng); background:var(--ng-bg); }
.warn { color:var(--warn); background:var(--warn-bg); }
.photos { display:grid; grid-template-columns:repeat(auto-fit, minmax(min(420px, 100%), 1fr)); gap:16px; }
figure { margin:0; } figure img { width:100%; height:auto; border-radius:8px; display:block; }
figcaption { color:var(--muted); font-size:.9rem; margin-top:4px; }
.table-wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:.95rem; }
th, td { text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--muted); font-weight:600; font-size:.85rem; white-space:nowrap; }
td.num { text-align:right; white-space:nowrap; }
.no { display:inline-block; width:1.7em; height:1.7em; line-height:1.7em; text-align:center; border-radius:50%;
  color:#fff; font-weight:700; font-size:.85rem; }
.detail { color:var(--muted); font-size:.8rem; }
details summary { cursor:pointer; color:var(--muted); }
"""


def _pill(result: str) -> str:
    cls = "ok" if result == "戻した" else "ng"
    return f'<span class="pill {cls}">{html.escape(result)}</span>'


def _outcome_pill(outcome: str) -> str:
    cls = {"すべて戻した": "ok", "動いている物はなかった": "ok", "一部の物を戻せなかった": "warn"}.get(outcome, "ng")
    return f'<span class="pill {cls}">{html.escape(outcome)}</span>'


def build_html(
    log: dict,
    photos: Mapping[str, np.ndarray],
    scene,
    report: ReportText,
    *,
    result: dict | None = None,
    title: str = "片付けの報告",
) -> str:
    """報告の HTML（画像を埋め込んだ 1 ファイル）。result（真値での結果）があれば開発用の欄に載せる。"""
    esc = html.escape
    facts = report_facts(log, scene)
    numbers = {o["name"]: o["no"] for o in facts["objects"]}
    marks = photo_marks(log, numbers)
    cost = total_cost_usd(log, report)
    cost_text = "不明" if cost is None else f"${cost:.3f}"
    by = "決まった型の文（VLM を使っていない）" if report.source == "template" else f"文：{esc(report.source)}"

    paragraphs = "".join(f"<p>{esc(line)}</p>" for line in report.text.splitlines() if line.strip())
    figures = []
    for name in ("reference", "current", "after", "final"):
        if name in photos:
            src = _jpeg_data_uri(draw_marks(photos[name], marks.get(name, [])))
            cap = PHOTO_CAPTIONS[name]
            figures.append(f'<figure><img src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>')

    rows = []
    for o in facts["objects"]:
        r, g, b = mark_color(o["no"])
        detail = esc(o["detail"]) if o["detail"] else ""
        notes = []
        if o["retried"]:
            notes.append("確認の後にやり直し")
        if o.get("last_seen") and o["result"] != "戻した":
            notes.append(f"確認のときの場所：{esc(o['last_seen'])}")
        note_html = "".join(f"<div>{n}</div>" for n in notes)
        rows.append(
            '<tr class="task">'
            f'<td><span class="no" style="background:rgb({r},{g},{b})">{o["no"]}</span></td>'
            f"<td>{esc(o['name'])}</td><td>{esc(o['from'])}</td><td>{esc(o['to'])}</td>"
            f'<td>{_pill(o["result"])}{note_html}<div class="detail">{detail}</div></td>'
            f'<td class="num">{o["pick_attempts"]}</td><td class="num">{o["duration_s"]:.0f} 秒</td>'
            "</tr>"
        )
    table = (
        '<div class="table-wrap"><table><thead><tr><th></th><th>物体</th><th>見つけた場所</th><th>戻し先</th>'
        "<th>結果</th><th>掴んだ回数</th><th>時間</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        if rows
        else "<p>いつもの場所から動いている物はありませんでした。</p>"
    )
    ignored = ""
    if facts.get("ignored"):
        items = "".join(f"<li>{esc(s)}</li>" for s in facts["ignored"])
        ignored = f'<section class="card"><h2>使わなかった VLM の答え</h2><ul>{items}</ul></section>'

    parts = [
        f'<!doctype html><html lang="ja"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{esc(title)}</title><style>{_CSS}</style></head><body><main>",
        f"<header><h1>{esc(title)}</h1>"
        f'<div class="meta">{_outcome_pill(facts["outcome"])} 見つけた物 {facts["found"]} 個・戻した物 '
        f"{facts['returned']} 個 ／ 所要時間 {facts['time_min']} 分 ／ VLM の費用 {cost_text}</div></header>",
        f'<section class="card report">{paragraphs}<div class="by">{by}</div></section>',
        f'<section class="card"><h2>写真</h2><div class="photos">{"".join(figures)}</div>'
        '<p class="meta">番号を添えた細い輪は見つけた物の位置、太い輪は戻し先（どちらも VLM の答え）。</p></section>',
        f'<section class="card"><h2>物体ごとの結果</h2>{table}</section>',
        ignored,
    ]
    parts.append(_dev_section(log, scene, report, result))
    parts.append("</main></body></html>")
    return "\n".join(p for p in parts if p)


def _dev_section(log: dict, scene, report: ReportText, result: dict | None) -> str:
    """開発用の欄（シミュレータの真値、実行の条件）。result がなければ出さない。"""
    if result is None:
        return ""
    esc = html.escape
    names = {o.key: o.name for o in scene.objects}
    yaw = result.get("home_yaw_err_deg", {})
    rows = "".join(
        f"<tr><td>{esc(names.get(k, k))}</td><td class='num'>{v * 1000:.1f} mm</td>"
        f"<td class='num'>{'' if k not in yaw else f'{yaw[k]:.1f}°'}</td></tr>"
        for k, v in result.get("home_err_m", {}).items()
    )
    meta = {k: log.get("meta", {}).get(k) for k in ("seed", "vlm", "perception", "milestone")}
    extra = {
        "all_returned": result.get("all_returned"),
        "min_clearance_m": result.get("min_clearance_m"),
        "untouched_moved_m": result.get("untouched_moved_m"),
        "report_error": report.error,
    }
    return (
        '<section class="card"><details><summary>開発用（シミュレータの真値）</summary>'
        "<div class='table-wrap'><table><thead><tr><th>物体</th><th>戻し先からの距離</th><th>向きの差</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
        f"<pre>{esc(json.dumps({**meta, **extra}, ensure_ascii=False, indent=2, default=str))}</pre>"
        "</details></section>"
    )


def load_run(run_dir: Path) -> tuple[dict, dict[str, np.ndarray], dict | None]:
    """保存済みの実行のフォルダ → (log, 写真の名前 → RGB, result.json の中身（なければ None))。"""
    run_dir = Path(run_dir)
    log = json.loads((run_dir / "log.json").read_text(encoding="utf-8"))
    photos = {}
    for name in log.get("photos", []):
        path = run_dir / f"{name}_rgb.png"
        if path.exists():
            photos[name] = np.asarray(Image.open(path).convert("RGB"))
    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
    return log, photos, result


def make_report(run_dir: Path, scene, writer=None) -> tuple[Path, ReportText]:
    """保存済みの実行のフォルダから報告文を書き、report.html・report.json を保存する。

    writer（文章だけの問い合わせができる VLM）が None なら、報告文はテンプレート。
    """
    run_dir = Path(run_dir)
    log, photos, result = load_run(run_dir)
    report = write_report(report_facts(log, scene), writer)
    path = run_dir / "report.html"
    path.write_text(build_html(log, photos, scene, report, result=result), encoding="utf-8")
    (run_dir / "report.json").write_text(
        json.dumps(
            {"text": report.text, "source": report.source, "cost_usd": report.cost_usd, "error": report.error},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path, report
