"""提出用の動画を作る（設計書 12 章。#20）。Isaac Sim は起動しない。

録画した 1 回の実行（run_episode.py --record の出力）と、報告の画面・評価の集計から、字幕つきの 1 本の mp4 を作る。
説明の場面は PIL で画像を作って静止画の動画にし、ロボットが動く場面は俯瞰の映像に追従の映像を小さく重ね、
倍速にして字幕（記録のイベントから作る。core.video.episode_captions）を重ねる。最後に ffmpeg でつなぐ。

使い方:
    .venv\\Scripts\\python.exe scripts\\make_video.py --run outputs\\m4_20\\seed104 \\
        --eval outputs\\m3_18\\claude\\summary.json --out outputs\\m4_20\\video
出力：<out>/restorebot_demo.mp4（場面ごとの画像と mp4 も <out>/parts に残す）
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from restorebot.core.report_page import draw_marks, mark_color, photo_marks
from restorebot.core.scene import HOME_V2
from restorebot.core.video import episode_captions
from restorebot.core.vlm.summary import report_facts

W, H, FPS = 1280, 720, 30
SPEED = 2  # ロボットが動く場面の倍速
FONT_B = "C:/Windows/Fonts/YuGothB.ttc"
FONT_M = "C:/Windows/Fonts/YuGothM.ttc"
BG, INK, MUTED, ACCENT = (246, 246, 243), (30, 30, 34), (105, 105, 112), (0, 110, 190)
SCENE = HOME_V2


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_B if bold else FONT_M, size)


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ffmpeg(*args: str) -> None:
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args])


ENCODE = ["-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p", "-r", str(FPS)]


# ---------------------------------------------------------------- 画像の部品


def canvas(title: str | None = None, step: str | None = None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    if title:
        if step:
            d.rounded_rectangle((56, 44, 56 + 46, 44 + 46), radius=23, fill=ACCENT)
            d.text((56 + 23, 44 + 23), step, font=font(26, True), fill="white", anchor="mm")
            d.text((120, 67), title, font=font(38, True), fill=INK, anchor="lm")
        else:
            d.text((56, 67), title, font=font(38, True), fill=INK, anchor="lm")
    return img, d


def paste_fit(img: Image.Image, photo: np.ndarray | Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int]:
    """photo を box（左, 上, 幅, 高さ）に収まるよう縮めて貼る。貼った左上を返す。"""
    p = photo if isinstance(photo, Image.Image) else Image.fromarray(photo)
    x, y, bw, bh = box
    s = min(bw / p.width, bh / p.height)
    p = p.resize((round(p.width * s), round(p.height * s)), Image.Resampling.LANCZOS)
    ox, oy = x + (bw - p.width) // 2, y + (bh - p.height) // 2
    img.paste(p, (ox, oy))
    return ox, oy


def wrap(d: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, width: int) -> list[str]:
    """日本語の文を、幅 width [px] に収まるよう 1 文字ずつ折り返す。"""
    lines, cur = [], ""
    for ch in text:
        if d.textlength(cur + ch, font=f) > width:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    return lines + ([cur] if cur else [])


def text_block(d, xy, text: str, size: int, width: int, *, fill=INK, bold=False, spacing=1.55) -> int:
    """折り返して書き、次の行の y を返す。"""
    f = font(size, bold)
    x, y = xy
    for para in text.split("\n"):
        for line in wrap(d, para, f, width) or [""]:
            d.text((x, y), line, font=f, fill=fill)
            y += round(size * spacing)
    return y


def caption_overlay(text: str, path: Path) -> None:
    """画面の下の字幕（透明な全画面の PNG）。"""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f = font(34, True)
    tw = d.textlength(text, font=f)
    x0, y0 = (W - tw) / 2 - 28, H - 96
    d.rounded_rectangle((x0, y0, x0 + tw + 56, y0 + 64), radius=14, fill=(0, 0, 0, 170))
    d.text((W / 2, y0 + 32), text, font=f, fill="white", anchor="mm")
    img.save(path)


def badge_overlay(text: str, path: Path) -> None:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((16, 16, 16 + 190, 16 + 48), radius=10, fill=(0, 0, 0, 170))
    d.text((16 + 95, 16 + 24), text, font=font(28, True), fill="white", anchor="mm")
    img.save(path)


# ---------------------------------------------------------------- 場面


def still(img: Image.Image, seconds: float, name: str, parts: Path) -> Path:
    png, mp4 = parts / f"{name}.png", parts / f"{name}.mp4"
    img.save(png)
    ffmpeg("-loop", "1", "-t", f"{seconds}", "-i", str(png), "-vf", f"fps={FPS},format=yuv420p", *ENCODE, str(mp4))
    return mp4


def action_clip(run_dir: Path, t0: float, t1: float, captions, name: str, parts: Path) -> Path:
    """追従の映像（全画面）に俯瞰の映像を右上に小さく重ね、SPEED 倍速にして字幕を重ねる。t0〜t1 はシミュの時刻。

    俯瞰のカメラは低い棚の側が画面の端になるので、ロボットがいつも写る追従の映像を主にする。
    """
    badge = parts / f"{name}_badge.png"
    badge_overlay(f"×{SPEED} 倍速", badge)
    inputs = ["-i", str(run_dir / "follow.mp4"), "-i", str(run_dir / "overview.mp4"), "-i", str(badge)]
    windows = []
    for i, c in enumerate(captions):
        a, b = max(c.t0, t0), min(c.t1, t1)
        if b - a < 0.05:
            continue
        png = parts / f"{name}_cap{i}.png"
        caption_overlay(c.text, png)
        inputs += ["-i", str(png)]
        windows.append(((a - t0) / SPEED, (b - t0) / SPEED))
    trim = f"trim=start={t0:.3f}:end={t1:.3f},setpts=(PTS-STARTPTS)/{SPEED}"
    fg = [
        f"[0:v]{trim}[ov]",
        f"[1:v]{trim},scale=448:252,pad=456:260:4:4:white[fl]",
        "[ov][fl]overlay=W-w-16:16[v0]",
        "[v0][2:v]overlay=0:0[v1]",
    ]
    last = "v1"
    for k, (a, b) in enumerate(windows):
        fg.append(f"[{last}][{3 + k}:v]overlay=0:0:enable='between(t,{a:.3f},{b:.3f})'[c{k}]")
        last = f"c{k}"
    mp4 = parts / f"{name}.mp4"
    ffmpeg(*inputs, "-filter_complex", ";".join(fg), "-map", f"[{last}]", *ENCODE, str(mp4))
    return mp4


def photo(run_dir: Path, name: str) -> np.ndarray:
    return np.asarray(Image.open(run_dir / f"{name}_rgb.png").convert("RGB"))


def first_frame(video: Path, at: float, out: Path) -> Image.Image:
    ffmpeg("-ss", f"{at}", "-i", str(video), "-frames:v", "1", str(out))
    return Image.open(out).convert("RGB")


def report_screenshot(report_html: Path, out: Path) -> Image.Image:
    """報告の画面を Edge（headless）で撮る。開いている Edge に渡されないよう、別のプロフィールで起動する。"""
    import time

    edge = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
    out = out.resolve()
    out.unlink(missing_ok=True)
    profile = out.parent / "edge_profile"
    run([str(edge), "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1100,1100",
         f"--user-data-dir={profile}", f"--screenshot={out}", report_html.resolve().as_uri()])  # fmt: skip
    for _ in range(100):  # 書き終わるのを待つ（最大 10 秒）
        if out.exists() and out.stat().st_size > 0:
            break
        time.sleep(0.1)
    return Image.open(out).convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run_episode.py --record の出力")
    ap.add_argument("--eval", type=Path, required=True, help="run_eval.py の summary.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg が見つかりません")
    parts = args.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    log = json.loads((args.run / "log.json").read_text(encoding="utf-8"))
    summary = json.loads(args.eval.read_text(encoding="utf-8"))["summary"]
    facts = report_facts(log, SCENE)
    numbers = {o["name"]: o["no"] for o in facts["objects"]}
    first_moves = log["detection"]["moves"]
    verify_moves = (log.get("verify") or {}).get("detection", {}).get("moves", [])
    verify_t = next((e["t"] for e in log["events"] if e["kind"] == "verify"), None)
    caps = episode_captions(log, SCENE)
    clips: list[Path] = []

    # 1. タイトル
    bg = first_frame(args.run / "overview.mp4", 1.0, parts / "frame0.png")
    bg = Image.blend(bg, Image.new("RGB", bg.size, (0, 0, 0)), 0.55)
    d = ImageDraw.Draw(bg)
    d.text((W / 2, 270), "写真 1 枚で「いつもの状態」を教えると、", font=font(46, True), fill="white", anchor="mm")
    d.text((W / 2, 340), "ロボットが部屋を元に戻す", font=font(58, True), fill="white", anchor="mm")
    d.text((W / 2, 440), "Physical AI 応用講座 最終課題", font=font(28), fill=(220, 220, 220), anchor="mm")
    d.text(
        (W / 2, 485),
        "NVIDIA Isaac Sim 6.1 ／ RidgebackFranka ／ Claude Sonnet 5.5",
        font=font(24),
        fill=(220, 220, 220),
        anchor="mm",
    )
    clips.append(still(bg, 6, "01_title", parts))

    # 2. 使い方
    img, d = canvas("使い方")
    steps = [
        (
            "1",
            "いつもの状態を教える",
            "片付いた部屋を、天井のカメラで 1 枚撮っておく。置き場所のルールは書かなくてよい",
        ),
        ("2", "散らかったら頼む", "今の写真をいつもの写真と見比べ、どの物がどこへ動いたかを VLM（Claude）が答える"),
        (
            "3",
            "ロボットが戻して報告する",
            "移動マニピュレータが物を掴んで戻し、片付けた後にもう一度確かめて、結果を報告する",
        ),
    ]
    y = 150
    for n, head, body in steps:
        d.rounded_rectangle((70, y, 70 + 56, y + 56), radius=28, fill=ACCENT)
        d.text((98, y + 28), n, font=font(30, True), fill="white", anchor="mm")
        d.text((150, y + 4), head, font=font(34, True), fill=INK)
        text_block(d, (150, y + 58), body, 26, 1040, fill=MUTED)
        y += 175
    clips.append(still(img, 10, "02_usage", parts))

    # 3. 教える・散らかす
    img, d = canvas("いつもの写真と、今の写真", "1")
    for i, (name, label) in enumerate([("reference", "いつもの写真（教える）"), ("current", "今の写真（散らかった）")]):
        x = 56 + i * 600
        paste_fit(img, photo(args.run, name), (x, 120, 568, 426))
        d.text((x + 284, 575), label, font=font(30, True), fill=INK, anchor="mm")
    moved = len(log["meta"]["moved_truth"])
    note = f"天井のカメラ（真下向き、RGB と深度）。物体 5 個のうち、{moved} 個を別の場所へ動かした"
    text_block(d, (56, 620), note, 24, 1170, fill=MUTED)
    clips.append(still(img, 9, "03_photos", parts))

    # 4. VLM の答え
    img, d = canvas("VLM（Claude）が見比べた答え", "2")
    marks = photo_marks(log, numbers)["current"]
    paste_fit(img, draw_marks(photo(args.run, "current"), marks), (40, 115, 720, 540))
    y = 130
    for m in first_moves:
        no = numbers.get(m["name"])
        if no is None:
            continue
        r, g, b = mark_color(no)
        d.ellipse((790, y + 2, 790 + 36, y + 38), fill=(r, g, b))
        d.text((808, y + 20), str(no), font=font(22, True), fill="white", anchor="mm")
        d.text((840, y), m["name"], font=font(30, True), fill=INK)
        y = text_block(d, (840, y + 46), f"今：{m['current_location']}", 22, 400, fill=MUTED, spacing=1.4)
        y = text_block(d, (840, y), f"戻し先：{m['target_location']}", 22, 400, fill=MUTED, spacing=1.4) + 20
    missed = [o["name"] for o in facts["objects"] if all(m["name"] != o["name"] for m in first_moves)]
    if missed:
        text_block(
            d,
            (790, y + 10),
            f"※ 少しだけ動いた{'、'.join(missed)}は、ここでは見逃している",
            22,
            450,
            fill=(180, 60, 40),
        )
    text_block(
        d,
        (40, 668),
        "答えは物体の名前・場所の言葉・画像上の点。点の深度から、掴む位置と戻し先を 3 次元で求める",
        22,
        1200,
        fill=MUTED,
    )
    clips.append(still(img, 12, "04_vlm", parts))

    # 5. 片付け（1 回目）
    split = verify_t if verify_t is not None else log["sim_time_s"]
    clips.append(action_clip(args.run, 0.0, split, caps, "05_action1", parts))

    # 6. 確認
    if verify_t is not None and verify_moves:
        img, d = canvas("片付けた後に、もう一度見比べる", "3")
        vmarks = photo_marks(log, numbers)["after"]
        paste_fit(img, draw_marks(photo(args.run, "after"), vmarks), (40, 115, 640, 540))
        names = "、".join(m["name"] for m in verify_moves)
        y = text_block(d, (720, 140), "片付けた後の写真を、いつもの写真ともう一度比べる。", 28, 520)
        y = text_block(d, (720, y + 20), f"まだ戻っていない物：{names}", 28, 520, bold=True, fill=(180, 60, 40))
        text_block(d, (720, y + 20), "最初の比較で見逃していた物に気づき、やり直す（1 回まで）", 26, 520, fill=MUTED)
        clips.append(still(img, 9, "06_verify", parts))
        clips.append(action_clip(args.run, verify_t, log["sim_time_s"], caps, "07_action2", parts))

    # 7. 前後の比較
    img, d = canvas("いつもの写真と、片付けた後")
    last = "final" if (args.run / "final_rgb.png").exists() else "after"
    for i, (name, label) in enumerate([("reference", "いつもの写真"), (last, "片付けた後")]):
        x = 56 + i * 600
        paste_fit(img, photo(args.run, name), (x, 120, 568, 426))
        d.text((x + 284, 575), label, font=font(30, True), fill=INK, anchor="mm")
    errs = json.loads((args.run / "result.json").read_text(encoding="utf-8")).get("home_err_m", {})
    err_text = "、".join(f"{SCENE.object(k).name} {v * 1000:.1f}mm" for k, v in errs.items())
    text_block(d, (56, 620), f"戻し先からのずれ（シミュレータの真値）：{err_text}", 24, 1170, fill=MUTED)
    clips.append(still(img, 8, "08_compare", parts))

    # 8. 報告
    report_html = args.run / "report.html"
    if report_html.exists():
        img, d = canvas("結果の報告（頼んだ人へ）", "4")
        shot = report_screenshot(report_html, parts / "report_shot.png")
        paste_fit(img, shot.crop((0, 0, shot.width, min(shot.height, 900))), (40, 110, 560, 590))
        text = json.loads((args.run / "report.json").read_text(encoding="utf-8"))["text"]
        d.text((640, 120), "Claude が書いた報告文", font=font(24, True), fill=ACCENT)
        y = text_block(d, (640, 165), text, 27, 590, spacing=1.6)
        text_block(d, (640, y + 16), "写真・物体ごとの結果・費用を、1 枚の画面（HTML）にまとめる", 22, 590, fill=MUTED)
        clips.append(still(img, 10, "09_report", parts))

    # 9. 評価
    s = summary
    img, d = canvas("評価：開発に使っていない 10 エピソード")
    rows = [
        ("エピソードの成功", f"{s['episode_success']} / {s['episodes']}"),
        ("物体の成功（戻し先から 3cm 以内）", f"{s['task_success']} / {s['tasks']}"),
        (
            "戻し先からのずれ（平均・最大）",
            f"{s['err_success_mm']['mean']:.1f} mm ・ {s['err_success_mm']['max']:.1f} mm",
        ),
        ("箱の向き（最大のずれ）", f"{s['box_yaw']['max_deg']:.1f}°"),
        (
            "VLM の検出（適合率・再現率）",
            f"{s['detection']['precision']:.2f} ・ {s['detection']['recall']:.2f}",
        ),
        ("VLM の費用（1 エピソード）", f"約 ${s['cost_usd'] / s['episodes']:.2f}"),
    ]
    y = 140
    for k, v in rows:
        d.text((90, y), k, font=font(30), fill=MUTED)
        d.text((640, y), v, font=font(32, True), fill=INK)
        y += 78
        if k.startswith("VLM の検出") and s["detection"]["fn"] and not s["failures"]:
            d.text(
                (640, y - 30), f"見逃し {s['detection']['fn']} 件は、確認とやり直しで回収", font=font(22), fill=MUTED
            )
            y += 16
    text_block(
        d,
        (90, y + 10),
        "条件：動かす物体 1〜3 個（計 18 個）、Claude Sonnet 5.5、確認とやり直しあり",
        24,
        1100,
        fill=MUTED,
    )
    clips.append(still(img, 10, "10_eval", parts))

    # 10. 工夫した点
    img, d = canvas("工夫した点")
    points = [
        ("真値を使わない", "物体の位置と向きは画像から、台車の位置は天井カメラから、掴めたかは指の開きから判定する"),
        ("箱は向きも戻す", "リモコンなどの箱は、いつもの向きに回して置く（円柱は向きを気にしない）"),
        ("確認とやり直し", "片付けた後にもう一度見比べ、VLM の見逃しや置き損ねを 1 回だけやり直す"),
        ("掴み方", "隣の物を避ける手の向き、運ぶ速さ、置いた後に少し上げてから離れる"),
        ("報告", "写真と結果の表、VLM が書いた報告文を 1 枚の画面にまとめる"),
    ]
    y = 135
    for head, body in points:
        d.rectangle((70, y + 8, 78, y + 72), fill=ACCENT)
        d.text((100, y), head, font=font(30, True), fill=INK)
        text_block(d, (100, y + 44), body, 24, 1100, fill=MUTED)
        y += 108
    clips.append(still(img, 12, "11_points", parts))

    # 11. 終わり
    img, d = canvas()
    d.text((W / 2, 300), "写真 1 枚で、いつもの状態に戻す", font=font(48, True), fill=INK, anchor="mm")
    d.text((W / 2, 390), "ロボットが動く場面は 2 倍速で再生しています", font=font(28), fill=MUTED, anchor="mm")
    d.text(
        (W / 2, 440),
        "VLM の答えは、評価の実行で Claude が答えたものを記録して再生しています",
        font=font(24),
        fill=MUTED,
        anchor="mm",
    )
    clips.append(still(img, 5, "12_end", parts))

    # つなぐ
    lst = parts / "list.txt"
    lst.write_text("".join(f"file '{c.resolve().as_posix()}'\n" for c in clips), encoding="utf-8")
    out = args.out / "restorebot_demo.mp4"
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(lst), *ENCODE, "-movflags", "+faststart", str(out))
    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    print(f"[make_video] {out} {float(dur):.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
