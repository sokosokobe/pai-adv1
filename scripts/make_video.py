"""提出用の動画を作る（設計書 12 章。#20）。Isaac Sim は起動しない。

録画した 1 回の実行（run_episode.py --record の出力）と、報告の画面・評価の集計から、字幕・ナレーションつきの
1 本の mp4 を作る。説明の場面は PIL で画像を作って静止画の動画にし、ロボットが動く場面は追従の映像に俯瞰の映像を
小さく重ね、倍速にして字幕（記録のイベントから作る。core.video.episode_captions）を重ねる。最後に ffmpeg でつなぐ。

ナレーション：VOICEVOX のエンジン（http://127.0.0.1:50021）が動いていれば VOICEVOX、なければ Windows の音声合成
（Microsoft Haruka）。場面の長さは、ナレーションが収まるように延ばす。BGM は --bgm で与える（小さい音量で重ねる）。

使い方:
    .venv\\Scripts\\python.exe scripts\\make_video.py --run outputs\\m4_20\\seed104 \\
        --eval outputs\\m3_18\\claude\\summary.json --out outputs\\m4_20\\video [--bgm music.mp3] [--voice sapi]
出力：<out>/restorebot_demo.mp4（場面ごとの画像・音声・mp4 も <out>/parts に残す）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from restorebot.core.report_page import draw_marks, mark_color, photo_marks
from restorebot.core.scene import HOME_V2
from restorebot.core.video import episode_captions
from restorebot.core.vlm.parse import normalize_name
from restorebot.core.vlm.summary import report_facts

W, H, FPS = 1280, 720, 30
SPEED = 2  # ロボットが動く場面の倍速
MAX_SECONDS = 180  # 提出の動画の長さの上限
FONT_B = "C:/Windows/Fonts/YuGothB.ttc"
FONT_M = "C:/Windows/Fonts/YuGothM.ttc"
BG, INK, MUTED, ACCENT, WARN = (246, 246, 243), (30, 30, 34), (105, 105, 112), (0, 110, 190), (180, 60, 40)
SCENE = HOME_V2
# 物体は日用品に見立てた単色の円柱・直方体なので、動画では見た目で呼ぶ（VLM には日用品の名前の一覧を渡している）
DISPLAY = {
    "cup": "赤い円柱",
    "pen_stand": "青い円柱",
    "spray": "緑の円柱",
    "remote": "黒い直方体",
    "small_box": "黄色い直方体",
}
KEY_BY_NAME = {normalize_name(o.name): o.key for o in SCENE.objects}
NO_BREAK_START = set("、。，．）」』！？・ー")  # 行の頭に来てはいけない文字
BREAK_AFTER = set("、。，．）」』！？ ")  # この文字の後で折り返してよい


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_B if bold else FONT_M, size)


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ffmpeg(*args: str) -> None:
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args])


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return float(out)


VIDEO = ["-c:v", "libx264", "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p", "-r", str(FPS)]
AUDIO = ["-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]


def display_name(name: str) -> str:
    """VLM の答えの名前（「黒いリモコン」など）→ 動画での呼び方（「黒い直方体」など）。"""
    key = KEY_BY_NAME.get(normalize_name(name))
    return DISPLAY.get(key, name)


# ---------------------------------------------------------------- 音声


class Voice:
    """ナレーションの音声合成。VOICEVOX（エンジンが動いていれば）か、Windows の音声合成（SAPI）。"""

    VOICEVOX = "http://127.0.0.1:50021"

    def __init__(self, engine: str, speaker: int, cache: Path) -> None:
        if engine == "auto":
            engine = "voicevox" if self._voicevox_alive() else "sapi"
        self.engine, self.speaker, self.cache = engine, speaker, cache
        cache.mkdir(parents=True, exist_ok=True)
        self.credit = None
        if engine == "voicevox":
            self.credit = f"VOICEVOX:{self._speaker_name()}"

    def _voicevox_alive(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.VOICEVOX}/version", timeout=2)
            return True
        except OSError:
            return False

    def _speaker_name(self) -> str:
        speakers = json.loads(urllib.request.urlopen(f"{self.VOICEVOX}/speakers", timeout=5).read())
        for sp in speakers:
            if any(st["id"] == self.speaker for st in sp["styles"]):
                return sp["name"]
        return str(self.speaker)

    def synth(self, text: str) -> Path:
        """text を読み上げた wav（同じ文・同じ声なら作り直さない）。"""
        h = hashlib.sha1(f"{self.engine}:{self.speaker}:{text}".encode()).hexdigest()[:12]
        wav = self.cache / f"{h}.wav"
        if wav.exists():
            return wav
        if self.engine == "voicevox":
            q = urllib.parse.urlencode({"text": text, "speaker": self.speaker})
            query = urllib.request.urlopen(urllib.request.Request(f"{self.VOICEVOX}/audio_query?{q}", method="POST"))
            body = json.loads(query.read())
            body["speedScale"] = 1.1
            req = urllib.request.Request(
                f"{self.VOICEVOX}/synthesis?speaker={self.speaker}",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            wav.write_bytes(urllib.request.urlopen(req).read())
        else:
            txt = self.cache / f"{h}.txt"
            txt.write_text(text, encoding="utf-8")
            ps = (
                "Add-Type -AssemblyName System.Speech;"
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
                "$s.SelectVoice('Microsoft Haruka Desktop'); $s.Rate = 1;"
                f"$s.SetOutputToWaveFile('{wav.resolve()}');"
                f"$s.Speak([IO.File]::ReadAllText('{txt.resolve()}', [Text.Encoding]::UTF8)); $s.Dispose()"
            )
            run(["powershell", "-NoProfile", "-Command", ps])
        return wav


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


def _kind(ch: str) -> str:
    if "ぁ" <= ch <= "ゟ":
        return "hira"
    if "゠" <= ch <= "ヿ" or "一" <= ch <= "鿿" or ch.isalnum() or ch in "「（":
        return "word"  # 漢字・カタカナ・英数字・開き括弧（内容語の頭になりやすい）
    return "other"


PARTICLE_END = set("のはがをにでとへもやたら")  # 文節の終わりになりやすいひらがな（助詞・過去の「た」など）


def _break_rank(text: str, i: int) -> int:
    """text[:i] と text[i:] の間の折り返しやすさ。

    2：句読点・空白・「|」の後、1：助詞などの後で漢字などが続く所、0：折り返さない。
    """
    if i <= 0 or i >= len(text) or text[i] in NO_BREAK_START:
        return 0
    prev = text[i - 1]
    if prev in BREAK_AFTER or prev == "|":
        return 2
    return 1 if prev in PARTICLE_END and _kind(text[i]) == "word" else 0


def wrap(d: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, width: int) -> list[str]:
    """日本語の文を幅 width [px] で折り返す。

    句読点・空白・「|」の後を優先し、なければ文節の切れ目らしい所（助詞などの後で漢字・カタカナが続く所）で
    折り返す。「|」は書かない（折り返してよい所の目印）。行の 4 割より前に折り返せる所がなければ、1 文字ずつで折り返す。
    句読点などは行の頭に置かない。
    """
    lines: list[str] = []
    rest = text
    while rest:
        n = len(rest)
        if d.textlength(rest.replace("|", ""), font=f) <= width:
            lines.append(rest)
            break
        fit = 1  # 幅に収まる最長の文字数
        while fit < n and d.textlength(rest[: fit + 1].replace("|", ""), font=f) <= width:
            fit += 1
        while fit < n and rest[fit] in NO_BREAK_START:  # 句読点は前の行に入れる（少しはみ出してよい）
            fit += 1
        cut = 0
        for rank in (2, 1):  # 句読点・「|」を優先し、なければ文節の切れ目らしい所
            cut = max((i for i in range(1, fit + 1) if _break_rank(rest, i) >= rank), default=0)
            if cut >= 0.4 * fit:
                break
        if cut < 0.4 * fit:
            cut = fit
        lines.append(rest[:cut])
        rest = rest[cut:]
    return [line.replace("|", "").strip() for line in lines]


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


def browser_frame(shot: Image.Image, filename: str, width: int) -> Image.Image:
    """画面の写しを、ブラウザの窓（ファイル名つきのアドレス欄）に入れる。"""
    s = width / shot.width
    body = shot.resize((width, round(shot.height * s)), Image.Resampling.LANCZOS)
    bar = 40
    out = Image.new("RGB", (width + 4, body.height + bar + 4), (200, 200, 205))
    d = ImageDraw.Draw(out)
    d.rectangle((2, 2, width + 1, bar), fill=(232, 232, 236))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse((14 + i * 20, 15, 26 + i * 20, 27), fill=c)
    d.rounded_rectangle((84, 8, width - 12, bar - 6), radius=8, fill="white")
    d.text((100, bar / 2 + 1), filename, font=font(18), fill=INK, anchor="lm")
    out.paste(body, (2, bar + 2))
    return out


# ---------------------------------------------------------------- 場面


class Clips:
    def __init__(self, parts: Path, voice: Voice | None) -> None:
        self.parts, self.voice, self.paths = parts, voice, []

    def _audio_args(self, narration: str | None, seconds: float) -> tuple[list[str], str]:
        """ナレーション（0.5 秒遅らせ、seconds まで無音で埋める）か無音の入力と、音声のフィルタ。"""
        if narration and self.voice is not None:
            wav = self.voice.synth(narration)
            return ["-i", str(wav)], (
                f"adelay=500:all=1,apad,atrim=0:{seconds:.3f},aformat=sample_rates=48000:channel_layouts=stereo"
            )
        return ["-f", "lavfi", "-t", f"{seconds:.3f}", "-i", "anullsrc=r=48000:cl=stereo"], "anull"

    def narration_seconds(self, narration: str | None) -> float:
        if not narration or self.voice is None:
            return 0.0
        return duration(self.voice.synth(narration)) + 1.2

    def still(self, img: Image.Image, seconds: float, name: str, narration: str | None = None) -> None:
        seconds = max(seconds, self.narration_seconds(narration))
        png, mp4 = self.parts / f"{name}.png", self.parts / f"{name}.mp4"
        img.save(png)
        a_in, a_f = self._audio_args(narration, seconds)
        ffmpeg(
            "-loop", "1", "-t", f"{seconds:.3f}", "-i", str(png), *a_in,
            "-filter_complex", f"[0:v]fps={FPS},format=yuv420p[v];[1:a]{a_f}[a]",
            "-map", "[v]", "-map", "[a]", "-t", f"{seconds:.3f}", *VIDEO, *AUDIO, str(mp4),
        )  # fmt: skip
        self.paths.append(mp4)

    def action(self, run_dir: Path, t0: float, t1: float, captions, name: str, narration: str | None = None) -> None:
        """追従の映像（全画面）に俯瞰の映像を右上に小さく重ね、SPEED 倍速にして字幕を重ねる。t0〜t1 はシミュの時刻。

        俯瞰のカメラは低い棚の側が画面の端になるので、ロボットがいつも写る追従の映像を主にする。
        """
        parts = self.parts
        seconds = (t1 - t0) / SPEED
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
        a_in, a_f = self._audio_args(narration, seconds)
        audio_index = 3 + len(windows)
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
        fg.append(f"[{audio_index}:a]{a_f}[a]")
        mp4 = parts / f"{name}.mp4"
        ffmpeg(*inputs, *a_in, "-filter_complex", ";".join(fg), "-map", f"[{last}]", "-map", "[a]",
               "-t", f"{seconds:.3f}", *VIDEO, *AUDIO, str(mp4))  # fmt: skip
        self.paths.append(mp4)


def photo(run_dir: Path, name: str) -> np.ndarray:
    return np.asarray(Image.open(run_dir / f"{name}_rgb.png").convert("RGB"))


def first_frame(video: Path, at: float, out: Path) -> Image.Image:
    ffmpeg("-ss", f"{at}", "-i", str(video), "-frames:v", "1", str(out))
    return Image.open(out).convert("RGB")


def report_screenshot(report_html: Path, out: Path) -> Image.Image:
    """報告の画面を Edge（headless）で撮る。開いている Edge に渡されないよう、別のプロフィールで起動する。"""
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
    ap.add_argument("--voice", choices=["auto", "voicevox", "sapi", "none"], default="auto")
    ap.add_argument("--speaker", type=int, default=3, help="VOICEVOX の話者の ID（既定 3：ずんだもん ノーマル）")
    ap.add_argument("--bgm", type=Path, default=None, help="BGM の音声ファイル（小さい音量で重ね、繰り返す）")
    ap.add_argument("--bgm-volume", type=float, default=0.12)
    args = ap.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg が見つかりません")
    parts = args.out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    voice = None if args.voice == "none" else Voice(args.voice, args.speaker, parts / "voice")
    print(f"[make_video] voice={None if voice is None else voice.engine}", flush=True)
    clips = Clips(parts, voice)

    log = json.loads((args.run / "log.json").read_text(encoding="utf-8"))
    summary = json.loads(args.eval.read_text(encoding="utf-8"))["summary"]
    facts = report_facts(log, SCENE)
    numbers = {o["name"]: o["no"] for o in facts["objects"]}
    first_moves = log["detection"]["moves"]
    verify_moves = (log.get("verify") or {}).get("detection", {}).get("moves", [])
    verify_t = next((e["t"] for e in log["events"] if e["kind"] == "verify"), None)
    caps = episode_captions(log, SCENE, DISPLAY)

    # 1. タイトル
    bg = first_frame(args.run / "overview.mp4", 1.0, parts / "frame0.png")
    bg = Image.blend(bg, Image.new("RGB", bg.size, (0, 0, 0)), 0.55)
    d = ImageDraw.Draw(bg)
    d.text((W / 2, 270), "写真 1 枚で「いつもの状態」を教えると、", font=font(46, True), fill="white", anchor="mm")
    d.text((W / 2, 340), "ロボットが部屋を元に戻す", font=font(58, True), fill="white", anchor="mm")
    d.text((W / 2, 440), "Physical AI 応用講座 最終課題", font=font(28), fill=(220, 220, 220), anchor="mm")
    sub = "NVIDIA Isaac Sim 6.1 ／ RidgebackFranka ／ Claude Sonnet 5.5"
    d.text((W / 2, 485), sub, font=font(24), fill=(220, 220, 220), anchor="mm")
    clips.still(bg, 5, "01_title", "写真1枚で、いつもの状態を教えると、ロボットが部屋を元に戻します。")

    # 2. 使い方
    img, d = canvas("使い方")
    steps = [
        (
            "1",
            "いつもの状態を教える",
            "片付いた部屋を、天井のカメラで 1 枚撮っておく。|置き場所のルールは書かなくてよい",
        ),
        ("2", "散らかったら頼む", "今の写真をいつもの写真と見比べ、|どの物がどこへ動いたかを VLM（Claude）が答える"),
        (
            "3",
            "ロボットが戻して報告する",
            "移動マニピュレータが物を掴んで戻し、|片付けた後にもう一度確かめて、結果を報告する",
        ),
    ]
    y = 140
    for n, head, body in steps:
        d.rounded_rectangle((70, y, 70 + 56, y + 56), radius=28, fill=ACCENT)
        d.text((98, y + 28), n, font=font(30, True), fill="white", anchor="mm")
        d.text((150, y + 4), head, font=font(34, True), fill=INK)
        text_block(d, (150, y + 58), body, 26, 1040, fill=MUTED)
        y += 160
    note = "物体は、日用品（コップ・リモコンなど）に見立てた単色の円柱と直方体"
    text_block(d, (70, y + 10), note, 22, 1100, fill=MUTED)
    clips.still(
        img,
        9,
        "02_usage",
        "片付いた部屋を、天井のカメラで一枚撮っておきます。散らかったら、今の写真と見比べて、"
        "ブイエルエムのクロードが、どの物がどこへ動いたかを答えます。ロボットが物を戻して、結果を報告します。",
    )

    # 3. 教える・散らかす
    img, d = canvas("いつもの写真と、今の写真", "1")
    for i, (name, label) in enumerate([("reference", "いつもの写真（教える）"), ("current", "今の写真（散らかった）")]):
        x = 56 + i * 600
        paste_fit(img, photo(args.run, name), (x, 120, 568, 426))
        d.text((x + 284, 575), label, font=font(30, True), fill=INK, anchor="mm")
    moved = len(log["meta"]["moved_truth"])
    note = f"天井のカメラ（真下向き、RGB と深度）。|物体 5 個のうち、{moved} 個を別の場所へ動かした"
    text_block(d, (56, 620), note, 24, 1170, fill=MUTED)
    clips.still(
        img, 7, "03_photos", f"左がいつもの写真、右が今の写真です。{moved}つの物が、別の場所へ動かされています。"
    )

    # 4. VLM の答え
    img, d = canvas("VLM（Claude）が見比べた答え", "2")
    marks = photo_marks(log, numbers)["current"]
    paste_fit(img, draw_marks(photo(args.run, "current"), marks), (40, 115, 680, 510))
    y = 125
    # 番号（片付けた順）の順に並べる（VLM が答えた順とは限らない）
    for m in sorted(first_moves, key=lambda m: numbers.get(m["name"], 99)):
        no = numbers.get(m["name"])
        if no is None:
            continue
        r, g, b = mark_color(no)
        d.ellipse((760, y + 2, 760 + 36, y + 38), fill=(r, g, b))
        d.text((778, y + 20), str(no), font=font(22, True), fill="white", anchor="mm")
        d.text((810, y), display_name(m["name"]), font=font(30, True), fill=INK)
        d.text((810, y + 42), f"（答えの名前：{m['name']}）", font=font(18), fill=MUTED)
        y = text_block(d, (810, y + 72), f"今：{m['current_location']}", 22, 430, fill=MUTED, spacing=1.4)
        y = text_block(d, (810, y), f"戻し先：{m['target_location']}", 22, 430, fill=MUTED, spacing=1.4) + 22
    missed = [display_name(o["name"]) for o in facts["objects"] if all(m["name"] != o["name"] for m in first_moves)]
    if missed:
        text_block(d, (760, y + 6), f"※ 少しだけ動いた{'、'.join(missed)}は、|ここでは見逃している", 22, 480, fill=WARN)
    note = "答えは、物体の名前・場所の言葉・画像上の点。|点の深度から、掴む位置と戻し先を 3 次元で求める"
    text_block(d, (40, 640), note, 22, 1200, fill=MUTED)
    clips.still(
        img,
        10,
        "04_vlm",
        "クロードは、動いた物の名前と場所、画像の上の点を答えます。点の深さから、掴む位置と戻し先を求めます。"
        + ("少しだけ動いた青い円柱は、ここでは見逃しています。" if missed else ""),
    )

    # 5. 片付け（1 回目）
    split = verify_t if verify_t is not None else log["sim_time_s"]
    clips.action(
        args.run,
        0.0,
        split,
        caps,
        "05_action1",
        "ロボットは棚と机の間を移動して、物を掴んで運びます。直方体は、向きも、いつもの向きに戻します。",
    )

    # 6. 確認
    if verify_t is not None and verify_moves:
        img, d = canvas("片付けた後に、もう一度見比べる", "3")
        vmarks = photo_marks(log, numbers)["after"]
        paste_fit(img, draw_marks(photo(args.run, "after"), vmarks), (40, 115, 640, 540))
        names = "、".join(display_name(m["name"]) for m in verify_moves)
        y = text_block(d, (720, 140), "片付けた後の写真を、|いつもの写真ともう一度比べる。", 28, 520)
        y = text_block(d, (720, y + 20), f"まだ戻っていない物：{names}", 28, 520, bold=True, fill=WARN)
        text_block(d, (720, y + 20), "最初の比較で見逃していた物に気づき、|やり直す（1 回まで）", 26, 520, fill=MUTED)
        clips.still(img, 6, "06_verify", "片付けた後にもう一度見比べると、見逃していた青い円柱に気づきました。")
        clips.action(args.run, verify_t, log["sim_time_s"], caps, "07_action2", "やり直して、元の場所に戻します。")

    # 7. 前後の比較
    img, d = canvas("いつもの写真と、片付けた後")
    last = "final" if (args.run / "final_rgb.png").exists() else "after"
    for i, (name, label) in enumerate([("reference", "いつもの写真"), (last, "片付けた後")]):
        x = 56 + i * 600
        paste_fit(img, photo(args.run, name), (x, 120, 568, 426))
        d.text((x + 284, 575), label, font=font(30, True), fill=INK, anchor="mm")
    errs = json.loads((args.run / "result.json").read_text(encoding="utf-8")).get("home_err_m", {})
    err_text = "、".join(f"{DISPLAY.get(k, k)} {v * 1000:.1f}mm" for k, v in errs.items())
    text_block(d, (56, 620), f"戻し先からのずれ（シミュレータの真値）：|{err_text}", 24, 1170, fill=MUTED)
    clips.still(img, 6, "08_compare", "いつもの写真と比べると、ずれは、どれも数ミリでした。")

    # 8. 報告
    report_html = args.run / "report.html"
    if report_html.exists():
        img, d = canvas("結果の報告：1 つの HTML ファイル", "4")
        shot = report_screenshot(report_html, parts / "report_shot.png")
        framed = browser_frame(shot.crop((0, 0, shot.width, min(shot.height, 900))), "report.html", 540)
        paste_fit(img, framed, (40, 108, 560, 600))
        text = json.loads((args.run / "report.json").read_text(encoding="utf-8"))["text"]
        d.text((640, 112), "Claude が書いた報告文（report.html の先頭）", font=font(22, True), fill=ACCENT)
        y = text_block(d, (640, 152), text, 25, 590, spacing=1.6)
        notes = (
            "写真・物体ごとの結果・費用を 1 つの HTML ファイルにまとめ、|ブラウザで開いて見る。"
            "|物体の名前は、VLM に渡した一覧の名前"
        )
        text_block(d, (640, y + 14), notes, 20, 590, fill=MUTED)
        clips.still(
            img,
            9,
            "09_report",
            "結果は、写真と結果の表、クロードが書いた報告文をまとめた、一つのエイチティーエムエルファイルで届きます。",
        )

    # 9. 想定する使い方
    img, d = canvas("想定する使い方")
    uses = [
        ("会議室", "会議の後、机の上の備品（リモコン・ペン立てなど）を、|いつもの配置に戻す"),
        ("ホテル・旅館の客室", "清掃の後に、備品が決まった場所にあるかを確かめ、|ずれていれば置き直す"),
        ("家庭", "外出中に片付けておく。|高齢の家族の住まいで物を定位置に戻し、探し物を減らす"),
        ("店舗", "棚の商品の乱れを、陳列の写真どおりに戻す"),
    ]
    y = 132
    for head, body in uses:
        d.rectangle((70, y + 8, 78, y + 76), fill=ACCENT)
        d.text((100, y), head, font=font(30, True), fill=INK)
        text_block(d, (100, y + 44), body, 24, 1100, fill=MUTED)
        y += 112
    text_block(
        d,
        (70, y + 6),
        "部屋ごと・人ごとに違う「いつもの状態」に、|プログラムを書かずに、写真 1 枚で合わせられる",
        26,
        1140,
        bold=True,
        fill=ACCENT,
    )
    clips.still(
        img,
        10,
        "10_usecases",
        "会議室やホテルの客室、家庭、店舗など、部屋ごとに違う、いつもの状態を、プログラムを書かずに、写真一枚で教えられます。",
    )

    # 10. 評価
    s = summary
    img, d = canvas("評価：開発に使っていない 10 エピソード")
    rows = [
        ("エピソードの成功", f"{s['episode_success']} / {s['episodes']}"),
        ("物体の成功（戻し先から 3cm 以内）", f"{s['task_success']} / {s['tasks']}"),
        (
            "戻し先からのずれ（平均・最大）",
            f"{s['err_success_mm']['mean']:.1f} mm ・ {s['err_success_mm']['max']:.1f} mm",
        ),
        ("直方体の向き（最大のずれ）", f"{s['box_yaw']['max_deg']:.1f}°"),
        ("VLM の検出（適合率・再現率）", f"{s['detection']['precision']:.2f} ・ {s['detection']['recall']:.2f}"),
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
    cond = "条件：動かす物体 1〜3 個（計 18 個）、Claude Sonnet 5.5、|確認とやり直しあり"
    text_block(d, (90, y + 10), cond, 24, 1100, fill=MUTED)
    mean_mm = round(s["err_success_mm"]["mean"])
    clips.still(
        img,
        8,
        "11_eval",
        f"開発に使っていない十回の試行で、すべての物を、平均{mean_mm}ミリのずれで戻せました。"
        "クロードの費用は、一回あたり約二セントです。",
    )

    # 11. 工夫した点
    img, d = canvas("工夫した点")
    points = [
        ("真値を使わない", "物体の位置と向きは画像から、台車の位置は天井カメラから、|掴めたかは指の開きから判定する"),
        ("直方体は向きも戻す", "リモコンなどの箱は、いつもの向きに回して置く（円柱は向きを気にしない）"),
        ("確認とやり直し", "片付けた後にもう一度見比べ、|VLM の見逃しや置き損ねを 1 回だけやり直す"),
        ("掴み方", "隣の物を避ける手の向き、運ぶ速さ、|置いた後に少し上げてから離れる"),
        ("報告", "写真と結果の表、VLM が書いた報告文を、1 つの HTML ファイルにまとめる"),
    ]
    y = 132
    for head, body in points:
        d.rectangle((70, y + 8, 78, y + 72), fill=ACCENT)
        d.text((100, y), head, font=font(30, True), fill=INK)
        text_block(d, (100, y + 44), body, 24, 1100, fill=MUTED)
        y += 108
    clips.still(
        img,
        9,
        "12_points",
        "物の位置は画像から、ロボットの位置は天井のカメラから、掴めたかどうかは指の開きから判断し、"
        "シミュレーターの正解の値は使っていません。",
    )

    # 12. 終わり
    img, d = canvas()
    d.text((W / 2, 290), "写真 1 枚で、いつもの状態に戻す", font=font(48, True), fill=INK, anchor="mm")
    lines = [
        f"ロボットが動く場面は {SPEED} 倍速で再生しています",
        "VLM の答えは、評価の実行で Claude が答えたものを記録して再生しています",
    ]
    if voice is not None:
        lines.append(f"音声：{voice.credit or 'Microsoft Haruka（Windows の音声合成）'}")
    if args.bgm is not None:
        lines.append(f"BGM：{args.bgm.stem}")
    for i, line in enumerate(lines):
        d.text((W / 2, 380 + i * 44), line, font=font(24), fill=MUTED, anchor="mm")
    clips.still(img, 5, "13_end")

    # つなぐ
    lst = parts / "list.txt"
    lst.write_text("".join(f"file '{c.resolve().as_posix()}'\n" for c in clips.paths), encoding="utf-8")
    joined = parts / "joined.mp4"
    ffmpeg("-f", "concat", "-safe", "0", "-i", str(lst), *VIDEO, *AUDIO, str(joined))
    out = args.out / "restorebot_demo.mp4"
    total = duration(joined)
    if args.bgm is not None:  # BGM を小さい音量で繰り返し重ね、最初と最後をフェードする
        mix = (
            f"[1:a]volume={args.bgm_volume},afade=t=in:d=2,afade=t=out:st={max(total - 3, 0):.2f}:d=3[b];"
            "[0:a][b]amix=inputs=2:duration=first:normalize=0[a]"
        )
        ffmpeg("-i", str(joined), "-stream_loop", "-1", "-i", str(args.bgm), "-filter_complex", mix,
               "-map", "0:v", "-map", "[a]", "-c:v", "copy", *AUDIO, "-movflags", "+faststart", str(out))  # fmt: skip
    else:
        ffmpeg("-i", str(joined), "-c", "copy", "-movflags", "+faststart", str(out))
    total = duration(out)
    warn = "" if total <= MAX_SECONDS else f"（上限 {MAX_SECONDS} 秒を超えている）"
    print(f"[make_video] {out} {total:.1f}s{warn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
