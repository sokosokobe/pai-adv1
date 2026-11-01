"""動画の記録の計算（設計書 12 章）：撮影用カメラの視点、画像を保存する間隔、ffmpeg のコマンド、字幕。

#6 の scripts/experiments/video_common.py から移した。headless で連番画像を保存し、ffmpeg（libx264）で mp4 にする。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from restorebot.core.types import Pose2D

Point3 = tuple[float, float, float]


def capture_interval(*, physics_hz: int, fps: int) -> int:
    """何ステップごとに画像を保存するか。物理の周波数が動画のフレームレートで割り切れる必要がある。"""
    if physics_hz % fps:
        raise ValueError(f"物理の周波数 {physics_hz}Hz が {fps}fps で割り切れません")
    return physics_hz // fps


def follow_camera_view(
    base: Pose2D, *, back: float, side: float, height: float, look_ahead: float, look_height: float
) -> tuple[Point3, Point3]:
    """台車の後ろ上方から台車の少し前を見る、追従カメラの (視点, 注視点)。side は台車の左方向が正。"""
    fx, fy = math.cos(base.yaw), math.sin(base.yaw)  # 台車の前方
    lx, ly = -fy, fx  # 台車の左方
    eye = (base.x - back * fx + side * lx, base.y - back * fy + side * ly, height)
    target = (base.x + look_ahead * fx, base.y + look_ahead * fy, look_height)
    return eye, target


def clamp_to_room(point: Point3, *, half: tuple[float, float], margin: float) -> Point3:
    """点を部屋の壁の内側（壁から margin 以上離れた所）に収める。カメラが壁の外に出て壁しか写らないのを防ぐ。"""
    hx, hy = half[0] - margin, half[1] - margin
    return (max(-hx, min(hx, point[0])), max(-hy, min(hy, point[1])), point[2])


def smooth_point(previous: Point3 | None, new: Point3, *, alpha: float) -> Point3:
    """指数移動平均で点をなめらかに動かす（カメラの揺れを抑える）。previous が None なら new をそのまま返す。"""
    if previous is None:
        return new
    return tuple(p + alpha * (n - p) for p, n in zip(previous, new, strict=True))  # type: ignore[return-value]


def encode_command(input_pattern: str, output: str, *, fps: int, quality: int = 18) -> list[str]:
    """連番画像（例：frames/%05d.png）から mp4 を作る ffmpeg のコマンド（libx264、CRF = quality）。

    yuv420p にして、多くの再生ソフト・動画サイトで再生できるようにする。
    """
    return [
        *("ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps), "-i", input_pattern),
        *("-c:v", "libx264", "-crf", str(quality), "-preset", "medium"),
        *("-pix_fmt", "yuv420p", "-movflags", "+faststart", output),
    ]


def hstack_command(inputs: Sequence[str], output: str, *, quality: int = 18) -> list[str]:
    """同じ高さ・同じ長さの動画を横に並べた 1 本の mp4 を作る ffmpeg のコマンド（libx264）。"""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for path in inputs:
        cmd += ["-i", path]
    cmd += ["-filter_complex", f"hstack=inputs={len(inputs)}"]
    cmd += ["-c:v", "libx264", "-crf", str(quality), "-preset", "medium", "-pix_fmt", "yuv420p"]
    cmd += ["-movflags", "+faststart", output]
    return cmd


@dataclass(frozen=True)
class Caption:
    t0: float  # シミュレーションの時刻 [s]（撮影用のカメラの動画の時刻と同じ）
    t1: float
    text: str


def episode_captions(log: dict, scene, names: dict[str, str] | None = None) -> list[Caption]:
    """1 回の実行の記録（log.json）から、動画の字幕（いつ何をしているか）を作る（#20）。字幕は途切れずにつながる。

    names：物体の key → 字幕に書く名前（省略時は物体の表示名）。

    今の写真と見比べる → タスクごと（物体：取る場所 → 置く場所）→ 片付けた後の確認 → やり直し → やり直した後の写真。
    """
    end = log["sim_time_s"]
    verify_t = next((e["t"] for e in log["events"] if e["kind"] == "verify"), None)
    caps: list[Caption] = []
    t = 0.0

    def add(t1: float, text: str) -> None:
        nonlocal t
        if t1 > t:
            caps.append(Caption(t, t1, text))
            t = t1

    def where(p) -> str:
        s = scene.surface_at(p[:2])
        return "?" if s is None else s.label

    tasks = sorted(log["tasks"], key=lambda r: r["started_s"])
    first = tasks[0]["started_s"] if tasks else (verify_t if verify_t is not None else end)
    add(first, "今の写真を撮り、いつもの写真と見比べる")
    n = 0
    retried = False
    for r in tasks:
        name = (names or {}).get(r["key"], scene.object(r["key"]).name)
        if r.get("round", 1) == 2 and not retried:
            add(r["started_s"], "片付けた後の写真で、もう一度見比べる")
            retried = True
        if r.get("round", 1) == 2:
            prefix = "やり直し："
        else:
            n += 1
            prefix = f"{n}. "
        text = f"{prefix}{name}：{where(r['pick_position'])} → {where(r['place_position'])}"
        if scene.object(r["key"]).shape == "cube":
            text += "（向きも戻す）"
        if r["status"] != "succeeded":
            text += "（失敗）"
        if caps and t < r["started_s"]:  # タスクの間のすき間は、前の字幕を伸ばして埋める
            caps[-1] = Caption(caps[-1].t0, r["started_s"], caps[-1].text)
            t = r["started_s"]
        add(r["started_s"] + r["duration_s"], text)
    if retried:
        add(end, "やり直した後の写真を撮る")
    elif verify_t is not None:
        add(end, "片付けた後の写真で、もう一度見比べる（残りなし）")
    else:
        add(end, "片付けた後の写真を撮る")
    return caps
