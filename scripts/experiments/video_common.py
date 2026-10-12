"""Issue #6 動画の記録方法の検証用の計算（Isaac Sim に依存しない）。

- 台車を経由点に沿って動かすための、経由点の切り替え
- 台車を追いかける撮影用カメラの視点・注視点（部屋の壁の内側に収め、なめらかに動かす）
- 物理のステップのうち、画像を保存するステップの間隔
- 連番画像から mp4 を作る ffmpeg のコマンド

単位は SI（m, rad）。台車の姿勢は (x, y, yaw)。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from experiments.ridgeback_common import Pose2D, pose_error

Point3 = tuple[float, float, float]


def next_waypoint(current: Pose2D, waypoints: Sequence[Pose2D], index: int, *, pos_tol: float, yaw_tol: float) -> int:
    """今の経由点に位置と向きの両方が届いたら次の経由点の番号を返す。最後の経由点では進まない。"""
    err = pose_error(current, waypoints[index])
    if err.distance <= pos_tol and abs(err.dyaw) <= yaw_tol and index < len(waypoints) - 1:
        return index + 1
    return index


def follow_camera_view(
    base: Pose2D, *, back: float, side: float, height: float, look_ahead: float, look_height: float
) -> tuple[Point3, Point3]:
    """台車の後ろ上方から台車の少し前を見る、追従カメラの (視点, 注視点)。side は台車の左方向が正。"""
    x, y, yaw = base
    fx, fy = math.cos(yaw), math.sin(yaw)  # 台車の前方
    lx, ly = -fy, fx  # 台車の左方
    eye = (x - back * fx + side * lx, y - back * fy + side * ly, height)
    target = (x + look_ahead * fx, y + look_ahead * fy, look_height)
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


def capture_interval(*, physics_hz: int, fps: int) -> int:
    """何ステップごとに画像を保存するか。物理の周波数が動画のフレームレートで割り切れる必要がある。"""
    if physics_hz % fps:
        raise ValueError(f"物理の周波数 {physics_hz}Hz が {fps}fps で割り切れません")
    return physics_hz // fps


# ------------------------------------------------------------
# ffmpeg
# ------------------------------------------------------------
_QUALITY_FLAG = {"libx264": "-crf", "h264_nvenc": "-cq"}


def encode_command(input_pattern: str, output: str, *, fps: int, codec: str, quality: int) -> list[str]:
    """連番画像（例：frames/%05d.png）から mp4 を作る ffmpeg のコマンド。

    quality は libx264 の CRF、h264_nvenc の CQ（どちらも小さいほど高画質）。
    yuv420p にして、多くの再生ソフト・動画サイトで再生できるようにする。
    """
    if codec not in _QUALITY_FLAG:
        raise ValueError(f"未対応のコーデックです: {codec}")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(fps), "-i", input_pattern]
    cmd += ["-c:v", codec, _QUALITY_FLAG[codec], str(quality)]
    if codec == "libx264":
        cmd += ["-preset", "medium"]
    else:
        cmd += ["-preset", "p5", "-rc", "vbr"]
    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", output]
    return cmd


def hstack_command(inputs: Sequence[str], output: str, *, quality: int) -> list[str]:
    """同じ高さ・同じ長さの動画を横に並べた 1 本の mp4 を作る ffmpeg のコマンド（libx264）。"""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for path in inputs:
        cmd += ["-i", path]
    cmd += ["-filter_complex", f"hstack=inputs={len(inputs)}"]
    cmd += ["-c:v", "libx264", "-crf", str(quality), "-preset", "medium", "-pix_fmt", "yuv420p"]
    cmd += ["-movflags", "+faststart", output]
    return cmd
