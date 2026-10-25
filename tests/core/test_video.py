"""restorebot.core.video（撮影用カメラの視点、画像を保存する間隔、ffmpeg のコマンド）の単体テスト。"""

import math

import pytest

from restorebot.core.types import Pose2D
from restorebot.core.video import (
    capture_interval,
    clamp_to_room,
    encode_command,
    follow_camera_view,
    hstack_command,
    smooth_point,
)


def test_capture_interval():
    assert capture_interval(physics_hz=60, fps=30) == 2
    with pytest.raises(ValueError):
        capture_interval(physics_hz=60, fps=25)


def test_follow_camera_is_behind_and_above_robot():
    eye, target = follow_camera_view(
        Pose2D(1.0, 0.0, math.pi / 2), back=1.5, side=0.5, height=1.8, look_ahead=0.8, look_height=0.7
    )
    assert eye == pytest.approx((1.0 - 0.5, 0.0 - 1.5, 1.8))  # +y を向いているので、後ろは -y、左は -x
    assert target == pytest.approx((1.0, 0.8, 0.7))


def test_clamp_to_room_and_smoothing():
    assert clamp_to_room((3.0, -2.0, 1.0), half=(2.0, 1.5), margin=0.1) == pytest.approx((1.9, -1.4, 1.0))
    assert smooth_point(None, (1.0, 2.0, 3.0), alpha=0.5) == (1.0, 2.0, 3.0)
    assert smooth_point((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), alpha=0.5) == pytest.approx((0.5, 1.0, 1.5))


def test_encode_command():
    cmd = encode_command("frames/%05d.png", "out.mp4", fps=30, quality=18)
    assert cmd[0] == "ffmpeg" and cmd[-1] == "out.mp4"
    assert cmd[cmd.index("-framerate") + 1] == "30"
    assert cmd[cmd.index("-c:v") + 1] == "libx264" and cmd[cmd.index("-crf") + 1] == "18"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"


def test_hstack_command():
    cmd = hstack_command(["a.mp4", "b.mp4"], "ab.mp4", quality=20)
    assert cmd.count("-i") == 2 and "hstack=inputs=2" in cmd
