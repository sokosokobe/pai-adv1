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


# ---- 動画の字幕（#20）----


def _log_with_retry():
    from restorebot.core.scene import HOME_V2

    desk = HOME_V2.object_center("cup", HOME_V2.tidy["cup"])  # 机の上の点
    shelf = HOME_V2.object_center("spray", HOME_V2.tidy["spray"])  # 低い棚の上の点
    return {
        "events": [
            {"t": 0.0, "kind": "start"},
            {"t": 2.1, "kind": "plan"},
            {"t": 43.0, "kind": "verify"},
            {"t": 65.0, "kind": "end"},
        ],
        "tasks": [
            {"key": "remote", "round": 1, "status": "succeeded", "started_s": 2.1, "duration_s": 21.2,
             "pick_position": list(shelf), "place_position": list(desk)},
            {"key": "spray", "round": 1, "status": "succeeded", "started_s": 23.4, "duration_s": 15.9,
             "pick_position": list(desk), "place_position": list(shelf)},
            {"key": "pen_stand", "round": 2, "status": "succeeded", "started_s": 43.1, "duration_s": 17.0,
             "pick_position": list(desk), "place_position": list(desk)},
        ],
        "sim_time_s": 65.0,
    }  # fmt: skip


def test_episode_captions_follow_the_log():
    from restorebot.core.scene import HOME_V2
    from restorebot.core.video import episode_captions

    caps = episode_captions(_log_with_retry(), HOME_V2)
    texts = [c.text for c in caps]
    assert texts[0].startswith("今の写真")
    assert texts[1] == "1. 黒いリモコン：低い棚の上 → 机の上（向きも戻す）"
    assert texts[2] == "2. 緑のスプレー缶：机の上 → 低い棚の上"
    assert "もう一度見比べる" in texts[3]
    assert texts[4] == "やり直し：青いペン立て：机の上 → 机の上"
    assert "やり直した後" in texts[5]
    # 時間は途切れずにつながる（字幕の出ない時間がない）
    assert caps[0].t0 == 0.0 and caps[-1].t1 == 65.0
    assert all(a.t1 == pytest.approx(b.t0) for a, b in zip(caps, caps[1:], strict=False))
    assert caps[3].t0 == pytest.approx(39.3) and caps[3].t1 == pytest.approx(43.1)


def test_episode_captions_without_retry():
    from restorebot.core.scene import HOME_V2
    from restorebot.core.video import episode_captions

    log = _log_with_retry()
    log["tasks"] = log["tasks"][:1]
    log["events"] = [e for e in log["events"] if e["kind"] != "verify"]
    log["sim_time_s"] = 30.0
    caps = episode_captions(log, HOME_V2)
    assert [c.text for c in caps][-1] == "片付けた後の写真を撮る"
    assert caps[-1].t1 == 30.0


def test_episode_captions_use_given_names():
    from restorebot.core.scene import HOME_V2
    from restorebot.core.video import episode_captions

    caps = episode_captions(_log_with_retry(), HOME_V2, {"remote": "黒い直方体"})
    assert caps[1].text == "1. 黒い直方体：低い棚の上 → 机の上（向きも戻す）"
    assert caps[2].text.startswith("2. 緑のスプレー缶")  # 与えなかった物体は表示名のまま
