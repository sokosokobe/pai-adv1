"""scripts/experiments/video_common.py（動画の記録方法の検証用の計算）の単体テスト。Isaac Sim 不要。"""

import math

import pytest

from experiments.video_common import (
    capture_interval,
    clamp_to_room,
    encode_command,
    follow_camera_view,
    hstack_command,
    next_waypoint,
    smooth_point,
)

# ---- 経由点の切り替え ----

WAYPOINTS = [(1.0, 0.0, 0.0), (1.0, 0.0, math.pi / 2), (0.0, 0.0, math.pi / 2)]


def test_next_waypoint_stays_until_reached():
    assert next_waypoint((0.5, 0.0, 0.0), WAYPOINTS, 0, pos_tol=0.02, yaw_tol=0.02) == 0


def test_next_waypoint_advances_when_reached():
    assert next_waypoint((0.995, 0.0, 0.0), WAYPOINTS, 0, pos_tol=0.02, yaw_tol=0.02) == 1
    # 位置が合っていても向きが違えば進まない
    assert next_waypoint((1.0, 0.0, 0.5), WAYPOINTS, 1, pos_tol=0.02, yaw_tol=0.02) == 1


def test_next_waypoint_stays_at_last():
    assert next_waypoint((0.0, 0.0, math.pi / 2), WAYPOINTS, 2, pos_tol=0.02, yaw_tol=0.02) == 2


# ---- 追従カメラ ----


def test_follow_camera_is_behind_and_above_robot():
    eye, target = follow_camera_view((1.0, 2.0, 0.0), back=1.5, side=0.0, height=2.0, look_ahead=0.5, look_height=0.4)
    assert eye == pytest.approx((-0.5, 2.0, 2.0))
    assert target == pytest.approx((1.5, 2.0, 0.4))


def test_follow_camera_rotates_with_robot_yaw():
    eye, target = follow_camera_view(
        (0.0, 0.0, math.pi / 2), back=1.0, side=0.5, height=2.0, look_ahead=0.5, look_height=0.4
    )
    # yaw 90°：ロボットの前は +y、後ろは -y、左（side が正）は -x
    assert eye == pytest.approx((-0.5, -1.0, 2.0))
    assert target == pytest.approx((0.0, 0.5, 0.4))


def test_clamp_to_room_keeps_point_inside_walls():
    assert clamp_to_room((-2.5, 0.3, 2.0), half=(1.6, 1.2), margin=0.1) == pytest.approx((-1.5, 0.3, 2.0))
    assert clamp_to_room((0.2, 3.0, 2.0), half=(1.6, 1.2), margin=0.1) == pytest.approx((0.2, 1.1, 2.0))
    assert clamp_to_room((0.2, 0.3, 2.0), half=(1.6, 1.2), margin=0.1) == pytest.approx((0.2, 0.3, 2.0))


def test_smooth_point_moves_part_way():
    assert smooth_point((0.0, 0.0, 0.0), (1.0, 2.0, 4.0), alpha=0.25) == pytest.approx((0.25, 0.5, 1.0))
    assert smooth_point(None, (1.0, 2.0, 4.0), alpha=0.25) == pytest.approx((1.0, 2.0, 4.0))


# ---- 保存するフレーム ----


def test_capture_interval():
    assert capture_interval(physics_hz=60, fps=30) == 2
    assert capture_interval(physics_hz=60, fps=60) == 1
    with pytest.raises(ValueError):
        capture_interval(physics_hz=60, fps=25)


# ---- ffmpeg のコマンド ----


def test_encode_command_libx264():
    cmd = encode_command("frames/%05d.png", "out.mp4", fps=30, codec="libx264", quality=18)
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-framerate") + 1] == "30"
    assert cmd[cmd.index("-i") + 1] == "frames/%05d.png"
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-crf") + 1] == "18"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"  # 多くの再生ソフトで再生できる形式
    assert cmd[-1] == "out.mp4"


def test_encode_command_nvenc_uses_cq():
    cmd = encode_command("f/%05d.png", "o.mp4", fps=30, codec="h264_nvenc", quality=19)
    assert cmd[cmd.index("-c:v") + 1] == "h264_nvenc"
    assert cmd[cmd.index("-cq") + 1] == "19"
    assert "-crf" not in cmd


def test_encode_command_rejects_unknown_codec():
    with pytest.raises(ValueError):
        encode_command("f/%05d.png", "o.mp4", fps=30, codec="mpeg4", quality=5)


def test_hstack_command_combines_inputs():
    cmd = hstack_command(["a.mp4", "b.mp4"], "ab.mp4", quality=20)
    assert cmd.count("-i") == 2
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "hstack=inputs=2" in fc
    assert cmd[-1] == "ab.mp4"
