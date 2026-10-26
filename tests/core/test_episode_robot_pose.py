"""全体の制御で、写真から台車の位置を推定してオドメトリと比べる（#26）の単体テスト。"""

import math

import numpy as np

from restorebot.core.config import M1_BASE_POSE
from restorebot.core.episode import EpisodeConfig
from restorebot.core.types import Observation, Pose2D
from tests.core.test_episode import CAM, FakeCamera, _episode, _reference_with_depth
from tests.core.test_robot_pose import render_robot


class RobotCamera:
    """台車を pose（本当の位置）に描いた深度を返す（オドメトリはいつも M1_BASE_POSE）。"""

    def __init__(self, pose: Pose2D):
        self.depth = render_robot(pose, base=_reference_with_depth().depth)

    def capture(self):
        return Observation(np.zeros((CAM.height, CAM.width, 3), np.uint8), self.depth.copy(), 0.0)


def _events(log, kind):
    return [e for e in log.events if e["kind"] == kind]


def test_records_estimate_for_each_photo():
    ep, _, _ = _episode(camera=RobotCamera(M1_BASE_POSE), config=EpisodeConfig(verify=True))
    log = ep.run()
    assert log.status == "succeeded", log.reason
    checks = _events(log, "robot_pose")
    # ダミーの VLM は確認でも同じ答えを返すので、やり直しまで進み、やり直した後の写真も確かめる
    assert [e["photo"] for e in checks] == ["current", "after", "final"]
    for e in checks:
        assert e["diff_m"] < 0.02 and abs(e["diff_deg"]) < 3.0
        assert e["odometry"] == [M1_BASE_POSE.x, M1_BASE_POSE.y, M1_BASE_POSE.yaw]
    assert not _events(log, "robot_pose_warning")


def _drifted():
    # 本当の台車は、オドメトリ（M1_BASE_POSE）より 8cm 右・4° 回った所にいる
    p = M1_BASE_POSE
    return Pose2D(p.x + 0.08, p.y, p.yaw + math.radians(4))


def test_warns_when_odometry_differs():
    ep, _, _ = _episode(camera=RobotCamera(_drifted()), config=EpisodeConfig(robot_pose_check="warn"))
    log = ep.run()
    (check, *_) = _events(log, "robot_pose")
    assert abs(check["diff_m"] - 0.08) < 0.02 and abs(check["diff_deg"] - 4.0) < 1.0
    assert [w["photo"] for w in _events(log, "robot_pose_warning")] == ["current", "after"]


def test_record_mode_does_not_warn_and_off_skips():
    ep, _, _ = _episode(camera=RobotCamera(_drifted()), config=EpisodeConfig(robot_pose_check="record"))
    log = ep.run()
    assert _events(log, "robot_pose") and not _events(log, "robot_pose_warning")
    ep, _, _ = _episode(camera=RobotCamera(_drifted()), config=EpisodeConfig(robot_pose_check="off"))
    assert not _events(ep.run(), "robot_pose")


def test_failed_estimate_is_recorded_and_episode_continues():
    ep, _, _ = _episode(camera=FakeCamera())  # 台車が写っていない
    log = ep.run()
    assert log.status == "succeeded"
    check = _events(log, "robot_pose")[0]
    assert check["estimate"] is None and check["message"]
