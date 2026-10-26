"""推定版の世界（EstimatedWorld）でエピソードを通す単体テスト（#23）。

ダミーのカメラは、撮るたびにその時点の物体の配置（真値）から深度を描く。ロボットには推定した物体の状態が渡り、
記録（置いた後の位置、推定の誤差）は真値で取る。
"""

import math

import numpy as np
import pytest

from restorebot.core.episode import Episode, EpisodeConfig
from restorebot.core.perception import EstimatedWorld
from restorebot.core.scene import HOME_V2, Placement
from restorebot.core.types import Observation
from restorebot.core.vlm.client import TruthVlmClient
from tests.core.test_episode import SCATTERED, FakeRobot, FakeSim, FakeWorld
from tests.core.test_tasks import render_depth

CAM = HOME_V2.camera


class RenderCamera:
    def __init__(self, world):
        self.world = world

    def capture(self):
        placements = {}
        for k, pos in self.world.pos.items():
            s = HOME_V2.surface_at(pos[:2])
            if s is not None:
                placements[k] = Placement(s.key, pos[0], pos[1], self.world.yaw[k])
        depth = render_depth(HOME_V2, placements)
        return Observation(np.zeros((CAM.height, CAM.width, 3), np.uint8), depth, 0.0)


class RecordingRobot(FakeRobot):
    def __init__(self, world):
        super().__init__(world)
        self.picked = []

    def pick(self, obj):
        self.picked.append(obj)
        return super().pick(obj)


def _run(truth, *, verify=False, vlm=None):
    est = EstimatedWorld(HOME_V2, truth)
    robot = RecordingRobot(truth)
    ep = Episode(
        HOME_V2,
        est,
        RenderCamera(truth),
        robot,
        vlm or TruthVlmClient(HOME_V2, lambda: dict(truth.pos)),
        FakeSim(),
        EpisodeConfig(verify=verify),
        truth=truth,
    )
    ep.reference = RenderCamera(FakeWorld(dict(HOME_V2.tidy))).capture()
    return ep.run(), est, robot


def test_episode_uses_estimated_states():
    truth = FakeWorld({**HOME_V2.tidy, **SCATTERED})
    start = {k: truth.pos[k] for k in SCATTERED}
    log, est, robot = _run(truth)
    assert log.status == "succeeded", log.reason
    # ロボットに渡った物体の状態は推定（真値に近いが同じではない）
    for obj in robot.picked:
        assert math.dist(obj.position[:2], start[obj.key][:2]) < 0.01
        assert obj.position != start[obj.key]
    # 記録には推定の誤差（真値との差）が入る
    for t in log.tasks:
        assert t.estimate_error_m is not None and t.estimate_error_m < 0.01
    remote = next(t for t in log.tasks if t.key == "remote")
    assert remote.estimate_yaw_error_deg is not None and remote.estimate_yaw_error_deg < 5.0
    # 置いた物体は、置いた位置にあるとみなされる
    for k in SCATTERED:
        assert math.dist(est.object_state(k).position[:2], HOME_V2.object_center(k, HOME_V2.tidy[k])[:2]) < 0.002


def test_final_positions_are_recorded_with_truth():
    truth = FakeWorld({**HOME_V2.tidy, **SCATTERED})
    log, _, _ = _run(truth)
    for t in log.tasks:
        assert t.final_position == pytest.approx(truth.pos[t.key])


def test_task_skipped_when_estimation_fails():
    # VLM が物体のない所を「今の位置」と答えた → 推定に失敗 → そのタスクを飛ばす
    from restorebot.core.vlm.client import DummyVlmClient, truth_response

    truth = FakeWorld({**HOME_V2.tidy, **SCATTERED})
    wrong = {"cup": (0.25, 0.60, 0.745)}  # 机の上だが何もない
    target = {"cup": HOME_V2.object_center("cup", HOME_V2.tidy["cup"])}
    log, _, robot = _run(truth, vlm=DummyVlmClient(truth_response(HOME_V2, wrong, target)))
    (cup,) = log.tasks
    assert cup.status == "skipped" and "位置が分からない" in cup.reason
    assert robot.picked == []
    assert any(e["kind"] == "perception" and "cup" in e.get("failed", {}) for e in log.events)


def test_verify_round_reestimates():
    from tests.core.test_episode_verify import PerturbRobot

    truth = FakeWorld({**HOME_V2.tidy, **SCATTERED})
    est = EstimatedWorld(HOME_V2, truth)
    robot = PerturbRobot(truth)  # コップを 1 回目だけ 12cm ずらして置く
    ep = Episode(
        HOME_V2,
        est,
        RenderCamera(truth),
        robot,
        TruthVlmClient(HOME_V2, lambda: dict(truth.pos)),
        FakeSim(),
        EpisodeConfig(verify=True),
        truth=truth,
    )
    ep.reference = RenderCamera(FakeWorld(dict(HOME_V2.tidy))).capture()
    log = ep.run()
    assert log.status == "succeeded"
    assert [t.key for t in log.tasks if t.round == 2] == ["cup"]
    cup_home = HOME_V2.object_center("cup", HOME_V2.tidy["cup"])
    assert math.dist(truth.pos["cup"][:2], cup_home[:2]) < 0.01
