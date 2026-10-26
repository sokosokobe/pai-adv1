"""restorebot.core.episode の確認とやり直し（設計書 10 章、#19）と TruthVlmClient の単体テスト。"""

import pytest

from restorebot.core.episode import Episode, EpisodeConfig
from restorebot.core.scene import HOME_V2
from restorebot.core.vlm.client import DummyVlmClient, TruthVlmClient, VlmApiError, truth_response
from tests.core.test_episode import SCATTERED, FakeCamera, FakeRobot, FakeSim, FakeWorld, _reference_with_depth

OFFSET = (0.12, 0.0, 0.0)  # わざと戻し損ねる量（コップを 12cm 横に置く）


class PerturbRobot(FakeRobot):
    """key の物体の置く位置を、最初の times 回だけ OFFSET ずらす。"""

    def __init__(self, world, key="cup", times=1):
        super().__init__(world)
        self.key, self.times = key, times

    def place(self, position, yaw):
        if self.held == self.key and self.times > 0:
            self.times -= 1
            position = tuple(p + d for p, d in zip(position, OFFSET, strict=True))
        return super().place(position, yaw)


def _home(key):
    return HOME_V2.object_center(key, HOME_V2.tidy[key])


def _truth_vlm(world):
    return TruthVlmClient(HOME_V2, lambda: dict(world.pos))


def _episode(world, robot, vlm, *, verify=True):
    ep = Episode(HOME_V2, world, FakeCamera(), robot, vlm, FakeSim(), EpisodeConfig(verify=verify))
    ep.reference = _reference_with_depth()
    return ep


def _world():
    return FakeWorld({**HOME_V2.tidy, **SCATTERED})


# ---- TruthVlmClient ----


def test_truth_vlm_reports_objects_away_from_home_at_call_time():
    world = _world()
    vlm = _truth_vlm(world)
    first = vlm.detect(b"", b"", "")
    assert first.text == truth_response(HOME_V2, {k: world.pos[k] for k in SCATTERED}, {k: _home(k) for k in SCATTERED})
    world.pos["cup"] = _home("cup")
    world.pos["remote"] = tuple(c + 0.005 for c in _home("remote"))  # 2cm 未満のずれは戻ったとみなす
    assert vlm.detect(b"", b"", "").text == '{"moved_objects": []}'


# ---- 確認とやり直し ----


def test_no_retry_when_everything_is_back():
    world = _world()
    vlm = _truth_vlm(world)
    log = _episode(world, FakeRobot(world), vlm).run()
    assert log.status == "succeeded"
    assert all(t.round == 1 for t in log.tasks)
    assert log.verify["detection"]["ok"] and log.verify["instructions"] == []
    assert any(e["kind"] == "verify" for e in log.events)
    assert "final" not in log.photos


def test_retry_returns_the_object_left_behind():
    world = _world()
    robot = PerturbRobot(world)
    log = _episode(world, robot, _truth_vlm(world)).run()
    assert log.status == "succeeded", log.reason
    retried = [t for t in log.tasks if t.round == 2]
    assert [t.key for t in retried] == ["cup"] and retried[0].status == "succeeded"
    first_cup = next(t for t in log.tasks if t.key == "cup" and t.round == 1)
    assert first_cup.error_m == pytest.approx(0.12, abs=1e-6)  # 1 回目はずれた
    # 戻し先は VLM の応答（0.1px 単位）を逆投影して求めるので、1mm 未満の差は残る
    assert world.pos["cup"] == pytest.approx(_home("cup"), abs=1e-3)
    assert {"after", "final"} <= set(log.photos)
    assert [i["key"] for i in log.verify["instructions"]] == ["cup"]


def test_retry_only_once():
    world = _world()
    robot = PerturbRobot(world, times=2)  # やり直しでもずれる
    remaining = truth_response(HOME_V2, {"cup": (0.57, 0.72, 0.745)}, {"cup": _home("cup")})
    vlm = DummyVlmClient(
        [truth_response(HOME_V2, {k: world.pos[k] for k in SCATTERED}, {k: _home(k) for k in SCATTERED}), remaining]
    )
    log = _episode(world, robot, vlm).run()
    assert len(vlm.prompts) == 2  # 最初の比較と、確認の 1 回だけ
    assert sum(t.round == 2 for t in log.tasks) == 1
    assert world.pos["cup"] != pytest.approx(_home("cup"), abs=0.01)


def test_verify_vlm_failure_keeps_first_round_result():
    world = _world()
    first = truth_response(HOME_V2, {k: world.pos[k] for k in SCATTERED}, {k: _home(k) for k in SCATTERED})
    vlm = DummyVlmClient([first, VlmApiError("timeout")])
    log = _episode(world, FakeRobot(world), vlm).run()
    assert log.status == "succeeded"
    assert log.verify["detection"]["ok"] is False
    assert any(e["kind"] == "verify_failed" for e in log.events)


def test_verify_disabled():
    world = _world()
    first = truth_response(HOME_V2, {k: world.pos[k] for k in SCATTERED}, {k: _home(k) for k in SCATTERED})
    vlm = DummyVlmClient([first])
    log = _episode(world, PerturbRobot(world), vlm, verify=False).run()
    assert len(vlm.prompts) == 1 and log.verify is None
    assert all(t.round == 1 for t in log.tasks)


def test_status_uses_last_record_per_object():
    # 1 回目に掴めずに飛ばしても、やり直しで戻せれば成功
    world = _world()
    robot = FakeRobot(world, pick_failures={"cup": 2})  # 1 回目の 2 回の試行は失敗、やり直しで成功
    log = _episode(world, robot, _truth_vlm(world)).run()
    assert [(t.key, t.round, t.status) for t in log.tasks if t.key == "cup"] == [
        ("cup", 1, "skipped"),
        ("cup", 2, "succeeded"),
    ]
    assert log.status == "succeeded"
