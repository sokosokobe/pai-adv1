"""restorebot.core.episode（M1 の状態遷移と失敗の扱い）の単体テスト。

ダミーの Robot・カメラ・世界・シミュレーション・VLM を使う（Isaac Sim・API 不要）。
"""

import json
import math

import numpy as np
import pytest

from restorebot.core.config import M1_BASE_POSE
from restorebot.core.episode import Episode, EpisodeConfig
from restorebot.core.scene import HOME_V2, Placement
from restorebot.core.types import ObjectState, Observation, Pose2D, Status, StepResult
from restorebot.core.vlm.client import DummyVlmClient, VlmApiError, truth_response

CAM = HOME_V2.camera
SCATTERED = {
    "cup": Placement("desk", 0.90, 0.62, 0.0),
    "remote": Placement("desk", 0.55, 0.62, 0.3),
}


# ---- ダミー ----
class FakeWorld:
    def __init__(self, placements):
        self.pos = {k: HOME_V2.object_center(k, p) for k, p in placements.items()}
        self.yaw = {k: p.yaw for k, p in placements.items()}

    def object_state(self, key):
        return ObjectState(key, self.pos[key], self.yaw[key], "desk")

    def robot_pose(self):
        return M1_BASE_POSE


class FakeAction:
    def __init__(self, name, steps=3, status=Status.SUCCEEDED, reason=None, on_success=None):
        self.name, self.steps, self.status, self.reason, self.on_success = name, steps, status, reason, on_success
        self.started = False

    def start(self):
        self.started = True
        self.n = 0

    def step(self, dt):
        self.n += 1
        if self.n < self.steps:
            return StepResult(Status.RUNNING)
        if self.status is Status.SUCCEEDED and self.on_success:
            self.on_success()
        return StepResult(self.status, self.reason)


class FakeRobot:
    """pick_failures[key] 回だけ掴み損ねる。place は world の物体を置く位置へ動かす。"""

    def __init__(self, world, pick_failures=None, place_fails=()):
        self.world = world
        self.pick_failures = dict(pick_failures or {})
        self.place_fails = set(place_fails)
        self.calls: list[str] = []
        self.held = None

    def navigate(self, path):
        raise AssertionError("M1 では台車を動かさない")

    def pick(self, obj):
        self.calls.append(f"pick:{obj.key}")
        if self.pick_failures.get(obj.key, 0) > 0:
            self.pick_failures[obj.key] -= 1
            return FakeAction("pick", status=Status.FAILED, reason="掴み損ね")

        def grab():
            self.held = obj.key

        return FakeAction("pick", on_success=grab)

    def place(self, position, yaw):
        key = self.held
        self.calls.append(f"place:{key}")
        if key in self.place_fails:
            return FakeAction("place", status=Status.FAILED, reason="時間切れ")

        def put():
            self.world.pos[key] = tuple(position)
            self.held = None

        return FakeAction("place", on_success=put)

    def arm_to(self, posture):
        self.calls.append(f"arm:{posture}")
        return FakeAction(f"arm:{posture}", steps=2)

    def holding(self):
        return self.held


class FakeCamera:
    """撮るたびに、ロボットの写り込み（深度の手前の領域）を hidden_shots 回だけ天板の上に入れる。"""

    def __init__(self, hidden_shots=0):
        self.hidden_shots = hidden_shots
        self.shots = 0

    def capture(self):
        self.shots += 1
        depth = np.full((CAM.height, CAM.width), 2.0, dtype=np.float32)
        if self.shots <= self.hidden_shots:
            u, v = CAM.project((0.8, 0.75, 0.7))
            depth[int(v) - 5 : int(v) + 5, int(u) - 5 : int(u) + 5] = 1.0  # 机の上にロボット
        return Observation(np.zeros((CAM.height, CAM.width, 3), np.uint8), depth, 0.0)


class FakeSim:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1
        return 1.0 / 60.0


REFERENCE = Observation(
    np.zeros((CAM.height, CAM.width, 3), np.uint8), np.full((CAM.height, CAM.width), 2.0, np.float32), 0.0
)


def _truth_vlm(keys=("cup", "remote")):
    current = {k: HOME_V2.object_center(k, SCATTERED[k]) for k in keys}
    target = {k: HOME_V2.object_center(k, HOME_V2.tidy[k]) for k in keys}
    return DummyVlmClient(truth_response(HOME_V2, current, target))


def _reference_with_depth():
    """戻し先の逆投影ができるよう、天板の深度を入れたいつもの写真（物体は写さない）。"""
    from tests.core.test_tasks import render_depth

    return Observation(REFERENCE.rgb, render_depth(HOME_V2, {}), 0.0)


def _episode(*, robot=None, camera=None, vlm=None, world=None, config=None):
    world = world or FakeWorld({**HOME_V2.tidy, **SCATTERED})
    robot = robot or FakeRobot(world)
    ep = Episode(
        HOME_V2,
        world,
        camera or FakeCamera(),
        robot,
        vlm or _truth_vlm(),
        FakeSim(),
        config or EpisodeConfig(),
    )
    ep.reference = _reference_with_depth()
    return ep, world, robot


# ---- 成功の流れ ----


def test_success_returns_all_objects():
    ep, world, robot = _episode()
    log = ep.run()
    assert log.status == "succeeded", log.reason
    assert [t.status for t in log.tasks] == ["succeeded", "succeeded"]
    for key in SCATTERED:
        assert world.pos[key] == pytest.approx(HOME_V2.object_center(key, HOME_V2.tidy[key]), abs=1e-3)
    # 撮影姿勢で撮る → 待機姿勢 → 近い順（台車 (0.75, -0.02) から：リモコン (0.55, 0.62) とコップ (0.90, 0.62)）
    assert robot.calls[0] == "arm:photo"
    assert [c for c in robot.calls if c.startswith("pick")] == ["pick:cup", "pick:remote"]
    assert robot.calls[-1] == "arm:standby"
    assert set(log.photos) == {"reference", "current", "after"}
    assert log.sim_time_s > 0


def test_no_tasks_when_vlm_finds_nothing():
    ep, _, robot = _episode(vlm=DummyVlmClient('{"moved_objects": []}'))
    log = ep.run()
    assert log.status == "succeeded" and log.tasks == []
    assert not any(c.startswith("pick") for c in robot.calls)


def test_vlm_failure_ends_episode():
    ep, _, robot = _episode(vlm=DummyVlmClient([VlmApiError("timeout")]))
    log = ep.run()
    assert log.status == "failed" and "VLM" in log.reason
    assert not any(c.startswith("pick") for c in robot.calls)
    assert log.detection["ok"] is False


# ---- 撮り直し ----


def test_retake_when_robot_hides_surface():
    camera = FakeCamera(hidden_shots=1)
    ep, _, robot = _episode(camera=camera)
    log = ep.run()
    assert camera.shots == 1 + 1 + 1  # 今の写真（隠れ）→ 撮り直し → 片付けた後
    assert sum(e["kind"] == "retake" for e in log.events) == 1
    assert log.status == "succeeded"


def test_proceed_after_max_retakes():
    camera = FakeCamera(hidden_shots=10)
    ep, _, _ = _episode(camera=camera, config=EpisodeConfig(max_retakes=2))
    log = ep.run()
    assert sum(e["kind"] == "retake" for e in log.events) == 2
    assert any(e["kind"] == "still_hidden" for e in log.events)
    assert log.status == "succeeded"  # M1 ではその写真で進める（設計書 2.3）


# ---- 失敗の扱い ----


def test_grasp_retry_then_success():
    world = FakeWorld({**HOME_V2.tidy, **SCATTERED})
    robot = FakeRobot(world, pick_failures={"cup": 1})
    ep, _, _ = _episode(robot=robot, world=world)
    log = ep.run()
    cup = next(t for t in log.tasks if t.key == "cup")
    assert cup.status == "succeeded" and cup.pick_attempts == 2
    assert log.status == "succeeded"


def test_grasp_fails_twice_then_skip():
    world = FakeWorld({**HOME_V2.tidy, **SCATTERED})
    robot = FakeRobot(world, pick_failures={"cup": 5})
    ep, _, _ = _episode(robot=robot, world=world)
    log = ep.run()
    cup = next(t for t in log.tasks if t.key == "cup")
    assert cup.status == "skipped" and cup.pick_attempts == 2 and "掴み損ね" in cup.reason
    remote = next(t for t in log.tasks if t.key == "remote")
    assert remote.status == "succeeded"  # 1 個の失敗で全体を止めない
    assert log.status == "partial"
    # 失敗したら待機姿勢に戻してから次へ
    i = max(i for i, c in enumerate(robot.calls) if c == "pick:cup")
    assert robot.calls[i + 1] == "arm:standby"


def test_place_failure_is_recorded():
    world = FakeWorld({**HOME_V2.tidy, **SCATTERED})
    robot = FakeRobot(world, place_fails={"remote"})
    ep, _, _ = _episode(robot=robot, world=world)
    log = ep.run()
    remote = next(t for t in log.tasks if t.key == "remote")
    assert remote.status == "failed" and "時間切れ" in remote.reason
    assert log.status == "partial"


def test_unreachable_task_is_skipped_in_m1():
    # 机の上でも、M1 の止め位置から届かない位置にある物体は飛ばす
    far = {"cup": Placement("desk", 0.30, 0.62, 0.0)}
    world = FakeWorld({**HOME_V2.tidy, **far})
    current = {"cup": HOME_V2.object_center("cup", far["cup"])}
    target = {"cup": HOME_V2.object_center("cup", HOME_V2.tidy["cup"])}
    ep, _, robot = _episode(world=world, vlm=DummyVlmClient(truth_response(HOME_V2, current, target)))
    log = ep.run()
    (cup,) = log.tasks
    assert cup.status == "skipped" and "届かない" in cup.reason
    assert not any(c.startswith("pick") for c in robot.calls)


def test_action_timeout_fails_task():
    class StuckRobot(FakeRobot):
        def pick(self, obj):
            self.calls.append(f"pick:{obj.key}")
            return FakeAction("pick", steps=10_000)

    world = FakeWorld({**HOME_V2.tidy, **SCATTERED})
    ep, _, _ = _episode(robot=StuckRobot(world), world=world, config=EpisodeConfig(action_timeout_s=0.5))
    log = ep.run()
    assert all(t.status == "skipped" for t in log.tasks)
    assert all("終わらなかった" in t.reason for t in log.tasks)


def test_dropped_predictions_are_logged():
    text = json.dumps(
        {
            "moved_objects": [
                {
                    "name": "緑のボール",
                    "current_location": "",
                    "current_point": [100, 100],
                    "target_location": "",
                    "target_point": [600, 600],
                }
            ]
        },
        ensure_ascii=False,
    )
    ep, _, _ = _episode(vlm=DummyVlmClient(text))
    log = ep.run()
    assert log.tasks == [] and any("緑のボール" in d for d in log.dropped)


# ---- 記録 ----


def test_log_is_saved(tmp_path):
    ep, _, _ = _episode()
    log = ep.run()
    log.save(tmp_path)
    data = json.loads((tmp_path / "log.json").read_text(encoding="utf-8"))
    assert data["status"] == "succeeded"
    assert [t["key"] for t in data["tasks"]] == ["cup", "remote"]
    assert data["detection"]["attempts"][0]["error"] is None
    for name in ("reference", "current", "after"):
        assert (tmp_path / f"{name}_rgb.png").is_file()
        assert (tmp_path / f"{name}_depth.npy").is_file()


def test_task_error_is_measured_against_place_position():
    ep, world, _ = _episode()
    log = ep.run()
    for t in log.tasks:
        assert t.error_m == pytest.approx(0.0, abs=1e-6)
        assert t.final_position == pytest.approx(world.pos[t.key])
    assert math.isfinite(log.to_dict()["sim_time_s"])


def test_initial_robot_pose_is_recorded():
    ep, _, _ = _episode()
    log = ep.run()
    assert log.meta["base_pose"] == pytest.approx([M1_BASE_POSE.x, M1_BASE_POSE.y, M1_BASE_POSE.yaw])
    assert isinstance(Pose2D(*log.meta["base_pose"]), Pose2D)
