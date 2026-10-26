"""restorebot.core.episode の M2（移動を含む）の状態遷移の単体テスト。

ダミーのロボットは navigate で台車の位置を経路の終わりへ動かし、台車から楽に届かない物体を掴む・置こうとすると
失敗する（止め位置へ正しく移動しているかを確かめるため）。
"""

import math

import numpy as np
import pytest

from restorebot.core.config import NAV
from restorebot.core.episode import Episode, EpisodeConfig
from restorebot.core.navigation import in_reach
from restorebot.core.scene import HOME_V2, Placement
from restorebot.core.types import ObjectState, Observation, Pose2D, Status
from restorebot.core.vlm.client import DummyVlmClient, truth_response
from tests.core.test_episode import FakeAction, FakeCamera, FakeSim
from tests.core.test_tasks import render_depth

CAM = HOME_V2.camera
START = Pose2D(0.0, -0.6, math.pi / 2)  # 部屋の南側の空いた所
# 机と低い棚をまたぐ散らかし方：コップを低い棚へ、スプレー缶を机へ
SCATTERED = {
    "cup": Placement("side_table", -1.45, -0.30, 0.0),
    "spray": Placement("desk", 0.95, 0.62, 0.0),
}
REFERENCE = Observation(np.zeros((CAM.height, CAM.width, 3), np.uint8), render_depth(HOME_V2, {}), 0.0)


class World:
    def __init__(self, placements, pose=START):
        self.pos = {k: HOME_V2.object_center(k, p) for k, p in placements.items()}
        self.yaw = {k: p.yaw for k, p in placements.items()}
        self.pose = pose

    def object_state(self, key):
        s = HOME_V2.surface_at(self.pos[key][:2])
        return ObjectState(key, self.pos[key], self.yaw[key], None if s is None else s.key)

    def robot_pose(self):
        return self.pose


class Robot:
    def __init__(self, world, nav_fails=False):
        self.world = world
        self.nav_fails = nav_fails
        self.calls: list[str] = []
        self.routes: list[list[Pose2D]] = []
        self.held = None

    def navigate(self, path):
        self.calls.append("nav")
        self.routes.append(list(path))
        if self.nav_fails:
            return FakeAction("nav", status=Status.FAILED, reason="スタック")

        def arrive():
            self.world.pose = path[-1]

        return FakeAction("nav", on_success=arrive)

    def pick(self, obj):
        self.calls.append(f"pick:{obj.key}")
        if not in_reach(self.world.pose, obj.position[:2]):
            return FakeAction("pick", status=Status.FAILED, reason="届かない所で掴もうとした")

        def grab():
            self.held = obj.key

        return FakeAction("pick", on_success=grab)

    def place(self, position, yaw):
        key = self.held
        self.calls.append(f"place:{key}")
        if not in_reach(self.world.pose, position[:2]):
            return FakeAction("place", status=Status.FAILED, reason="届かない所で置こうとした")

        def put():
            self.world.pos[key] = tuple(position)
            self.held = None

        return FakeAction("place", on_success=put)

    def arm_to(self, posture):
        self.calls.append(f"arm:{posture}")
        return FakeAction(f"arm:{posture}", steps=2)

    def holding(self):
        return self.held


def _episode(*, world=None, robot=None, camera=None, vlm=None, scattered=SCATTERED):
    world = world or World({**HOME_V2.tidy, **scattered})
    robot = robot or Robot(world)
    if vlm is None:
        current = {k: world.pos[k] for k in scattered}
        target = {k: HOME_V2.object_center(k, HOME_V2.tidy[k]) for k in scattered}
        vlm = DummyVlmClient(truth_response(HOME_V2, current, target))
    ep = Episode(HOME_V2, world, camera or FakeCamera(), robot, vlm, FakeSim(), EpisodeConfig(milestone="m2"))
    ep.reference = REFERENCE
    return ep, world, robot


def test_m2_moves_between_desk_and_side_table():
    ep, world, robot = _episode()
    log = ep.run()
    assert log.status == "succeeded", (log.reason, [(t.key, t.status, t.reason) for t in log.tasks])
    for key in SCATTERED:
        assert world.pos[key] == pytest.approx(HOME_V2.object_center(key, HOME_V2.tidy[key]), abs=1e-3)
    # タスクごとに：待機姿勢 → 移動 → 掴む → 移動 → 置く
    seq = [c for c in robot.calls if c.split(":")[0] in ("nav", "pick", "place")]
    for key in log.tasks:
        i = seq.index(f"pick:{key.key}")
        assert seq[i - 1] == "nav" and seq[i + 1] == "nav" and seq[i + 2] == f"place:{key.key}"


def test_m2_moves_with_arm_in_standby():
    ep, world, robot = _episode()
    ep.run()
    for i, c in enumerate(robot.calls):
        if c == "nav":
            # 動く前のアームの姿勢は待機姿勢（掴む前）か、物を持った待機姿勢（掴んだ後）
            prev = [x for x in robot.calls[:i] if x.startswith(("arm:", "pick:"))][-1]
            assert prev in ("arm:standby",) or prev.startswith("pick:")


def test_m2_stays_when_object_already_in_reach():
    # 台車が最初から机の前にいて、机の上だけで片付くなら移動しない
    from restorebot.core.config import M1_BASE_POSE

    scattered = {"cup": Placement("desk", 0.90, 0.62, 0.0)}
    world = World({**HOME_V2.tidy, **scattered}, pose=M1_BASE_POSE)
    ep, _, robot = _episode(world=world, scattered=scattered)
    log = ep.run()
    assert log.status == "succeeded"
    assert "nav" not in robot.calls


def test_m2_skips_unreachable_object():
    scattered = {"cup": Placement("desk", 0.80, 1.05, 0.0)}  # 机の奥：どの辺からも届かない
    ep, _, robot = _episode(scattered=scattered)
    log = ep.run()
    (cup,) = log.tasks
    assert cup.status == "skipped" and "届かない" in cup.reason
    assert not any(c.startswith("pick") for c in robot.calls)


def test_m2_navigation_failure_skips_task():
    world = World({**HOME_V2.tidy, **SCATTERED})
    ep, _, robot = _episode(world=world, robot=Robot(world, nav_fails=True))
    log = ep.run()
    assert all(t.status == "skipped" and "スタック" in t.reason for t in log.tasks)
    assert log.status == "failed"


def test_m2_retake_retreats_straight_back():
    camera = FakeCamera(hidden_shots=1)
    ep, world, robot = _episode(camera=camera)
    log = ep.run()
    retakes = [e for e in log.events if e["kind"] == "retake"]
    assert len(retakes) == 1 and retakes[0]["retreated"] is True
    first_route = robot.routes[0]
    assert len(first_route) == 1
    p = first_route[0]
    assert (p.x, p.y, p.yaw) == pytest.approx((START.x, START.y - NAV.shoot_retreat, START.yaw))
    assert robot.calls.index("nav") < robot.calls.index(next(c for c in robot.calls if c.startswith("pick")))


def test_m2_retake_in_place_when_cannot_retreat():
    near_wall = Pose2D(0.0, -0.95, math.pi / 2)  # 後ろに 20cm 下がると南の壁に近すぎる
    world = World({**HOME_V2.tidy, **SCATTERED}, pose=near_wall)
    camera = FakeCamera(hidden_shots=1)
    ep, _, robot = _episode(world=world, camera=camera)
    log = ep.run()
    retakes = [e for e in log.events if e["kind"] == "retake"]
    assert len(retakes) == 1 and retakes[0]["retreated"] is False
    assert camera.shots == 3
