"""Robot（core.types の Protocol）を、#13 の掴む・置くの部品で実装したもの。

※ SimulationApp を生成した後で import すること。
- 掴んだ物体と手先の向きを覚えておき、置くときに使う（core の TidyTask は物体の向きだけを持つ）
- 隣の物体は真値（SimWorld）から作る
- 姿勢の変更：待機姿勢（standby）は RMPflow で手先を動かす。撮影姿勢（photo）は関節の目標をなめらかに補間して動かす
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from restorebot.core.grasp import footprints
from restorebot.core.scene import SceneSpec
from restorebot.core.types import ObjectState, Pose2D, Status, StepResult, WorldState
from restorebot.sim import robot as rb
from restorebot.sim.base import NavigateAction
from restorebot.sim.manipulation import ArmContext, pick_action, place_action, standby_action
from restorebot.sim.scene_builder import SimScene, object_path
from restorebot.sim.world import SimWorld

# 撮影姿勢：畳んだ姿勢のまま、第 1 関節を台車の後ろ向きに回す（関節の限界は ±2.8973rad）。
# 天井カメラから見てアームが台車の上に収まり、天板を隠さない（#14。M1 では台車を下げずに写真を撮る）。
# 待機姿勢の第 1 関節は -1.261rad なので、+2.81 より -2.85 の方が回る量が少ない
PHOTO_ARM = (-2.85, *rb.FOLDED_ARM[1:])
POSTURES = {"photo": PHOTO_ARM, "folded": rb.FOLDED_ARM}


class JointMoveAction:
    """アームの関節の目標を、今の角度から target へなめらかに補間して動かす（台車・指は動かさない）。"""

    # tol：撮影姿勢では 0.03rad に届かない関節があった（#14。差 0.037rad）。写真を撮る姿勢には 0.06rad で十分
    def __init__(self, name: str, sim: SimScene, target: Sequence[float], *, speed: float = 1.0, tol: float = 0.06):
        self.name = name
        self.sim = sim
        self.target = np.asarray(target, dtype=float)
        self.speed = speed  # 関節の角速度の目安 [rad/s]（最も大きく回る関節がこの速さになる時間をかける）
        self.tol = tol
        self.idx = sim.dof_indices(rb.ARM_JOINTS)

    def start(self) -> None:
        self.t = 0.0
        self.q0 = self.sim.robot.get_dof_positions().numpy()[0][self.idx].astype(float)
        self.duration = max(float(np.max(np.abs(self.target - self.q0))) / self.speed, 0.5)

    def step(self, dt: float) -> StepResult:
        self.t += dt
        s = min(self.t / self.duration, 1.0)
        s = s * s * (3 - 2 * s)  # なめらかに加速・減速する
        q = self.q0 + (self.target - self.q0) * s
        self.sim.robot.set_dof_position_targets([q.tolist()], dof_indices=self.idx)
        if self.t < self.duration:
            return StepResult(Status.RUNNING)
        err = np.abs(self.sim.robot.get_dof_positions().numpy()[0][self.idx] - self.target)
        if float(np.max(err)) <= self.tol:
            return StepResult(Status.SUCCEEDED)
        if self.t > self.duration + 3.0:
            worst = int(np.argmax(err))
            return StepResult(
                Status.FAILED, f"{self.name}：{rb.ARM_JOINTS[worst]} が目標に届かない（差 {err[worst]:.3f}rad）"
            )
        return StepResult(Status.RUNNING)


class _OnSuccess:
    """Action が成功したら callback を呼ぶ包み。"""

    def __init__(self, action, callback) -> None:
        self.action, self.callback, self.name = action, callback, action.name

    def start(self) -> None:
        self.action.start()

    def step(self, dt: float) -> StepResult:
        r = self.action.step(dt)
        if r.status is Status.SUCCEEDED:
            self.callback()
        return r


class SimRobot:
    """Robot の実装。台車の移動は sim.base.NavigateAction（M2）。"""

    def __init__(
        self, scene: SceneSpec, sim: SimScene, world: SimWorld, ctx: ArmContext, *, objects: WorldState | None = None
    ) -> None:
        """objects は隣の物体の位置を取る世界（推定版の EstimatedWorld など）。省略すると world（真値）。

        world（真値）は、台車の位置（オドメトリに相当）と、掴めたかどうかの判定（物体の高さ）に使う。
        """
        self.objects = objects or world
        self.scene = scene
        self.sim = sim
        self.world = world
        self.ctx = ctx
        self._held: str | None = None
        self._hand_yaw: float | None = None

    def _neighbors(self, key: str):
        positions = {k: self.objects.object_state(k).position for k in self.scene.tidy if k != key}
        return footprints(self.scene, positions, key)

    def navigate(self, path: list[Pose2D]):
        return NavigateAction(self.sim, self.world, path)

    def pick(self, obj: ObjectState):
        base = self.world.robot_pose()
        action = pick_action(
            self.ctx,
            self.scene,
            obj,
            base,
            lambda: float(self.world.object_pose(obj.key)[0][2]),
            object_path(obj.key),
            self._neighbors(obj.key),
        )

        def held():
            self._held, self._hand_yaw = obj.key, action.hand_yaw

        return _OnSuccess(action, held)

    def place(self, position: tuple[float, float, float], yaw: float):
        """yaw（物体の向き）は使わない：物体の向きは変えず、手先の向きは掴んだときの向きから決める（設計書 9.2）。"""
        if self._held is None:
            raise RuntimeError("何も掴んでいません")
        key = self._held
        action = place_action(
            self.ctx,
            self.scene.object(key).shape,
            position,
            self._hand_yaw,
            self.world.robot_pose(),
            object_path(key),
            self._neighbors(key),
        )

        def released():
            self._held = self._hand_yaw = None

        return _OnSuccess(action, released)

    def arm_to(self, posture: str):
        if posture == "standby":
            return standby_action(self.ctx, self.world.robot_pose())
        if posture in POSTURES:
            return JointMoveAction(posture, self.sim, POSTURES[posture])
        raise ValueError(f"未知の姿勢です: {posture}")

    def holding(self) -> str | None:
        return self._held
