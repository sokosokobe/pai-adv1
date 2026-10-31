"""Robot（core.types の Protocol）を、#13 の掴む・置くの部品で実装したもの。

※ SimulationApp を生成した後で import すること。
- 掴んだ物体と手先の向きを覚えておき、置くときに使う（core の TidyTask は物体の向きだけを持つ）
- 隣の物体は真値（SimWorld）から作る
- 掴めたか・運ぶ途中で落としていないかは、指の開きで判定する（#27）。真値による判定は比較のために grip_checks に記録する
- 姿勢の変更：待機姿勢（standby）は RMPflow で手先を動かす。撮影姿勢（photo）は関節の目標をなめらかに補間して動かす
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from restorebot.core.grasp import GRASP_OFFSET_Z, bearing_yaw, box_place_hand_yaw, footprints, judge_grip
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

        world（真値）は、台車の位置（オドメトリに相当）に使う。掴めたかどうかは指の開きで判定し、真値の判定は
        比較のために grip_checks（掴むたび・置く前ごとの記録）に残すだけ（#27）。
        """
        self.objects = objects or world
        self.scene = scene
        self.sim = sim
        self.world = world
        self.ctx = ctx
        self._held: str | None = None
        self._hand_yaw: float | None = None
        self._object_yaw: float | None = None  # 掴んだときの物体の向き（箱の向きを戻すのに使う。#25）
        self.grip_checks: list[dict] = []  # 指の開きによる判定と、真値による判定の記録（#27）

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
            record=self.grip_checks.append,
        )

        def held():
            self._held, self._hand_yaw, self._object_yaw = obj.key, action.hand_yaw, obj.yaw

        return _OnSuccess(action, held)

    def place(self, position: tuple[float, float, float], yaw: float):
        """物体を position に、向き yaw（置いたときの物体の向き）で置く。

        箱：手先を「掴んだときの手先の向き ＋（yaw − 掴んだときの物体の向き）」に回して置く（#25。
        180° 違う 2 通りのうち、台車から置く位置を見た方向に近い方）。
        円柱：yaw は使わず、置く位置の隣の物体を避ける向きを選ぶ（設計書 9.2）。
        """
        if self._held is None:
            raise RuntimeError("何も掴んでいません")
        key = self._held
        shape = self.scene.object(key).shape
        base = self.world.robot_pose()
        hand_yaw = self._hand_yaw
        if shape == "cube":
            hand_yaw = box_place_hand_yaw(self._hand_yaw, self._object_yaw, yaw, bearing_yaw(base, position[:2]))
        width = self.scene.object(key).grasp_width
        self._check_before_place(key, width)
        action = place_action(
            self.ctx, shape, position, hand_yaw, base, object_path(key), self._neighbors(key), grip_width=width
        )

        def released():
            self._held = self._hand_yaw = self._object_yaw = None

        return _OnSuccess(action, released)

    def arm_to(self, posture: str):
        if posture == "standby":
            return standby_action(self.ctx, self.world.robot_pose())
        if posture in POSTURES:
            return JointMoveAction(posture, self.sim, POSTURES[posture])
        raise ValueError(f"未知の姿勢です: {posture}")

    def holding(self) -> str | None:
        return self._held

    def _check_before_place(self, key: str, width: float) -> None:
        """置く前（運んだ後）に、まだ持っているかを指の開きと真値の両方で判定して記録する（比較のため）。

        真値：物体の中心が指先の間（panda_hand から GRASP_OFFSET_Z 下）から 5cm 以内なら持っている。
        """
        art = self.sim.robot
        q = art.get_dof_positions().numpy()[0]
        opening = float(sum(q[i] for i in self.sim.dof_indices(rb.FINGER_JOINTS)))
        hand = self.ctx.planning.tool_pose()[0]  # panda_hand（手先は真下向き）
        grip_point = np.array([hand[0], hand[1], hand[2] - GRASP_OFFSET_Z])
        dist = float(np.linalg.norm(np.asarray(self.world.object_pose(key)[0]) - grip_point))
        self.grip_checks.append(
            {
                "phase": "before_place",
                "key": key,
                "opening_m": opening,
                "grip_ok": judge_grip(opening, width, self.ctx.params) is None,
                "truth_dist_m": dist,
                "truth_ok": bool(dist < 0.05),
            }
        )
