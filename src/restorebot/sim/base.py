"""台車の経路追従（設計書 8.3）。

※ SimulationApp を生成した後で import すること。
台車の x・y・yaw の仮想関節（ワールド基準）に速度の目標を与える（#2・#15。関節の摩擦は scene_builder で 0）。
速度指令は core.navigation.PathFollower が作る。到着したら関節を位置制御に戻して、その場で止める。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from restorebot.core.navigation import PathFollower, wrap_angle
from restorebot.core.types import Pose2D, Status, StepResult
from restorebot.sim import robot as rb
from restorebot.sim.scene_builder import SimScene
from restorebot.sim.world import SimWorld


class NavigateAction:
    """経路 path に沿って台車を動かす。stall_time 秒のあいだ進まなければ失敗（簡単なスタックの検知。設計書 8.5）。"""

    def __init__(
        self,
        sim: SimScene,
        world: SimWorld,
        path: Sequence[Pose2D],
        *,
        stall_time: float = 3.0,
        stall_dist: float = 0.01,
        stall_yaw: float = 0.02,
    ) -> None:
        self.name = "navigate"
        self.sim = sim
        self.world = world
        self.path = list(path)
        self.stall_time, self.stall_dist, self.stall_yaw = stall_time, stall_dist, stall_yaw
        self.idx = sim.dof_indices(rb.BASE_JOINTS)

    def start(self) -> None:
        self.t = 0.0
        self.follower = PathFollower(self.path)
        self.sim.robot.switch_dof_control_mode("velocity", dof_indices=self.idx)
        self._ref = self.world.robot_pose()
        self._ref_t = 0.0

    def _hold(self) -> None:
        """速度を 0 にし、関節を位置制御に戻して今の姿勢で止める。"""
        art = self.sim.robot
        art.set_dof_velocity_targets([[0.0, 0.0, 0.0]], dof_indices=self.idx)
        q = art.get_dof_positions().numpy()[0]
        art.switch_dof_control_mode("position", dof_indices=self.idx)
        art.set_dof_position_targets([[float(q[i]) for i in self.idx]], dof_indices=self.idx)

    def step(self, dt: float) -> StepResult:
        self.t += dt
        pose = self.world.robot_pose()
        cmd = self.follower.command(pose)
        if cmd is None:
            self._hold()
            return StepResult(Status.SUCCEEDED)
        if math.hypot(pose.x - self._ref.x, pose.y - self._ref.y) > self.stall_dist or (
            abs(wrap_angle(pose.yaw - self._ref.yaw)) > self.stall_yaw
        ):
            self._ref, self._ref_t = pose, self.t
        elif self.t - self._ref_t > self.stall_time:
            self._hold()
            return StepResult(
                Status.FAILED, f"スタック（{self.stall_time:.0f} 秒進まない。({pose.x:.2f}, {pose.y:.2f}) 付近）"
            )
        self.sim.robot.set_dof_velocity_targets([list(cmd)], dof_indices=self.idx)
        return StepResult(Status.RUNNING)
