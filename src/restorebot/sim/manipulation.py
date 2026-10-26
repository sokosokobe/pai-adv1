# SPDX-License-Identifier: Apache-2.0
#
# Isaac Sim 6.1 の Pick and Place サンプルの PickPlaceController（isaacsim.robot_motion.examples、
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES、Apache License 2.0）の構成（RmpFlowController ＋
# JointGripperController を段階ごとに切り替え、段階が変わるたびに両方をリセットする）を、
# 「掴む」と「置く」の 2 つの Action に分けて移植したもの。段階の列と進める判定は restorebot.core.grasp。
"""掴む・置く・待機姿勢へ動かす Action（設計書 9.2）。

※ SimulationApp を生成した後で import すること。
Action は全体の制御が物理の 1 ステップごとに step(dt) を呼ぶ形（設計書 3.3）。step の中ではシミュレーションを進めない。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import isaacsim.core.experimental.utils.app as app_utils
import numpy as np
from isaacsim.robot_motion.cumotion import RmpFlowController

from restorebot.core.grasp import (
    GRASP_OFFSET_Z,
    FingerSettle,
    Footprint,
    GraspParams,
    Gripper,
    Stage,
    StageSequencer,
    bearing_yaw,
    choose_grasp_yaw,
    hand_down_quat_wxyz,
    move_toward,
    pick_stages,
    place_stages,
    quat_angle_wxyz,
    standby_grasp_position,
)
from restorebot.core.scene import SceneSpec
from restorebot.core.types import ObjectState, Pose2D, Status, StepResult
from restorebot.sim import robot as rb
from restorebot.sim.arm import ArmPlanning

app_utils.enable_extension("isaacsim.robot_motion.examples")
from isaacsim.robot_motion.examples.manipulation.controllers import (  # noqa: E402
    GripperCommand,
    JointGripperController,
)

LIFT_MIN = 0.03  # 掴めたとみなす持ち上げの高さ [m]（#4 の classify_trial と同じ）


class ArmContext:
    """アームの動作計画（ArmPlanning）と、指の開閉のコントローラ。Action どうしで共有する。"""

    def __init__(self, planning: ArmPlanning, params: GraspParams | None = None) -> None:
        self.planning = planning
        self.params = params or GraspParams()
        common = {
            "robot_joint_space": planning.joint_space,
            "joint_names": rb.FINGER_JOINTS,
            "open_positions": (rb.FINGER_OPEN, rb.FINGER_OPEN),
            "closed_positions": (rb.FINGER_CLOSED, rb.FINGER_CLOSED),
            "velocity_tolerance": 0.1,  # サンプルの Franka の設定と同じ
        }
        self.grippers = {
            Gripper.OPEN: JointGripperController(**common, command=GripperCommand.OPEN),
            Gripper.CLOSE: JointGripperController(**common, command=GripperCommand.CLOSE),
        }

    def make_arm_controller(self) -> RmpFlowController:
        p = self.planning
        return RmpFlowController(
            cumotion_robot=p.cumotion_robot,
            cumotion_world_interface=p.world_interface,
            robot_joint_space=p.joint_space,
            robot_site_space=[rb.TOOL_FRAME],
            tool_frame=rb.TOOL_FRAME,
        )


class StagedAction:
    """段階の列を、手先の目標（RMPflow）と指の開閉で順にたどる Action。"""

    def __init__(
        self,
        name: str,
        ctx: ArmContext,
        stages: Sequence[Stage],
        *,
        obstacle_prim: str | None = None,
        on_finish: Callable[[Status], StepResult] | None = None,
    ) -> None:
        self.name = name
        self.ctx = ctx
        self.seq = StageSequencer(stages, ctx.params)
        self.obstacle_prim = obstacle_prim  # 掴む物体（段階に応じて動作計画の障害物から外す）
        self.on_finish = on_finish
        self.t = 0.0
        self._arm = ctx.make_arm_controller()
        self._stage_index = -1
        self._obstacle_enabled = True
        self.log: list[tuple[str, float]] = []  # (段階, 開始時刻) の記録
        self.hand_yaw = stages[0].yaw  # 最初の段階の手先の向き（pick では掴むときの向き。place に渡す）
        self._cmd_stage = -1  # 手先の目標を少しずつ動かしている段階
        self._cmd_pos: tuple[float, float, float] | None = None
        self._settle = FingerSettle((rb.FINGER_OPEN, rb.FINGER_OPEN))

    def start(self) -> None:
        self.t = 0.0
        self.seq.start(0.0)
        self._stage_index = -1
        self._cmd_stage = -1

    def _set_obstacle(self, enabled: bool) -> None:
        if self.obstacle_prim is not None and enabled != self._obstacle_enabled:
            self.ctx.planning.set_obstacles_enabled(self.obstacle_prim, enabled)
            self._obstacle_enabled = enabled

    def _hand_target(self, stage: Stage) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        # 手先は真下向きなので、panda_hand は掴む位置の GRASP_OFFSET_Z 上
        x, y, z = stage.position
        return (x, y, z + GRASP_OFFSET_Z), hand_down_quat_wxyz(stage.yaw)

    def step(self, dt: float) -> StepResult:
        planning = self.ctx.planning
        self.t += dt
        if self._stage_index != self.seq.index:  # 段階が変わった：アームと指のコントローラをリセットする
            self._stage_index = self.seq.index
            stage = self.seq.stage
            self.log.append((stage.name, round(self.t, 3)))
            self._set_obstacle(stage.object_is_obstacle)
            planning.sync()
            state = planning.robot_state()
            if not self._arm.reset(state, None, self.t) or not self.ctx.grippers[stage.gripper].reset(
                state, None, self.t
            ):
                return self._finish(Status.FAILED, f"{stage.name}：コントローラのリセットに失敗")
            self._settle.reset()
        stage = self.seq.stage
        gripper = self.ctx.grippers[stage.gripper]
        planning.sync()
        state = planning.robot_state()
        target_pos, target_q = self._hand_target(stage)
        hand_pos, hand_q = planning.tool_pose()
        gripper_done = gripper.is_complete(state)
        if stage.gripper is Gripper.CLOSE:
            # 物を挟んで止まった指は速度の値が振れるので、位置が止まったことでも閉じ終わりとみなす（#25）
            gripper_done = self._settle.update(self.t, _finger_positions(state)) or gripper_done
        status = self.seq.update(
            self.t,
            float(np.linalg.norm(hand_pos - np.asarray(target_pos))),
            quat_angle_wxyz(hand_q, target_q),
            gripper_done,
        )
        if status is not Status.RUNNING:
            return self._finish(status, self.seq.failure_reason)
        stage = self.seq.stage  # 次の段階に進んだ場合は、次の目標で動かす
        target_pos, target_q = self._hand_target(stage)
        if stage.max_speed is not None:
            # 物を持って動く段階：手先の目標を、今の手先の位置から max_speed で少しずつ動かす（速く振らない）
            if self._cmd_stage != self.seq.index:
                self._cmd_stage = self.seq.index
                self._cmd_pos = tuple(float(c) for c in hand_pos)
            self._cmd_pos = move_toward(self._cmd_pos, target_pos, stage.max_speed * dt)
            target_pos = self._cmd_pos
        planning.apply(self._arm.forward(state, planning.target_state(target_pos, target_q), self.t))
        planning.apply(self.ctx.grippers[stage.gripper].forward(state, None, self.t))
        return StepResult(Status.RUNNING)

    def _finish(self, status: Status, reason: str | None) -> StepResult:
        if status is Status.FAILED:
            self._set_obstacle(True)  # 失敗したら障害物に戻す
        result = StepResult(status, reason)
        return self.on_finish(result.status) if self.on_finish and status is Status.SUCCEEDED else result


def _finger_positions(state) -> tuple[float, ...]:
    joints = state.joints
    q = joints.positions.numpy()
    return tuple(float(q[joints.position_names.index(n)]) for n in rb.FINGER_JOINTS)


def pick_action(
    ctx: ArmContext,
    scene: SceneSpec,
    obj: ObjectState,
    base: Pose2D,
    object_z: Callable[[], float],
    obstacle_prim: str,
    neighbors: Sequence[Footprint] = (),
) -> StagedAction:
    """物体 obj を掴んで持ち上げ、待機姿勢へ。最後に物体が LIFT_MIN 以上持ち上がっていなければ失敗（掴み損ね）。

    手先の向きは、台車から物体を見た方向を基準に、隣の物体 neighbors に手が当たらない向きを選ぶ（choose_grasp_yaw）。
    """
    xy = obj.position[:2]
    yaw = choose_grasp_yaw(scene.object(obj.key).shape, obj.yaw, bearing_yaw(base, xy), xy, neighbors, ctx.params)
    z0 = obj.position[2]

    def check(_: Status) -> StepResult:
        lift = object_z() - z0
        if lift < LIFT_MIN:
            return StepResult(Status.FAILED, f"掴み損ね（持ち上がった高さ {lift:.3f}m）")
        return StepResult(Status.SUCCEEDED)

    stages = pick_stages(obj.position, yaw, base, ctx.params)
    return StagedAction(f"pick:{obj.key}", ctx, stages, obstacle_prim=obstacle_prim, on_finish=check)


def place_action(
    ctx: ArmContext,
    shape: str,
    position: Sequence[float],
    hand_yaw: float,
    base: Pose2D,
    obstacle_prim: str,
    neighbors: Sequence[Footprint] = (),
) -> StagedAction:
    """掴んでいる物体を position（置いたときの物体の中心）に置き、待機姿勢へ。

    物体の向きは変えない（設計書 18 章 Q3）。箱は掴んだときの手先の向き hand_yaw か、それを 180° 回した向き
    （箱の形は同じ）のうち、台車から置く位置を見た方向に近い方。円柱は軸のまわりに回しても見た目が変わらないので、
    置く位置の隣の物体に手が当たらない向きを選び直す（#13）。
    """
    xy = position[:2]
    yaw = choose_grasp_yaw(shape, hand_yaw, bearing_yaw(base, xy), xy, neighbors, ctx.params)
    stages = place_stages(position, yaw, base, ctx.params)
    return StagedAction(f"place:{obstacle_prim.rsplit('/', 1)[-1]}", ctx, stages, obstacle_prim=obstacle_prim)


def standby_action(ctx: ArmContext, base: Pose2D, gripper: Gripper = Gripper.OPEN) -> StagedAction:
    """待機姿勢へ動かす（台車座標で手先 (0.4, 0, 0.95)、真下向き）。"""
    stage = Stage(
        "standby", standby_grasp_position(base), base.yaw, gripper, False, True, ctx.params.timeouts["standby"]
    )
    return StagedAction("standby", ctx, [stage])


def run_action(simulation_app, action, dt: float, *, timeout_s: float = 60.0, on_step=None) -> StepResult:
    """Action を終わるまで（または timeout_s まで）1 ステップずつ進める（確認用・テスト用）。"""
    action.start()
    steps = int(math.ceil(timeout_s / dt))
    for i in range(steps):
        result = action.step(dt)
        if result.status is not Status.RUNNING:
            return result
        simulation_app.update()
        if on_step is not None:
            on_step(i)
    return StepResult(Status.FAILED, f"{action.name}：{timeout_s:.0f} 秒で終わらなかった")
