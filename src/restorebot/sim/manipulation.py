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
    GripMonitor,
    Gripper,
    Stage,
    StageSequencer,
    bearing_yaw,
    choose_grasp_yaw,
    hand_down_quat_wxyz,
    judge_grip,
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
        grip_width: float | None = None,
        on_end: Callable[[StepResult], None] | None = None,
    ) -> None:
        """grip_width：持っている物体の掴む幅。与えると、物を持って動く段階（指を閉じたまま動く段階）で指の開きを監視し、
        落としたら失敗にする（#27）。on_end：終わったときに最後の結果を受け取る（記録用）。
        """
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
        self.on_end = on_end
        self.grip_opening: float | None = None  # 最後に測った指の開き（左右の指の関節の値の和）[m]
        self._grip = None if grip_width is None else GripMonitor(grip_width, ctx.params)

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
        fingers = _finger_positions(state)
        self.grip_opening = sum(fingers)
        if self._grip is not None and stage.gripper is Gripper.CLOSE and not stage.wait_gripper:
            dropped = self._grip.update(self.t, self.grip_opening)  # 物を持って動く段階：落としていないか
            if dropped is not None:
                return self._finish(Status.FAILED, f"{stage.name}：{dropped}")
        gripper_done = gripper.is_complete(state)
        if stage.gripper is Gripper.CLOSE:
            # 物を挟んで止まった指は速度の値が振れるので、位置が止まったことでも閉じ終わりとみなす（#25）
            gripper_done = self._settle.update(self.t, fingers) or gripper_done
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
        if self.on_finish and status is Status.SUCCEEDED:
            result = self.on_finish(result.status)
        if self.on_end is not None:
            self.on_end(result)
        return result


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
    record: Callable[[dict], None] | None = None,
) -> StagedAction:
    """物体 obj を掴んで持ち上げ、待機姿勢へ。

    掴めたかどうかは、ロボット自身の指の開きで判定する（judge_grip。#27）。持ち上げ・待機姿勢へ動く間も監視し、
    落としたら失敗。真値（物体の高さ object_z が LIFT_MIN 以上上がったか）は、判定の比較のために record に渡すだけ。
    手先の向きは、台車から物体を見た方向を基準に、隣の物体 neighbors に手が当たらない向きを選ぶ（choose_grasp_yaw）。
    """
    spec = scene.object(obj.key)
    xy = obj.position[:2]
    yaw = choose_grasp_yaw(spec.shape, obj.yaw, bearing_yaw(base, xy), xy, neighbors, ctx.params)
    z0 = obj.position[2]
    width = spec.grasp_width

    def check(_: Status) -> StepResult:
        reason = judge_grip(action.grip_opening, width, ctx.params)
        if reason is not None:
            return StepResult(Status.FAILED, f"掴み損ね（{reason}）")
        return StepResult(Status.SUCCEEDED)

    def end(result: StepResult) -> None:
        if record is None:
            return
        lift = float(object_z() - z0)  # 真値の世界では z0 が numpy の値（記録は JSON にするので Python の型にする）
        record(
            {
                "phase": "pick",
                "key": obj.key,
                "opening_m": action.grip_opening,
                "grip_ok": result.status is Status.SUCCEEDED,
                "reason": result.reason,
                "truth_lift_m": lift,
                "truth_ok": bool(lift >= LIFT_MIN),
            }
        )

    stages = pick_stages(obj.position, yaw, base, ctx.params)
    action = StagedAction(
        f"pick:{obj.key}", ctx, stages, obstacle_prim=obstacle_prim, on_finish=check, grip_width=width, on_end=end
    )
    return action


def place_action(
    ctx: ArmContext,
    shape: str,
    position: Sequence[float],
    hand_yaw: float,
    base: Pose2D,
    obstacle_prim: str,
    neighbors: Sequence[Footprint] = (),
    grip_width: float | None = None,
) -> StagedAction:
    """掴んでいる物体を position（置いたときの物体の中心）に置き、待機姿勢へ。

    grip_width（物体の掴む幅）を与えると、指を開くまでの間に指の開きを監視し、運ぶ途中で落としていたら失敗（#27）。

    物体の向きは変えない（設計書 18 章 Q3）。箱は掴んだときの手先の向き hand_yaw か、それを 180° 回した向き
    （箱の形は同じ）のうち、台車から置く位置を見た方向に近い方。円柱は軸のまわりに回しても見た目が変わらないので、
    置く位置の隣の物体に手が当たらない向きを選び直す（#13）。
    """
    xy = position[:2]
    yaw = choose_grasp_yaw(shape, hand_yaw, bearing_yaw(base, xy), xy, neighbors, ctx.params)
    stages = place_stages(position, yaw, base, ctx.params)
    name = f"place:{obstacle_prim.rsplit('/', 1)[-1]}"
    return StagedAction(name, ctx, stages, obstacle_prim=obstacle_prim, grip_width=grip_width)


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
