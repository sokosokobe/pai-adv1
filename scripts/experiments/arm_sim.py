"""Issue #3 アームの検証用：Isaac Sim / cuMotion に依存する共通処理。

※ このモジュールは SimulationApp を生成した後で import すること。

ポイント（docs/experiments/3_arm_motion.md 参照）:
- cuMotion 同梱の Franka 設定（`load_cumotion_supported_robot("franka")`）を RidgebackFranka のアームに使う
- cuMotion の「ロボットの根元」は Franka の `base_link`（= `panda_link0` と同じ位置・向き）。RidgebackFranka では
  `panda_link0` のワールド姿勢を `update_world_to_robot_root_transforms` に渡し、台車が動くたびに更新する
- cuMotion の衝突モデルはアームだけなので、台車本体を箱の障害物として追加し、台車と一緒に動かす
- `robot_joint_space` には全 DOF の名前を渡す。計画結果のインデックスがそのまま Articulation の DOF 番号になる
"""

from __future__ import annotations

import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import warp as wp
from isaacsim.core.experimental.prims import Articulation, XformPrim
from isaacsim.robot_motion.cumotion import CumotionWorldInterface, load_cumotion_supported_robot
from pxr import Usd, UsdGeom

from experiments import ridgeback_sim as rs
from experiments.arm_common import compose_pose

TOOL_FRAME = "panda_hand"
HAND_DOWN_WXYZ = (0.0, 1.0, 0.0, 0.0)  # panda_hand の z 軸が下向き（Pick and Place サンプルと同じ）
# cuMotion / Pick and Place サンプルの Franka の既定姿勢（アームを畳んだ姿勢）
FRANKA_DEFAULT = {
    "panda_joint1": 0.012,
    "panda_joint2": -0.568,
    "panda_joint3": 0.0,
    "panda_joint4": -2.811,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}
BASE_OBSTACLE_PATH = "/World/virtual_ridgeback_base_obstacle"  # cuMotion 内だけの仮想の障害物（USD には作らない）


def zero_base_joint_friction() -> None:
    """#2 の決定事項：台車の x・y 関節の physxJoint:jointFriction を 0 にする（物理開始前に呼ぶ）。"""
    for joint_name in rs.BASE_JOINTS[:2]:
        rs.find_joint_prim(joint_name).GetAttribute("physxJoint:jointFriction").Set(0.0)


def set_robot_configuration(simulation_app, art: Articulation, base_pose2d, steps: int = 30) -> None:
    """台車を base_pose2d (x, y, yaw) に置き、アームを既定姿勢にして保持する（台車は位置制御）。"""
    names = list(art.dof_names)
    base_idx = [names.index(n) for n in rs.BASE_JOINTS]
    q = art.get_dof_positions().numpy()[0].copy()
    q[base_idx] = base_pose2d
    for name, value in FRANKA_DEFAULT.items():
        q[names.index(name)] = value
    art.switch_dof_control_mode("position", dof_indices=base_idx)
    art.set_dof_positions(q)
    art.set_dof_velocities(np.zeros_like(q))
    art.set_dof_position_targets(q)
    for _ in range(steps):
        simulation_app.update()


def _pose_wp(position, quat_wxyz) -> tuple[wp.array, wp.array]:
    return (
        wp.array(np.asarray([position], dtype=np.float32), dtype=wp.float32, device="cpu"),
        wp.array(np.asarray([quat_wxyz], dtype=np.float32), dtype=wp.float32, device="cpu"),
    )


class ArmPlanning:
    """cuMotion の world interface とロボット設定をまとめたもの。"""

    def __init__(self, art: Articulation, *, include_base_obstacle: bool = True) -> None:
        self.art = art
        self.cumotion_robot = load_cumotion_supported_robot("franka")
        self.joint_space = list(art.dof_names)
        self.link0 = XformPrim(rs.link_path("panda_link0"))
        self.hand = XformPrim(rs.link_path(TOOL_FRAME))
        self.base = XformPrim(rs.base_link_path())

        # ---- 周囲の衝突物（地面・机など）を world interface に同期。ロボット自身は除く ----
        robot_pos, _ = rs.world_pose(self.base)
        objects = mg.SceneQuery().get_prims_in_aabb(
            search_box_origin=robot_pos,
            search_box_minimum=[-10.0, -10.0, -10.0],
            search_box_maximum=[10.0, 10.0, 10.0],
            tracked_api=mg.TrackableApi.PHYSICS_COLLISION,
            exclude_prim_paths=[rs.ROBOT_PATH],
        )
        self.tracked_prims = list(objects)
        self.world_interface = CumotionWorldInterface(device="cpu")
        self.binding = mg.WorldBinding(
            world_interface=self.world_interface,
            obstacle_strategy=mg.ObstacleStrategy(),
            tracked_prims=objects,
            tracked_collision_api=mg.TrackableApi.PHYSICS_COLLISION,
        )
        self.binding.initialize()

        # ---- 台車本体を箱の障害物として追加（台車リンク座標系での中心・大きさ）----
        self.include_base_obstacle = include_base_obstacle
        self.base_box_center_local, self.base_box_size = self._base_box_in_base_frame()
        if include_base_obstacle:
            pos, quat = self._base_box_world_pose()
            self.world_interface.add_cubes(
                [BASE_OBSTACLE_PATH],
                sizes=wp.array([1.0], dtype=wp.float32, device="cpu"),
                scales=wp.array(np.asarray([self.base_box_size], dtype=np.float32), dtype=wp.float32, device="cpu"),
                safety_tolerances=wp.array([0.0], dtype=wp.float32, device="cpu"),
                poses=_pose_wp(pos, quat),
                enabled_array=wp.array([True], dtype=wp.bool, device="cpu"),
            )
        self.sync()

    # ------------------------------------------------------------
    def _base_box_in_base_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """台車リンクの形状の AABB（台車リンク座標系）。上面はアームの根元より 2cm 下に下げる。"""
        prim = self.base.prims[0]
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], useExtentsHint=True)
        rng = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
        lo, hi = np.array(rng.GetMin(), dtype=float), np.array(rng.GetMax(), dtype=float)
        # アームの根元（panda_link0）の高さを台車リンク座標系で求め、箱の上面がそれを超えないようにする
        base_pos, base_q = rs.world_pose(self.base)
        link0_pos, _ = rs.world_pose(self.link0)
        rel = link0_pos - base_pos
        w, x, y, z = base_q
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        link0_local = rot.T @ rel
        hi[2] = min(hi[2], link0_local[2] - 0.02)
        return (lo + hi) / 2, hi - lo

    def _base_box_world_pose(self) -> tuple[tuple, tuple]:
        base_pos, base_q = rs.world_pose(self.base)
        return compose_pose((tuple(base_pos), tuple(base_q)), (tuple(self.base_box_center_local), (1.0, 0.0, 0.0, 0.0)))

    def arm_base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """cuMotion に渡すロボットの根元（panda_link0）のワールド姿勢。"""
        return rs.world_pose(self.link0)

    def sync(self) -> None:
        """台車が動いた後に呼ぶ：根元の姿勢・台車の障害物・周囲の物体の位置を world interface に反映する。"""
        pos, quat = self.arm_base_pose()
        self.world_interface.update_world_to_robot_root_transforms(_pose_wp(pos, quat))
        if self.include_base_obstacle:
            bpos, bquat = self._base_box_world_pose()
            self.world_interface.update_obstacle_transforms([BASE_OBSTACLE_PATH], _pose_wp(bpos, bquat))
        self.binding.synchronize_transforms()

    def solve_ik(self, position_world, quat_wxyz, seed=None) -> np.ndarray | None:
        """衝突を考えた IK（cuMotion の CollisionFreeIkSolver）。解けなければ None。

        Returns:
            cuMotion が制御する関節（`cumotion_robot.controlled_joint_names` の順）の角度
        """
        import cumotion
        from isaacsim.robot_motion.cumotion.impl.utils import isaac_sim_to_cumotion_pose

        base_pos, base_q = self.arm_base_pose()
        target = isaac_sim_to_cumotion_pose(
            position_world_to_target=np.asarray(position_world, dtype=float),
            orientation_world_to_target=np.asarray(quat_wxyz, dtype=float),
            position_world_to_base=base_pos,
            orientation_world_to_base=base_q,
        )
        self.world_interface.world_view.update()
        config = cumotion.create_default_collision_free_ik_solver_config(
            self.cumotion_robot.robot_description, TOOL_FRAME, self.world_interface.world_view
        )
        solver = cumotion.create_collision_free_ik_solver(config)
        ik_target = cumotion.CollisionFreeIkSolver.TaskSpaceTarget(
            cumotion.CollisionFreeIkSolver.TranslationConstraint.target(target.translation),
            cumotion.CollisionFreeIkSolver.OrientationConstraint.target(target.rotation),
        )
        seeds = [np.asarray(seed, dtype=float)] if seed is not None else []
        res = solver.solve(ik_target, seeds)
        if res.status() != cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS:
            return None
        return np.asarray(res.cspace_positions()[0], dtype=float).reshape(-1)

    # ------------------------------------------------------------
    def tool_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return rs.world_pose(self.hand)

    def robot_state(self) -> mg.RobotState:
        """現在の関節状態と手先の姿勢（RmpFlowController などに渡す）。"""
        hp, hq = self.tool_pose()
        js = self.joint_space
        return mg.RobotState(
            joints=mg.JointState.from_name(
                robot_joint_space=js,
                positions=(js, self.art.get_dof_positions()),
                velocities=(js, self.art.get_dof_velocities()),
            ),
            sites=mg.SpatialState.from_name(
                spatial_space=[TOOL_FRAME],
                positions=([TOOL_FRAME], wp.array(np.asarray([hp], dtype=np.float32), device="cpu")),
                orientations=([TOOL_FRAME], wp.array(np.asarray([hq], dtype=np.float32), device="cpu")),
            ),
        )

    def target_state(self, position, quat_wxyz) -> mg.RobotState:
        """手先（panda_hand）のワールド座標の目標。"""
        return mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=[TOOL_FRAME],
                positions=([TOOL_FRAME], wp.array(np.asarray([position], dtype=np.float32), device="cpu")),
                orientations=([TOOL_FRAME], wp.array(np.asarray([quat_wxyz], dtype=np.float32), device="cpu")),
            )
        )

    def apply(self, desired: mg.RobotState | None) -> None:
        """コントローラ／軌道の出力を関節の位置・速度の目標として与える。"""
        if desired is None or desired.joints is None:
            return
        joints = desired.joints
        if joints.positions is not None:
            self.art.set_dof_position_targets(joints.positions, dof_indices=joints.position_indices)
        if joints.velocities is not None:
            self.art.set_dof_velocity_targets(joints.velocities, dof_indices=joints.velocity_indices)
