"""cuMotion によるアームの動作計画（設計書 9.1）。#3 の scripts/experiments/arm_sim.py の ArmPlanning を移した。

※ SimulationApp を生成した後で import すること。

RidgebackFranka で cuMotion の Franka 設定を使うための 4 つの処理（#3）:
1. `panda_link0` のワールド姿勢をロボットの根元として渡し、台車が動くたびに更新する（sync()）
2. `robot_joint_space` に全 DOF の名前を渡す（計画結果のインデックスがそのまま Articulation の DOF 番号になる）
3. tool frame に `panda_hand` を明示する
4. 台車本体を箱の障害物として加え、台車と一緒に動かす
台車が動いたら必ず sync() を呼ぶ。忘れると 20cm・10° ずれる（#3）。
"""

from __future__ import annotations

import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import warp as wp
from isaacsim.core.experimental.objects import Cone, Cylinder, Mesh
from isaacsim.core.experimental.prims import Articulation, XformPrim
from isaacsim.robot_motion.cumotion import CumotionWorldInterface, load_cumotion_supported_robot
from pxr import Usd, UsdGeom

from restorebot.sim.robot import ROBOT_PATH, TOOL_FRAME, base_link_path, link_path, world_pose

BASE_OBSTACLE_PATH = "/World/virtual_ridgeback_base_obstacle"  # cuMotion の中だけの仮想の障害物（USD には作らない）


def _rotation_matrix_wxyz(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _obstacle_strategy() -> mg.ObstacleStrategy:
    """6.1 の Pick and Place サンプルと同じ近似（メッシュ・円錐・円柱は向き付きの箱、余裕 1cm）。"""
    strategy = mg.ObstacleStrategy()
    for shape in (Mesh, Cone, Cylinder):
        strategy.set_default_configuration(shape, mg.ObstacleConfiguration("obb", 0.01))
    return strategy


def _pose_wp(position, quat_wxyz) -> tuple[wp.array, wp.array]:
    return (
        wp.array(np.asarray([position], dtype=np.float32), dtype=wp.float32, device="cpu"),
        wp.array(np.asarray([quat_wxyz], dtype=np.float32), dtype=wp.float32, device="cpu"),
    )


class ArmPlanning:
    """cuMotion の world interface とロボットの設定をまとめたもの。"""

    def __init__(self, art: Articulation) -> None:
        self.art = art
        self.cumotion_robot = load_cumotion_supported_robot("franka")
        self.joint_space = list(art.dof_names)
        self.link0 = XformPrim(link_path("panda_link0"))
        self.hand = XformPrim(link_path(TOOL_FRAME))
        self.base = XformPrim(base_link_path())

        # ---- 周囲の衝突物（地面・家具・物体・壁）を world interface に同期。ロボット自身は除く ----
        robot_pos, _ = world_pose(self.base)
        objects = mg.SceneQuery().get_prims_in_aabb(
            search_box_origin=robot_pos,
            search_box_minimum=[-10.0, -10.0, -10.0],
            search_box_maximum=[10.0, 10.0, 10.0],
            tracked_api=mg.TrackableApi.PHYSICS_COLLISION,
            exclude_prim_paths=[ROBOT_PATH],
        )
        self.tracked_prims = list(objects)
        self.world_interface = CumotionWorldInterface(device="cpu")
        self.binding = mg.WorldBinding(
            world_interface=self.world_interface,
            obstacle_strategy=_obstacle_strategy(),
            tracked_prims=objects,
            tracked_collision_api=mg.TrackableApi.PHYSICS_COLLISION,
        )
        self.binding.initialize()

        # ---- 台車本体を箱の障害物として追加（台車リンク座標系での中心・大きさ）----
        self.base_box_center_local, self.base_box_size = self._base_box_in_base_frame()
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

    def _base_box_in_base_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """台車リンクの形状の AABB（台車リンク座標系）。上面はアームの根元より 2cm 下に下げる。"""
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], useExtentsHint=True)
        rng = cache.ComputeUntransformedBound(self.base.prims[0]).ComputeAlignedRange()
        lo, hi = np.array(rng.GetMin(), dtype=float), np.array(rng.GetMax(), dtype=float)
        base_pos, base_q = world_pose(self.base)
        link0_pos, _ = world_pose(self.link0)
        link0_local = _rotation_matrix_wxyz(base_q).T @ (link0_pos - base_pos)
        hi[2] = min(hi[2], link0_local[2] - 0.02)
        return (lo + hi) / 2, hi - lo

    def _base_box_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        base_pos, base_q = world_pose(self.base)
        return base_pos + _rotation_matrix_wxyz(base_q) @ self.base_box_center_local, base_q

    def set_obstacles_enabled(self, prim_path: str, enabled: bool) -> None:
        """prim_path 以下の追跡中の障害物を有効／無効にする（掴む物体を一時的に外すため）。"""
        root = prim_path.rstrip("/")
        paths = [p for p in self.tracked_prims if p == root or p.startswith(root + "/")]
        if not paths:
            raise ValueError(f"追跡中の障害物に {prim_path} がありません")
        self.world_interface.update_obstacle_enables(
            prim_paths=paths, enabled_array=wp.array([enabled] * len(paths), dtype=wp.bool, device="cpu")
        )

    def sync(self) -> None:
        """台車が動いた後に呼ぶ：根元の姿勢・台車の障害物・周囲の物体の位置を world interface に反映する。"""
        pos, quat = world_pose(self.link0)
        self.world_interface.update_world_to_robot_root_transforms(_pose_wp(pos, quat))
        bpos, bquat = self._base_box_world_pose()
        self.world_interface.update_obstacle_transforms([BASE_OBSTACLE_PATH], _pose_wp(bpos, bquat))
        self.binding.synchronize_transforms()

    def tool_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """手先（panda_hand）のワールド姿勢 (位置, wxyz)。"""
        return world_pose(self.hand)

    def robot_state(self) -> mg.RobotState:
        """今の関節の状態と手先の姿勢（RmpFlowController などに渡す）。"""
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
        """コントローラの出力を、関節の位置・速度の目標として与える。"""
        if desired is None or desired.joints is None:
            return
        joints = desired.joints
        if joints.positions is not None:
            self.art.set_dof_position_targets(joints.positions, dof_indices=joints.position_indices)
        if joints.velocities is not None:
            self.art.set_dof_velocity_targets(joints.velocities, dof_indices=joints.velocity_indices)
