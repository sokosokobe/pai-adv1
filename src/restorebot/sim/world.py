"""物体・ロボットの真値（WorldState）と天井カメラ（CeilingCamera）（設計書 7.1、5.4）。

※ SimulationApp を生成した後で import すること。
"""

from __future__ import annotations

import time

import numpy as np

from restorebot.core.scene import Placement
from restorebot.core.types import ObjectState, Observation, Pose2D, quat_wxyz_from_yaw, yaw_from_quat_wxyz
from restorebot.sim import robot as rb
from restorebot.sim.scene_builder import SimScene

# 物体を瞬間的に動かした後、描画が落ち着くまで進めるフレーム数。RTX の時間方向の処理で残像が残る（#5：30 では残った）
SETTLE_FRAMES = 150


class SimWorld:
    """シミュレータの真値を返す WorldState。"""

    def __init__(self, sim: SimScene) -> None:
        self.sim = sim

    def object_pose(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        """物体の中心の位置と向き (位置 (3,), wxyz (4,))。"""
        positions, orientations = self.sim.objects[key].get_world_poses()
        return positions.numpy()[0].astype(float), orientations.numpy()[0].astype(float)

    def object_state(self, key: str) -> ObjectState:
        pos, q = self.object_pose(key)
        scene = self.sim.scene
        surface = scene.surface_at(pos[:2])
        # 天板の上に置かれている（物体の中心が天板の高さ＋高さ/2 から 2cm 以内）ときだけ、面を返す
        expected_z = None if surface is None else surface.z + scene.object(key).height / 2
        on_surface = expected_z is not None and abs(pos[2] - expected_z) < 0.02
        return ObjectState(key, tuple(pos), yaw_from_quat_wxyz(q), surface.key if on_surface else None)

    def robot_pose(self) -> Pose2D:
        q = self.sim.robot.get_dof_positions().numpy()[0]
        x, y, yaw = (float(q[i]) for i in self.sim.dof_indices(rb.BASE_JOINTS))
        return Pose2D(x, y, yaw)

    def set_placements(self, placements: dict[str, Placement]) -> None:
        """物体を指定の配置に瞬間的に置く（シナリオの準備用。止まった状態にする）。"""
        scene = self.sim.scene
        for key, p in placements.items():
            body = self.sim.objects[key]
            center = scene.object_center(key, p)
            body.set_world_poses(
                positions=[[center[0], center[1], center[2] + 0.001]], orientations=[list(quat_wxyz_from_yaw(p.yaw))]
            )
            body.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))


class SimCeilingCamera:
    """天井カメラ（RGB＋深度）。capture() は描画が落ち着くまでシミュレーションを進めてから撮る。

    Observation.time_s は撮った時の壁時計の時刻（記録用）。シミュレーションの時刻ではない。
    """

    def __init__(self, sim: SimScene, simulation_app, *, settle_frames: int = SETTLE_FRAMES) -> None:
        self.sim = sim
        self.app = simulation_app
        self.settle_frames = settle_frames

    def capture(self) -> Observation:
        for _ in range(self.settle_frames):
            self.app.update()
        sensor = self.sim.ceiling
        for _ in range(120):
            rgb, _ = sensor.get_data("rgb")
            depth, _ = sensor.get_data("distance_to_image_plane")
            if rgb is not None and depth is not None:
                break
            self.app.update()
        else:
            raise RuntimeError("天井カメラの画像を取得できませんでした")
        return Observation(
            rgb=np.ascontiguousarray(rgb.numpy()[:, :, :3], dtype=np.uint8),
            depth=depth.numpy()[:, :, 0].astype(np.float32),
            time_s=time.time(),
        )
