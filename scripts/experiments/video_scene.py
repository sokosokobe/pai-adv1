"""Issue #6 動画の検証用：Isaac Sim に依存する共通処理（#5 の部屋＋ RidgebackFranka、撮影用のカメラ、台車の経路）。

※ このモジュールは SimulationApp を生成した後で import すること。
"""

from __future__ import annotations

import math

from isaacsim.core.experimental.objects import Cone, Cube, Cylinder, Sphere
from isaacsim.core.experimental.prims import Articulation, XformPrim
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.sensors.experimental.rtx import RtxCamera

from experiments import arm_sim
from experiments import ridgeback_sim as rs
from experiments.arm_common import quat_from_yaw_wxyz
from experiments.ridgeback_common import velocity_command
from experiments.video_common import clamp_to_room, follow_camera_view, next_waypoint, smooth_point
from experiments.vlm_common import OBJECTS, ROOM_HALF, SURFACES, TIDY_LAYOUT, object_center

# 台車の経路（関節座標 = ワールド基準の x, y, yaw）。机・棚に当たらない部屋の手前側を往復する
START = (-0.3, -0.6, 0.0)
WAYPOINTS = [(0.8, -0.6, 0.0), (0.8, -0.6, math.pi / 2), (0.8, -0.6, 0.0), (-0.3, -0.6, 0.0)]
V_MAX, W_MAX, KP_LIN, KP_ANG = 0.5, 1.0, 2.0, 2.0  # #2 と同じ
POS_TOL, YAW_TOL = 0.03, math.radians(3)
# 撮影用のカメラ
CAMERAS = ("overview", "follow", "ceiling")
# 台車の経路（部屋の手前側）を正面から見るため、反対側（奥・左）の隅の天井付近に置く
OVERVIEW_EYE, OVERVIEW_TARGET = (-1.45, 1.05, 2.2), (0.35, -0.6, 0.3)
FOLLOW = {"back": 1.3, "side": 0.4, "height": 1.9, "look_ahead": 0.6, "look_height": 0.4}
FOLLOW_ALPHA = 0.15  # 追従カメラのなめらかさ（1 フレームで目標に近づく割合）
FOCAL_MM = {"overview": 14.0, "follow": 16.0, "ceiling": 12.0}  # 水平アパーチャ 20.955mm のとき
WALL_HEIGHT, WALL_T = 2.4, 0.05


def build_scene(simulation_app) -> None:
    """地面・ライト・ロボット（#2）と、#5 の部屋（見た目だけ。衝突なし）を作る。台車の x・y 関節の摩擦を 0 にする。"""
    rs.build_scene(simulation_app, "multiphysics")
    arm_sim.zero_base_joint_friction()
    hx, hy = ROOM_HALF
    # 地面（z=0）と重なってちらつかないよう、床の板は 2mm 上げる
    Cube(
        "/World/room/floor",
        sizes=1.0,
        scales=[[2 * hx, 2 * hy, 0.02]],
        positions=[[0, 0, -0.008]],
        colors=[[0.45, 0.45, 0.48]],
    )
    for name, pos, size in [
        ("north", [0, hy + WALL_T / 2, WALL_HEIGHT / 2], [2 * hx + 2 * WALL_T, WALL_T, WALL_HEIGHT]),
        ("south", [0, -hy - WALL_T / 2, WALL_HEIGHT / 2], [2 * hx + 2 * WALL_T, WALL_T, WALL_HEIGHT]),
        ("east", [hx + WALL_T / 2, 0, WALL_HEIGHT / 2], [WALL_T, 2 * hy, WALL_HEIGHT]),
        ("west", [-hx - WALL_T / 2, 0, WALL_HEIGHT / 2], [WALL_T, 2 * hy, WALL_HEIGHT]),
    ]:
        Cube(f"/World/room/wall_{name}", sizes=1.0, scales=[size], positions=[pos], colors=[[0.92, 0.9, 0.85]])
    for key, color in {"desk": [0.55, 0.36, 0.2], "shelf": [0.35, 0.25, 0.18]}.items():
        s = SURFACES[key]
        Cube(
            f"/World/room/{key}",
            sizes=1.0,
            scales=[[s.size[0], s.size[1], s.z]],
            positions=[[s.center[0], s.center[1], s.z / 2]],
            colors=[color],
        )
    for o in OBJECTS:
        path, color = f"/World/objects/{o.key}", [list(o.color)]
        pos = [list(object_center(o, TIDY_LAYOUT[o.key]))]
        if o.shape == "cube":
            yaw = [list(quat_from_yaw_wxyz(TIDY_LAYOUT[o.key].yaw))]
            Cube(path, sizes=1.0, scales=[list(o.size)], positions=pos, orientations=yaw, colors=color)
        elif o.shape == "cylinder":
            Cylinder(path, radii=o.size[0], heights=o.size[1], axes="Z", positions=pos, colors=color)
        elif o.shape == "cone":
            Cone(path, radii=o.size[0], heights=o.size[1], axes="Z", positions=pos, colors=color)
        else:
            Sphere(path, radii=o.size[0], positions=pos, colors=color)


def camera_path(name: str) -> str:
    return f"/World/cam_{name}"


def create_camera(name: str) -> RtxCamera:
    """撮影用のカメラの prim を作る。ceiling は #5 と同じく真下向き（回転なし）、overview は固定の視点。"""
    if name == "ceiling":
        rtx = RtxCamera(camera_path(name), positions=[[0.0, 0.0, 2.6]], orientations=[[1.0, 0.0, 0.0, 0.0]])
    else:
        rtx = RtxCamera(camera_path(name))
    rtx.camera.set_focal_lengths([FOCAL_MM[name] / 10])  # API の単位は USD の値の 1/10
    rtx.camera.set_apertures(horizontal_apertures=[2.0955])
    rtx.camera.set_clipping_ranges(near_distances=[0.05], far_distances=[30.0])
    if name == "overview":
        ViewportManager.set_camera_view(camera_path(name), eye=list(OVERVIEW_EYE), target=list(OVERVIEW_TARGET))
    return rtx


class BaseDriver:
    """台車を START に置き、経由点に沿って速度指令で動かす（#2 の推奨）。step() を物理の 1 ステップごとに呼ぶ。"""

    def __init__(self, simulation_app) -> None:
        self.art: Articulation = rs.articulation()
        names = list(self.art.dof_names)
        self.base_idx = [names.index(n) for n in rs.BASE_JOINTS]
        arm_sim.set_robot_configuration(simulation_app, self.art, START)
        self.art.switch_dof_control_mode("velocity", dof_indices=self.base_idx)
        self.base_link = XformPrim(rs.base_link_path())
        self.wp_index = 0

    def reinit(self) -> None:
        """物理がリセットされた後（Movie Capture の開始時など）に、Articulation を作り直して START の状態に戻す。

        アプリの更新（simulation_app.update）を呼ばないので、物理のコールバックの中から呼べる。
        """
        import numpy as np

        self.art = rs.articulation()
        names = list(self.art.dof_names)
        q = self.art.get_dof_positions().numpy()[0].copy()
        q[self.base_idx] = START
        for name, value in arm_sim.FRANKA_DEFAULT.items():
            q[names.index(name)] = value
        self.art.set_dof_positions(q)
        self.art.set_dof_velocities(np.zeros_like(q))
        self.art.set_dof_position_targets(q)
        self.art.switch_dof_control_mode("velocity", dof_indices=self.base_idx)
        self.wp_index = 0

    def joint_pose(self) -> tuple[float, float, float]:
        q = self.art.get_dof_positions().numpy()[0]
        return tuple(float(v) for v in q[self.base_idx])

    def step(self) -> tuple[float, float, float]:
        cur = self.joint_pose()
        self.wp_index = next_waypoint(cur, WAYPOINTS, self.wp_index, pos_tol=POS_TOL, yaw_tol=YAW_TOL)
        cmd = velocity_command(cur, WAYPOINTS[self.wp_index], kp_lin=KP_LIN, kp_ang=KP_ANG, v_max=V_MAX, w_max=W_MAX)
        self.art.set_dof_velocity_targets([list(cmd)], dof_indices=self.base_idx)
        return cur


class FollowCamera:
    """台車の後ろ上方から台車を追いかけるカメラ。update() を描画の前に呼ぶ。"""

    def __init__(self, driver: BaseDriver) -> None:
        self.driver = driver
        self.eye = self.target = None

    def update(self) -> None:
        eye, target = follow_camera_view(rs.pose2d(self.driver.base_link), **FOLLOW)
        eye = clamp_to_room(eye, half=ROOM_HALF, margin=0.1)
        self.eye = smooth_point(self.eye, eye, alpha=FOLLOW_ALPHA)
        self.target = smooth_point(self.target, target, alpha=FOLLOW_ALPHA)
        ViewportManager.set_camera_view(camera_path("follow"), eye=list(self.eye), target=list(self.target))
