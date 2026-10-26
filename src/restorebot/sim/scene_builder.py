"""SceneSpec から Isaac Sim のシーン（床・壁・家具・物体・ロボット・天井カメラ）を作る（設計書 5 章）。

※ SimulationApp を生成した後で import すること。#5・#9 の vlm_scene.py と #4 の grasp_place.py から移した。
家具は衝突ありの固定物（剛体なし）、物体は剛体（摩擦 0.8 / 0.6）。物理は 60Hz。
"""

from __future__ import annotations

from dataclasses import dataclass

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.materials import RigidBodyMaterial
from isaacsim.core.experimental.objects import Cube, Cylinder, DomeLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

from restorebot.core.scene import Placement, SceneSpec
from restorebot.core.types import Pose2D, quat_wxyz_from_yaw
from restorebot.sim import robot as rb

PHYSICS_DT = 1.0 / 60.0
WALL_HEIGHT = 2.4
WALL_THICKNESS = 0.05
FLOOR_TOP = 0.002  # 見た目の床の板は、ロボットが乗る地面（衝突あり）と重ならないよう 2mm 上げる（#10）
OBJECT_FRICTION = (0.8, 0.6)  # 静止・動（#4）
FURNITURE_COLORS = {"desk": [0.55, 0.36, 0.2], "side_table": [0.35, 0.25, 0.18]}
OBJECTS_ROOT = "/World/objects"


@dataclass
class SimScene:
    scene: SceneSpec
    objects: dict[str, RigidPrim]  # key → 物体の剛体
    robot: Articulation
    ceiling: CameraSensor
    dof_names: list[str]

    def dof_indices(self, names) -> list[int]:
        return [self.dof_names.index(n) for n in names]


def object_path(key: str) -> str:
    return f"{OBJECTS_ROOT}/{key}"


def _collider(shape) -> GeomPrim:
    return GeomPrim(shape.paths, apply_collision_apis=True)


def build_scene(simulation_app, scene: SceneSpec, placements: dict[str, Placement], base_pose: Pose2D) -> SimScene:
    """シーンを作り、物理を開始して、台車を base_pose に置く（台車の関節は位置制御で保持。アームは待機姿勢）。"""
    hx, hy = scene.room_half
    GroundPlane("/World/ground_plane")
    # 真下からの平行光源は床に光源の映り込み（白飛び）を作ったので、ドームライトだけにする（#5）
    DomeLight("/World/dome_light").set_intensities(1000)
    Cube(
        "/World/room/floor",
        sizes=1.0,
        scales=[[2 * hx, 2 * hy, 0.02]],
        positions=[[0, 0, FLOOR_TOP - 0.01]],
        colors=[[0.45, 0.45, 0.48]],
    )
    t = WALL_THICKNESS
    walls = {
        "north": ([0, hy + t / 2, WALL_HEIGHT / 2], [2 * hx + 2 * t, t, WALL_HEIGHT]),
        "south": ([0, -hy - t / 2, WALL_HEIGHT / 2], [2 * hx + 2 * t, t, WALL_HEIGHT]),
        "east": ([hx + t / 2, 0, WALL_HEIGHT / 2], [t, 2 * hy, WALL_HEIGHT]),
        "west": ([-hx - t / 2, 0, WALL_HEIGHT / 2], [t, 2 * hy, WALL_HEIGHT]),
    }
    for name, (pos, size) in walls.items():
        wall = Cube(f"/World/room/wall_{name}", sizes=1.0, scales=[size], positions=[pos], colors=[[0.92, 0.9, 0.85]])
        _collider(wall)
    for key, s in scene.surfaces.items():
        furniture = Cube(
            f"/World/room/{key}",
            sizes=1.0,
            scales=[[s.size[0], s.size[1], s.z]],
            positions=[[s.center[0], s.center[1], s.z / 2]],
            colors=[FURNITURE_COLORS.get(key, [0.4, 0.3, 0.2])],
        )
        _collider(furniture)
    if scene.marker is not None:  # 画像の向きを確かめる目印（床の左上 = -x, +y の隅）
        m = scene.marker
        Cube(
            "/World/room/orientation_marker",
            sizes=1.0,
            scales=[[m.size[0], m.size[1], 0.005]],
            positions=[[m.center[0], m.center[1], FLOOR_TOP + 0.0025]],
            colors=[[0.05, 0.05, 0.05]],
        )

    # ---- 物体（剛体）----
    material = RigidBodyMaterial(
        "/World/materials/object_material",
        static_frictions=OBJECT_FRICTION[0],
        dynamic_frictions=OBJECT_FRICTION[1],
        restitutions=0.0,
    )
    objects: dict[str, RigidPrim] = {}
    for o in scene.objects:
        p = placements[o.key]
        pose = {
            "positions": [list(scene.object_center(o.key, p))],
            "orientations": [list(quat_wxyz_from_yaw(p.yaw))],
        }
        if o.shape == "cube":
            shape = Cube(object_path(o.key), sizes=1.0, scales=[list(o.size)], colors=[list(o.color)], **pose)
        elif o.shape == "cylinder":
            shape = Cylinder(
                object_path(o.key), radii=o.size[0], heights=o.size[1], axes="Z", colors=[list(o.color)], **pose
            )
        else:
            raise ValueError(f"未対応の形です: {o.shape}")
        _collider(shape).apply_physics_materials(material)
        body = RigidPrim(shape.paths)
        body.set_masses([o.mass])
        objects[o.key] = body

    # ---- 天井カメラ（真下向き：回転なしでワールドの -Z を向き、画像の上がワールドの +Y）----
    cam = scene.camera
    rtx = RtxCamera("/World/ceiling_camera", positions=[list(cam.position)], orientations=[[1.0, 0.0, 0.0, 0.0]])
    # Camera の API の単位は USD の値の 1/10（focal_length と aperture で同じなので比は変わらない。#5）
    rtx.camera.set_focal_lengths([cam.focal_length / 10])
    rtx.camera.set_apertures(horizontal_apertures=[cam.horizontal_aperture / 10])
    rtx.camera.set_clipping_ranges(near_distances=[0.05], far_distances=[20.0])
    ceiling = CameraSensor(rtx, resolution=(cam.height, cam.width), annotators=["rgb", "distance_to_image_plane"])

    # ---- ロボット ----
    stage_utils.add_reference_to_stage(rb.asset_url(), rb.ROBOT_PATH)
    while stage_utils.is_stage_loading():
        simulation_app.update()
    rb.zero_base_joint_friction()

    # ---- 物理の開始 ----
    SimulationManager.setup_simulation(dt=PHYSICS_DT)
    app_utils.play()
    for _ in range(30):
        simulation_app.update()
    art = Articulation(rb.find_articulation_root())
    sim_scene = SimScene(scene=scene, objects=objects, robot=art, ceiling=ceiling, dof_names=list(art.dof_names))
    k, d, f = rb.FINGER_DRIVE
    finger_idx = sim_scene.dof_indices(rb.FINGER_JOINTS)
    art.set_dof_gains(stiffnesses=[k, k], dampings=[d, d], dof_indices=finger_idx)
    art.set_dof_max_efforts([f, f], dof_indices=finger_idx)
    set_robot_pose(simulation_app, sim_scene, base_pose, rb.STANDBY_ARM)
    return sim_scene


def set_robot_pose(simulation_app, sim: SimScene, base: Pose2D, arm: tuple[float, ...], steps: int = 30) -> None:
    """台車を base に、アームを arm の関節角に瞬間的に置き、指を開いて保持する（台車の関節は位置制御）。"""
    art = sim.robot
    base_idx = sim.dof_indices(rb.BASE_JOINTS)
    q = art.get_dof_positions().numpy()[0].copy()
    q[base_idx] = (base.x, base.y, base.yaw)
    q[sim.dof_indices(rb.ARM_JOINTS)] = arm
    q[sim.dof_indices(rb.FINGER_JOINTS)] = rb.FINGER_OPEN
    art.set_dof_positions(q)
    art.set_dof_velocities(np.zeros_like(q))
    art.set_dof_position_targets(q)
    hold_base(sim, (base.x, base.y, base.yaw))
    for _ in range(steps):
        simulation_app.update()


def hold_base(sim: SimScene, pose: tuple[float, float, float]) -> None:
    """台車の関節を位置制御にして pose (x, y, yaw) で止める。x・y は BASE_HOLD_DRIVE の硬いドライブで保持する。"""
    art = sim.robot
    idx = sim.dof_indices(rb.BASE_JOINTS)
    art.switch_dof_control_mode("position", dof_indices=idx)
    k, d, f = rb.BASE_HOLD_DRIVE
    xy = idx[:2]
    # 既定のゲイン（移動中の速度制御で使う）は変えない
    art.set_dof_gains(stiffnesses=[k, k], dampings=[d, d], dof_indices=xy, update_default_gains=False)
    art.set_dof_max_efforts([f, f], dof_indices=xy)
    art.set_dof_position_targets([list(pose)], dof_indices=idx)


def release_base(sim: SimScene) -> None:
    """台車の関節を速度制御にする（ゲイン・最大力はアセットのまま）。"""
    art = sim.robot
    idx = sim.dof_indices(rb.BASE_JOINTS)
    art.switch_dof_control_mode("velocity", dof_indices=idx)
    f = rb.BASE_DRIVE_MAX_EFFORT
    art.set_dof_max_efforts([f, f], dof_indices=idx[:2])
