"""Issue #2 台車の検証用：Isaac Sim に依存する共通処理（シーン構築・カメラ・姿勢の取得）。

※ このモジュールは SimulationApp を生成した後で import すること。
"""

from __future__ import annotations

from pathlib import Path

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.objects import DomeLight, GroundPlane
from isaacsim.core.experimental.prims import Articulation, XformPrim
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
from isaacsim.storage.native import get_assets_root_path
from PIL import Image
from pxr import Usd, UsdPhysics

from experiments.ridgeback_common import yaw_from_quat_wxyz

# 比較する2種類のアセット（Isaac Sim 6.1 のアセットサーバー上のパス）
ASSETS = {
    "multiphysics": "/Isaac/Robots_Multiphysics/Clearpath/RidgebackFranka/ridgeback_franka.usda",
    "legacy": "/Isaac/Robots/Clearpath/RidgebackFranka/ridgeback_franka.usd",
}
ROBOT_PATH = "/World/robot"
BASE_JOINTS = ("dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint", "dummy_base_revolute_z_joint")


def asset_url(asset_key: str) -> str:
    root = get_assets_root_path()
    if root is None:
        raise RuntimeError("Isaac Sim のアセットルートを解決できません")
    return root + ASSETS[asset_key]


def wait_for_loading(simulation_app) -> None:
    while stage_utils.is_stage_loading():
        simulation_app.update()


def build_scene(simulation_app, asset_key: str) -> None:
    """地面・ライト・ロボットを配置する。"""
    GroundPlane("/World/ground_plane")
    DomeLight("/World/DomeLight").set_intensities(1000)
    stage_utils.add_reference_to_stage(asset_url(asset_key), ROBOT_PATH)
    wait_for_loading(simulation_app)


def find_articulation_root(robot_path: str = ROBOT_PATH) -> str:
    """robot_path 以下で ArticulationRootAPI が付いた prim のパスを返す。"""
    stage = stage_utils.get_current_stage()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(prim.GetPath())
    raise RuntimeError(f"{robot_path} 以下に ArticulationRootAPI がありません")


def find_joint_prim(joint_name: str, robot_path: str = ROBOT_PATH) -> Usd.Prim:
    stage = stage_utils.get_current_stage()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path), Usd.TraverseInstanceProxies()):
        if prim.GetName() == joint_name and prim.IsA(UsdPhysics.Joint):
            return prim
    raise RuntimeError(f"関節 {joint_name} が見つかりません")


def base_link_path() -> str:
    """台車本体のリンク（yaw 関節の子リンク）のパス。"""
    joint = UsdPhysics.Joint(find_joint_prim(BASE_JOINTS[2]))
    targets = joint.GetBody1Rel().GetTargets()
    if not targets:
        raise RuntimeError("yaw 関節の body1 がありません")
    return str(targets[0])


def link_path(link_name: str, robot_path: str = ROBOT_PATH) -> str:
    stage = stage_utils.get_current_stage()
    for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_path), Usd.TraverseInstanceProxies()):
        if prim.GetName() == link_name and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(prim.GetPath())
    raise RuntimeError(f"リンク {link_name} が見つかりません")


def world_pose(prim: XformPrim) -> tuple[np.ndarray, np.ndarray]:
    """(位置 (3,), クォータニオン wxyz (4,)) を返す。"""
    positions, orientations = prim.get_world_poses()
    return positions.numpy()[0].astype(float), orientations.numpy()[0].astype(float)


def pose2d(prim: XformPrim) -> tuple[float, float, float]:
    """ワールド上の (x, y, yaw)。"""
    pos, quat_wxyz = world_pose(prim)
    return float(pos[0]), float(pos[1]), yaw_from_quat_wxyz(quat_wxyz)


def articulation(path: str | None = None) -> Articulation:
    return Articulation(path or find_articulation_root())


# ------------------------------------------------------------
# カメラ（斜め上から見下ろす1台。画像の向きは保存して目視確認する）
# ------------------------------------------------------------
class Camera:
    def __init__(self, path: str, eye, target, resolution=(480, 640)) -> None:
        self._sensor = CameraSensor(RtxCamera(path), resolution=resolution, annotators=["rgb"])
        self._path = path
        self.look_at(eye, target)

    def look_at(self, eye, target) -> None:
        ViewportManager.set_camera_view(self._path, eye=list(eye), target=list(target))

    def save(self, simulation_app, out_path: Path, max_wait_steps: int = 60) -> Path:
        """次に届いた RGB 画像を PNG で保存する（数フレーム進めて最新の画像を使う）。"""
        for _ in range(3):
            simulation_app.update()
        for _ in range(max_wait_steps):
            rgb, _ = self._sensor.get_data("rgb")
            if rgb is not None:
                break
            simulation_app.update()
        else:
            raise RuntimeError("カメラ画像を取得できませんでした")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.ascontiguousarray(rgb.numpy()[:, :, :3], dtype=np.uint8)).save(out_path)
        return out_path


def play_and_warmup(simulation_app, steps: int = 30) -> None:
    app_utils.play()
    for _ in range(steps):
        simulation_app.update()
