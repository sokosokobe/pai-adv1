"""RidgebackFranka の定数と、USD の上でロボットのリンク・関節を探す処理（設計書 5.3）。

※ SimulationApp を生成した後で import すること。#2・#3 の scripts/experiments/ridgeback_sim.py・arm_sim.py から移した。
"""

from __future__ import annotations

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.storage.native import get_assets_root_path
from pxr import Usd, UsdPhysics

ROBOT_ASSET = "/Isaac/Robots_Multiphysics/Clearpath/RidgebackFranka/ridgeback_franka.usda"  # Multiphysics 版（#2）
ROBOT_PATH = "/World/robot"
# 台車の x・y・yaw の仮想関節（ワールド基準。関節の値 = 台車のワールド姿勢。#2）
BASE_JOINTS = ("dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint", "dummy_base_revolute_z_joint")
ARM_JOINTS = tuple(f"panda_joint{i}" for i in range(1, 8))
FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")
FINGER_OPEN = 0.04
FINGER_CLOSED = 0.0
# 指のドライブ（剛性 [N/m]・減衰 [N·s/m]・最大力 [N]）。アセット（1e7・1e5・1200N）では掴めない（#4）
FINGER_DRIVE = (2000.0, 100.0, 40.0)
# 待機姿勢：台車座標で手先 (0.4, 0, 0.95)、真下向き（#3 の IK の解）。移動中・運搬中もこの姿勢
STANDBY_ARM = (-1.261, -1.378, 0.182, -2.254, 0.229, 0.892, 0.228)
# 畳んだ姿勢：Franka の既定姿勢（cuMotion・Pick and Place サンプルと同じ）。写真を撮るとき（#10）
FOLDED_ARM = (0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741)
TOOL_FRAME = "panda_hand"


def asset_url() -> str:
    root = get_assets_root_path()
    if root is None:
        raise RuntimeError("Isaac Sim のアセットルートを解決できません")
    return root + ROBOT_ASSET


def _prims(robot_path: str = ROBOT_PATH):
    stage = stage_utils.get_current_stage()
    return Usd.PrimRange(stage.GetPrimAtPath(robot_path), Usd.TraverseInstanceProxies())


def find_articulation_root(robot_path: str = ROBOT_PATH) -> str:
    for prim in _prims(robot_path):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return str(prim.GetPath())
    raise RuntimeError(f"{robot_path} 以下に ArticulationRootAPI がありません")


def find_joint_prim(joint_name: str, robot_path: str = ROBOT_PATH) -> Usd.Prim:
    for prim in _prims(robot_path):
        if prim.GetName() == joint_name and prim.IsA(UsdPhysics.Joint):
            return prim
    raise RuntimeError(f"関節 {joint_name} が見つかりません")


def link_path(link_name: str, robot_path: str = ROBOT_PATH) -> str:
    for prim in _prims(robot_path):
        if prim.GetName() == link_name and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(prim.GetPath())
    raise RuntimeError(f"リンク {link_name} が見つかりません")


def base_link_path() -> str:
    """台車本体のリンク（yaw 関節の子リンク）のパス。"""
    joint = UsdPhysics.Joint(find_joint_prim(BASE_JOINTS[2]))
    targets = joint.GetBody1Rel().GetTargets()
    if not targets:
        raise RuntimeError("yaw 関節の body1 がありません")
    return str(targets[0])


def zero_base_joint_friction() -> None:
    """台車の x・y 関節の physxJoint:jointFriction を 0 にする（物理の開始前に呼ぶ）。

    アセットのままでは速度指令が 8〜20cm 手前で止まる（#2）。
    """
    for joint_name in BASE_JOINTS[:2]:
        find_joint_prim(joint_name).GetAttribute("physxJoint:jointFriction").Set(0.0)


def world_pose(prim: XformPrim) -> tuple[np.ndarray, np.ndarray]:
    """(位置 (3,), クォータニオン wxyz (4,))。"""
    positions, orientations_wxyz = prim.get_world_poses()
    return positions.numpy()[0].astype(float), orientations_wxyz.numpy()[0].astype(float)
