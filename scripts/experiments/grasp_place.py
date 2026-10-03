"""Issue #4：台車を止めたまま、机の上の物体を掴んで別の位置へ置く。成功率・配置の誤差・失敗の傾向を調べる。

手順は 6.1 の Pick and Place サンプルの PickPlaceController（isaacsim.robot_motion.examples）を使う:
    真上へ移動 → 下降 → 閉じる → 持ち上げる → 目標の真上へ移動 → 下降 → 開く → 退避
アームは cuMotion の RmpFlowController（#3 の推奨）。RidgebackFranka 用の処理は #3 の ArmPlanning を使う。

使い方::

    .venv\\Scripts\\python.exe scripts\\experiments\\grasp_place.py                       # 3 種類 × 10 回
    .venv\\Scripts\\python.exe scripts\\experiments\\grasp_place.py --objects box --trials 5 --variant low_friction
    .venv\\Scripts\\python.exe scripts\\experiments\\grasp_place.py --objects box --trials 5 --variant heavy

出力:
    outputs/m0_4/<variant>_<finger-drive>/summary.json
    outputs/m0_4/frames/<variant>_<finger-drive>_<物体>_t<番号>/ の連番画像
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # experiments.* を import するため

from experiments.arm_common import quat_from_yaw_wxyz, quat_mul_wxyz, world_from_base  # noqa: E402
from experiments.grasp_common import (  # noqa: E402
    TrialObservation,
    classify_trial,
    sample_initial_offsets,
    tilt_deg_from_quat_wxyz,
)

# ------------------------------------------------------------
# 条件
# ------------------------------------------------------------
BASE_A = (1.0, -0.5, math.radians(30))  # 台車の姿勢（#3 と同じ）
TABLE_CENTER_IN_A = (0.85, 0.0, 0.35)  # 机：台車の前面から 8cm、天板の高さ 0.7m（#3 と同じ）
TABLE_SIZE = (0.6, 1.2, 0.7)
TABLE_TOP = TABLE_SIZE[2]
PICK_IN_A = (0.72, 0.12)  # 物体の基準位置（台車 A の座標系）。#3 の「楽に届く範囲」の中
PLACE_IN_A = (0.72, -0.15)  # 置き場所
HOME_HAND_IN_BASE = (0.4, 0.0, 0.95)  # 待機姿勢（#3）
PARK_IN_A = (-2.0, 2.0)  # 使わない物体を置いておく場所（床の上、ロボットから離れた所）

# 物体（幅はすべて 8cm 未満）。box は 4cm の面を掴む（手先の y 方向＝指が閉じる方向に 4cm の辺を合わせる）
OBJECTS = {
    "cube": {"shape": "cube", "size": (0.05, 0.05, 0.05)},
    "cylinder": {"shape": "cylinder", "radius": 0.025, "height": 0.08},
    "box": {"shape": "cube", "size": (0.06, 0.04, 0.10)},
}
VARIANTS = {  # 物体の摩擦（静止, 動）と質量 [kg]
    "base": {"friction": (0.8, 0.6), "mass": 0.1},
    "low_friction": {"friction": (0.2, 0.15), "mass": 0.1},
    "heavy": {"friction": (0.8, 0.6), "mass": 0.5},
}
# 指のドライブ（剛性 [N/m], 減衰 [N·s/m], 最大力 [N]）。None はアセットのまま（1e7, 1e5, 1200N）
FINGER_DRIVES = {
    "asset": None,
    "asset_70N": (1.0e7, 1.0e5, 70.0),  # 最大力だけ実機の Franka（連続 70N 程度）に合わせる
    "franka_sample": (400.0, 80.0, 7.2),  # 6.1 の Franka 単体アセット（Pick and Place サンプルで使用）と同じ値
    # 重い物体用：剛性を上げて把持力を増やす（指が全閉から 2cm 手前で止まるとき 2000 × 0.02 = 40N）
    "franka_strong": (2000.0, 100.0, 40.0),
}
HAND_DOWN_WXYZ = (0.0, 1.0, 0.0, 0.0)
FRANKA_GRASP_OFFSET = (0.0, 0.0, 0.1034)  # panda_hand → 指先の間（サンプルの robots.py と同じ）
# 真上へ近づく高さ・持ち上げ／退避の高さ [m]。サンプルの既定（0.20 / 0.30）では、アームの根元が低い
# RidgebackFranka は背の高い物体の退避位置（手先の高さ約 1.16m）に届かず時間切れになった
APPROACH_HEIGHT = 0.15
LIFT_HEIGHT = 0.15
TRIAL_TIMEOUT_S = 40.0
FRAME_EVERY_STEPS = 15  # 連番画像：4Hz
SETTLE_AFTER_S = 1.0


def object_height(spec: dict) -> float:
    return spec["size"][2] if spec["shape"] == "cube" else spec["height"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", nargs="+", choices=list(OBJECTS), default=list(OBJECTS))
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--variant", choices=list(VARIANTS), default="base")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--finger-drive",
        choices=list(FINGER_DRIVES),
        default="asset",
        help="指のドライブ（剛性・減衰・最大力）。asset は RidgebackFranka のまま",
    )
    parser.add_argument("--frames-trials", type=int, nargs="*", default=[0], help="連番画像を保存する試行の番号")
    args = parser.parse_args()
    run_name = f"{args.variant}_{args.finger_drive}"
    out_dir = REPO_ROOT / "outputs" / "m0_4" / run_name
    frames_root = REPO_ROOT / "outputs" / "m0_4" / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    # ------------------------------------------------------------
    # SimulationApp を最初に起動（Isaac Sim のモジュールはこの後で import）
    # ------------------------------------------------------------
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    variant = VARIANTS[args.variant]
    summary: dict = {
        "variant": args.variant,
        "variant_params": variant,
        "seed": args.seed,
        "finger_drive": args.finger_drive,
        "approach_height": APPROACH_HEIGHT,
        "lift_height": LIFT_HEIGHT,
        "trials": [],
    }
    exit_code = 1
    try:
        import isaacsim.core.experimental.utils.app as app_utils
        import numpy as np
        from isaacsim.core.experimental.materials import RigidBodyMaterial
        from isaacsim.core.experimental.objects import Cube, Cylinder
        from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.robot_motion.cumotion import RmpFlowController

        app_utils.enable_extension("isaacsim.robot_motion.examples")
        from isaacsim.robot_motion.examples.manipulation.controllers import (
            GripperCommand,
            JointGripperController,
            PickPlaceController,
            PickPlacePhase,
        )

        from experiments import arm_sim
        from experiments import ridgeback_sim as rs

        # ---- シーン：地面・ロボット・机・物体・カメラ ----
        rs.build_scene(simulation_app, "multiphysics")
        arm_sim.zero_base_joint_friction()
        table_center = world_from_base(BASE_A, TABLE_CENTER_IN_A)
        table = Cube(
            "/World/table",
            sizes=1.0,
            scales=[list(TABLE_SIZE)],
            positions=[list(table_center)],
            orientations=[list(quat_from_yaw_wxyz(BASE_A[2]))],
            colors="brown",
        )
        GeomPrim(table.paths, apply_collision_apis=True)
        material = RigidBodyMaterial(
            "/World/materials/object_material",
            static_frictions=variant["friction"][0],
            dynamic_frictions=variant["friction"][1],
            restitutions=0.0,
        )
        bodies: dict[str, RigidPrim] = {}
        for i, (name, spec) in enumerate(OBJECTS.items()):
            path = f"/World/obj_{name}"
            park = world_from_base(BASE_A, (PARK_IN_A[0], PARK_IN_A[1] + 0.3 * i, object_height(spec) / 2))
            if spec["shape"] == "cube":
                shape = Cube(path, sizes=1.0, scales=[list(spec["size"])], positions=[list(park)], colors="blue")
            else:
                shape = Cylinder(
                    path, radii=spec["radius"], heights=spec["height"], axes="Z", positions=[list(park)], colors="blue"
                )
            geom = GeomPrim(shape.paths, apply_collision_apis=True)
            geom.apply_physics_materials(material)
            body = RigidPrim(shape.paths)
            body.set_masses([variant["mass"]])
            bodies[name] = body
        pick_nominal = world_from_base(BASE_A, (*PICK_IN_A, 0.0))
        place_xy = world_from_base(BASE_A, (*PLACE_IN_A, 0.0))[:2]
        eye = world_from_base(BASE_A, (-0.4, -1.6, 1.8))
        camera = rs.Camera("/World/grasp_camera", eye=eye, target=[pick_nominal[0], pick_nominal[1], TABLE_TOP])

        # ---- 物理開始、台車を A に置いて止める ----
        SimulationManager.setup_simulation(dt=1.0 / 60.0)
        rs.play_and_warmup(simulation_app)
        dt = SimulationManager.get_physics_dt()
        art = rs.articulation()
        names = list(art.dof_names)
        arm_sim.set_robot_configuration(simulation_app, art, BASE_A)
        planning = arm_sim.ArmPlanning(art)
        controlled = planning.cumotion_robot.controlled_joint_names
        controlled_idx = [names.index(n) for n in controlled]
        home_q = planning.solve_ik(
            world_from_base(BASE_A, HOME_HAND_IN_BASE),
            HAND_DOWN_WXYZ,
            seed=[arm_sim.FRANKA_DEFAULT[n] for n in controlled],
        )
        if home_q is None:
            raise RuntimeError("待機姿勢の IK が解けませんでした")
        finger_names = ("panda_finger_joint1", "panda_finger_joint2")
        finger_idx = [names.index(n) for n in finger_names]
        drive = FINGER_DRIVES[args.finger_drive]
        if drive is not None:
            k, d, f = drive
            art.set_dof_gains(stiffnesses=[k, k], dampings=[d, d], dof_indices=finger_idx)
            art.set_dof_max_efforts([f, f], dof_indices=finger_idx)
        stiff, damp = (a.numpy()[0] for a in art.get_dof_gains(dof_indices=finger_idx))
        summary["finger_drive_actual"] = {
            "stiffness": [float(v) for v in stiff],
            "damping": [float(v) for v in damp],
            "max_effort": [float(v) for v in art.get_dof_max_efforts(dof_indices=finger_idx).numpy()[0]],
        }
        summary["tracked_prims"] = planning.tracked_prims
        summary["finger_physics_materials"] = _finger_materials(rs)
        # 指を閉じたり開いたりしている間（掴む物体に近づく〜離す）は、その物体を障害物から外す（サンプルと同じ）
        no_obstacle_phases = {
            PickPlacePhase.DESCEND_PICK,
            PickPlacePhase.GRASP,
            PickPlacePhase.LIFT,
            PickPlacePhase.APPROACH_PLACE,
            PickPlacePhase.DESCEND_PLACE,
            PickPlacePhase.RELEASE,
        }

        def reset_arm() -> None:
            q = art.get_dof_positions().numpy()[0].copy()
            q[controlled_idx] = home_q
            q[finger_idx] = 0.04
            art.set_dof_positions(q)
            art.set_dof_velocities(np.zeros_like(q))
            art.set_dof_position_targets(q)
            for _ in range(10):
                simulation_app.update()

        def make_controller() -> PickPlaceController:
            common = {
                "robot_joint_space": planning.joint_space,
                "joint_names": finger_names,
                "open_positions": (0.04, 0.04),
                "closed_positions": (0.0, 0.0),
                "velocity_tolerance": 0.1,  # サンプルの Franka 設定と同じ
            }
            arm = RmpFlowController(
                cumotion_robot=planning.cumotion_robot,
                cumotion_world_interface=planning.world_interface,
                robot_joint_space=planning.joint_space,
                robot_site_space=[arm_sim.TOOL_FRAME],
                tool_frame=arm_sim.TOOL_FRAME,
            )
            return PickPlaceController(
                arm_controller=arm,
                gripper_open_controller=JointGripperController(**common, command=GripperCommand.OPEN),
                gripper_close_controller=JointGripperController(**common, command=GripperCommand.CLOSE),
                robot_site_space=[arm_sim.TOOL_FRAME],
                tool_frame=arm_sim.TOOL_FRAME,
                controller_to_grasp_position=FRANKA_GRASP_OFFSET,
                grasp_orientation=HAND_DOWN_WXYZ,
                approach_height=APPROACH_HEIGHT,
                lift_height=LIFT_HEIGHT,
            )

        def goals(pick_pos, pick_yaw, place_pos) -> object:
            import isaacsim.robot_motion.experimental.motion_generation as mg
            import warp as wp

            q_pick = quat_mul_wxyz(quat_from_yaw_wxyz(pick_yaw), HAND_DOWN_WXYZ)
            q_place = quat_mul_wxyz(quat_from_yaw_wxyz(BASE_A[2]), HAND_DOWN_WXYZ)
            sites = ["pick", "place"]
            return mg.RobotState(
                sites=mg.SpatialState.from_name(
                    spatial_space=sites,
                    positions=(sites, wp.array(np.asarray([pick_pos, place_pos], dtype=np.float32), device="cpu")),
                    orientations=(sites, wp.array(np.asarray([q_pick, q_place], dtype=np.float32), device="cpu")),
                )
            )

        def run_trial(obj_name: str, trial: int, offset: tuple[float, float, float]) -> dict:
            spec = OBJECTS[obj_name]
            body = bodies[obj_name]
            h = object_height(spec)
            reset_arm()
            # 物体を初期位置に置く（台車 A の向きを基準に、ずらしと回転を加える）
            start_xy = world_from_base(BASE_A, (PICK_IN_A[0] + offset[0], PICK_IN_A[1] + offset[1], 0.0))[:2]
            obj_yaw = BASE_A[2] + offset[2]
            body.set_world_poses(
                positions=[[start_xy[0], start_xy[1], TABLE_TOP + h / 2 + 0.001]],
                orientations=[list(quat_from_yaw_wxyz(obj_yaw))],
            )
            body.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
            for _ in range(30):
                simulation_app.update()
            p0 = body.get_world_poses()[0].numpy()[0].astype(float)
            # 円柱は向きに関係なく掴めるので、手先の向きは台車に合わせる
            grasp_yaw = obj_yaw if spec["shape"] == "cube" else BASE_A[2]
            place_pos = (place_xy[0], place_xy[1], TABLE_TOP + h / 2 + 0.005)
            planning.sync()
            ctrl = make_controller()
            ctrl.reset(planning.robot_state(), goals(p0, grasp_yaw, place_pos), 0.0)

            frames_dir = frames_root / f"{run_name}_{obj_name}_t{trial}" if trial in args.frames_trials else None
            obstacle_off = False
            moved_before_grasp, max_lift, lifted, dropped = None, 0.0, False, False
            phases_seen, t, step = [], 0.0, 0
            while t < TRIAL_TIMEOUT_S and not ctrl.is_done and not ctrl.failed:
                want_off = ctrl.phase in no_obstacle_phases
                if want_off != obstacle_off:
                    planning.set_obstacles_enabled(body.paths[0], not want_off)
                    obstacle_off = want_off
                planning.sync()
                desired = ctrl.forward(planning.robot_state(), None, t)
                planning.apply(desired)
                simulation_app.update()
                step += 1
                t = step * dt
                if not phases_seen or phases_seen[-1][0] != ctrl.phase.name:
                    fingers = art.get_dof_positions().numpy()[0][finger_idx]
                    phases_seen.append((ctrl.phase.name, round(t, 3), [round(float(f), 4) for f in fingers]))
                p = body.get_world_poses()[0].numpy()[0].astype(float)
                if moved_before_grasp is None and ctrl.phase is PickPlacePhase.GRASP:
                    moved_before_grasp = float(np.linalg.norm(p[:2] - p0[:2]))
                lift = p[2] - p0[2]
                max_lift = max(max_lift, lift)
                lifted = lifted or lift > 0.03
                before_place = ctrl.phase in {PickPlacePhase.LIFT, PickPlacePhase.APPROACH_PLACE}
                if lifted and before_place and lift < 0.01:
                    dropped = True
                if frames_dir is not None and step % FRAME_EVERY_STEPS == 0:
                    camera.capture(frames_dir / f"{step:05d}.png")
            if obstacle_off:
                planning.set_obstacles_enabled(body.paths[0], True)
            for _ in range(int(SETTLE_AFTER_S / dt)):  # 置いた後に落ち着くのを待つ
                simulation_app.update()
            p1, q1 = (a.numpy()[0].astype(float) for a in body.get_world_poses())
            o = TrialObservation(
                controller_done=ctrl.is_done,
                moved_before_grasp_m=moved_before_grasp if moved_before_grasp is not None else 0.0,
                max_lift_m=max_lift,
                dropped=dropped,
                place_xy_err_m=float(math.hypot(p1[0] - place_pos[0], p1[1] - place_pos[1])),
                tilt_deg=tilt_deg_from_quat_wxyz(q1),
            )
            result = {
                "object": obj_name,
                "trial": trial,
                "offset": offset,
                "outcome": classify_trial(o),
                "observation": o.__dict__,
                "place_z_err_m": float(p1[2] - (TABLE_TOP + h / 2)),
                "duration_s": t,
                "final_phase": ctrl.phase.name,
                "failure_reason": ctrl.failure_reason,
                "phases": phases_seen,
            }
            print(
                f"[grasp] {run_name:26s} {obj_name:8s} t{trial} {result['outcome']:22s} "
                f"xy_err={o.place_xy_err_m * 1000:.1f}mm tilt={o.tilt_deg:.1f}deg time={t:.1f}s "
                f"phase={ctrl.phase.name} {ctrl.failure_reason or ''}",
                flush=True,
            )
            return result

        offsets = sample_initial_offsets(args.trials, seed=args.seed)
        for obj_name in args.objects:
            for trial, offset in enumerate(offsets):
                summary["trials"].append(run_trial(obj_name, trial, offset))
            # 次の物体の邪魔にならないよう、待機場所へ戻す
            i = list(OBJECTS).index(obj_name)
            park = world_from_base(BASE_A, (PARK_IN_A[0], PARK_IN_A[1] + 0.3 * i, object_height(OBJECTS[obj_name]) / 2))
            bodies[obj_name].set_world_poses(positions=[list(park)], orientations=[[1.0, 0.0, 0.0, 0.0]])
            bodies[obj_name].set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
        summary["stats"] = _stats(summary["trials"])
        summary["status"] = "ok"
        exit_code = 0
    except Exception as exc:
        traceback.print_exc()
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
        )
        print(f"[grasp] status={summary.get('status')} -> {out_dir}", flush=True)
    simulation_app.close(exit_code=exit_code)
    return exit_code


def _finger_materials(rs) -> dict:
    """指の衝突形状に付いている物理マテリアル（摩擦）を調べる。"""
    from pxr import Usd, UsdPhysics, UsdShade

    stage = rs.stage_utils.get_current_stage()
    out = {}
    for link in ("panda_leftfinger", "panda_rightfinger"):
        root = stage.GetPrimAtPath(rs.link_path(link))
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
            info = {"material": str(mat.GetPath()) if mat else None}
            if mat:
                api = UsdPhysics.MaterialAPI(mat.GetPrim())
                info.update(
                    static=api.GetStaticFrictionAttr().Get(),
                    dynamic=api.GetDynamicFrictionAttr().Get(),
                )
            out[str(prim.GetPath())] = info
    return out


def _stats(trials: list[dict]) -> dict:
    import statistics
    from collections import Counter

    out = {}
    for obj in sorted({t["object"] for t in trials}):
        ts = [t for t in trials if t["object"] == obj]
        ok = [t for t in ts if t["outcome"] == "success"]
        errs = [t["observation"]["place_xy_err_m"] * 1000 for t in ok]
        out[obj] = {
            "n": len(ts),
            "success": len(ok),
            "outcomes": dict(Counter(t["outcome"] for t in ts)),
            "place_xy_err_mean_mm": statistics.mean(errs) if errs else None,
            "place_xy_err_max_mm": max(errs) if errs else None,
            "tilt_max_deg": max(t["observation"]["tilt_deg"] for t in ok) if ok else None,
            "duration_mean_s": statistics.mean(t["duration_s"] for t in ok) if ok else None,
        }
    return out


if __name__ == "__main__":
    sys.exit(main())
