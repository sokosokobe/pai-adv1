"""Issue #2：RidgebackFranka を読み込み、関節・リンクの一覧と台車の関節の向きを調べる。

使い方::

    .venv\\Scripts\\python.exe scripts\\experiments\\ridgeback_inspect.py --asset multiphysics
    .venv\\Scripts\\python.exe scripts\\experiments\\ridgeback_inspect.py --asset legacy

出力: outputs/m0_2/inspect_<asset>/ に inspect.json と overview.png
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=["multiphysics", "legacy"], default="multiphysics")
    args = parser.parse_args()
    out_dir = REPO_ROOT / "outputs" / "m0_2" / f"inspect_{args.asset}"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    # ------------------------------------------------------------
    # SimulationApp を最初に起動（Isaac Sim のモジュールはこの後で import）
    # ------------------------------------------------------------
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    exit_code = 1
    result: dict = {"asset": args.asset}
    try:
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.prims import XformPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Usd, UsdPhysics

        from experiments import ridgeback_sim as rs

        # ---- シーン構築 ----
        rs.build_scene(simulation_app, args.asset)
        result["asset_url"] = rs.asset_url(args.asset)
        root_path = rs.find_articulation_root()
        result["articulation_root"] = root_path
        camera = rs.Camera("/World/inspect_camera", eye=[3.0, 3.0, 2.2], target=[0.0, 0.0, 0.5])

        # ---- USD 上の関節（固定関節を含む）とリンクの一覧 ----
        stage = stage_utils.get_current_stage()
        joints, links = [], []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(rs.ROBOT_PATH), Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                axis_attr = prim.GetAttribute("physics:axis")
                joints.append(
                    {
                        "name": prim.GetName(),
                        "type": prim.GetTypeName(),
                        "axis_local": axis_attr.Get() if axis_attr else None,
                        "body0": [str(t) for t in joint.GetBody0Rel().GetTargets()],
                        "body1": [str(t) for t in joint.GetBody1Rel().GetTargets()],
                    }
                )
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                has_collision = any(
                    p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
                )
                mass_attr = prim.GetAttribute("physics:mass")
                links.append(
                    {
                        "name": prim.GetName(),
                        "path": str(prim.GetPath()),
                        "has_collision": has_collision,
                        "mass_authored_kg": mass_attr.Get() if mass_attr and mass_attr.HasAuthoredValue() else None,
                    }
                )
        result["usd_joints"] = joints
        result["links"] = links

        # ---- 物理を開始し、Articulation から DOF の情報を取得（SI 単位：m, rad）----
        SimulationManager.setup_simulation(dt=1.0 / 60.0)
        rs.play_and_warmup(simulation_app)
        art = rs.articulation(root_path)
        names = list(art.dof_names)
        lower, upper = (a.numpy()[0] for a in art.get_dof_limits())
        stiff, damp = (a.numpy()[0] for a in art.get_dof_gains())
        max_eff = art.get_dof_max_efforts().numpy()[0]
        max_vel = art.get_dof_max_velocities().numpy()[0]
        try:
            drive_types = art.get_dof_drive_types()
            drive_types = list(drive_types[0]) if drive_types is not None else [None] * len(names)
        except Exception as exc:  # backend によっては未対応
            drive_types = [f"unavailable: {exc}"] * len(names)
        result["num_dofs"] = art.num_dofs
        result["dofs"] = [
            {
                "index": i,
                "name": n,
                "type": str(art.dof_types[i]),
                "lower": float(lower[i]),
                "upper": float(upper[i]),
                "stiffness": float(stiff[i]),
                "damping": float(damp[i]),
                "max_effort": float(max_eff[i]),
                "max_velocity": float(max_vel[i]),
                "drive_type": str(drive_types[i]),
            }
            for i, n in enumerate(names)
        ]
        camera.save(simulation_app, out_dir / "overview.png")

        # ---- 台車の各仮想関節を1つずつ動かし、台車リンクのワールド姿勢の変化を見る ----
        base = XformPrim(rs.base_link_path())
        result["base_link"] = base.paths[0]
        idx = {n: names.index(n) for n in rs.BASE_JOINTS}
        home = art.get_dof_positions().numpy()[0].copy()
        p0 = rs.pose2d(base)
        result["base_pose_at_home"] = p0
        probes = {}
        for joint_name, delta in (
            (rs.BASE_JOINTS[0], 0.5),
            (rs.BASE_JOINTS[1], 0.5),
            (rs.BASE_JOINTS[2], math.pi / 2),
        ):
            q = home.copy()
            q[idx[joint_name]] += delta
            art.set_dof_positions(q)
            art.set_dof_position_targets(q)
            for _ in range(5):
                simulation_app.update()
            p = rs.pose2d(base)
            probes[joint_name] = {
                "delta": delta,
                "base_dx": p[0] - p0[0],
                "base_dy": p[1] - p0[1],
                "base_dyaw": p[2] - p0[2],
            }
            art.set_dof_positions(home)
            art.set_dof_position_targets(home)
            for _ in range(5):
                simulation_app.update()
        # 組み合わせ：yaw を 90° にした状態で x 関節を動かしたとき、ワールドの x に動くか（関節がワールド基準か）
        q = home.copy()
        q[idx[rs.BASE_JOINTS[2]]] += math.pi / 2
        q[idx[rs.BASE_JOINTS[0]]] += 0.5
        art.set_dof_positions(q)
        art.set_dof_position_targets(q)
        for _ in range(5):
            simulation_app.update()
        p = rs.pose2d(base)
        probes["yaw90_then_x"] = {"base_dx": p[0] - p0[0], "base_dy": p[1] - p0[1], "base_dyaw": p[2] - p0[2]}
        result["base_joint_probes"] = probes

        result["status"] = "ok"
        exit_code = 0
        print(json.dumps({k: result[k] for k in ("num_dofs", "base_joint_probes")}, indent=2), flush=True)
    except Exception as exc:
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        (out_dir / "inspect.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        print(f"[inspect] status={result.get('status')} -> {out_dir}", flush=True)
    simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
