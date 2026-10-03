"""Issue #3：RidgebackFranka のアームが届く範囲を、cuMotion の衝突を考えた IK で調べる（物理は進めない）。

台車を原点（向き 0）に置き、台車座標系の格子点（距離 × 高さ × 方向）それぞれについて、
手先（panda_hand）を下向きにした目標と、位置だけの目標を IK で解く。台車本体と地面を障害物に含める。

使い方::

    .venv\\Scripts\\python.exe scripts\\experiments\\arm_reach.py

出力: outputs/m0_3/reach/reach.json
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # experiments.* を import するため

from experiments.arm_common import reach_grid  # noqa: E402  （Isaac Sim 非依存）

DISTANCES = [round(0.3 + 0.1 * i, 2) for i in range(9)]  # 0.3〜1.1 m（台車の中心からの水平距離）
HEIGHTS = [round(0.2 + 0.1 * i, 2) for i in range(10)]  # 0.2〜1.1 m（手先の高さ）
ANGLES = [-90, -60, -30, 0, 30, 60, 90]  # 台車の前方からの方向 [deg]（左が正）


def main() -> int:
    out_dir = REPO_ROOT / "outputs" / "m0_3" / "reach"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    # ------------------------------------------------------------
    # SimulationApp を最初に起動（Isaac Sim のモジュールはこの後で import）
    # ------------------------------------------------------------
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    result: dict = {"distances": DISTANCES, "heights": HEIGHTS, "angles_deg": ANGLES, "points": []}
    exit_code = 1
    try:
        import cumotion
        import numpy as np
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.robot_motion.cumotion.impl.utils import isaac_sim_to_cumotion_pose

        from experiments import arm_sim
        from experiments import ridgeback_sim as rs

        # ---- シーン：地面＋ロボット（台車は原点・向き 0、アームは既定姿勢）----
        rs.build_scene(simulation_app, "multiphysics")
        arm_sim.zero_base_joint_friction()
        SimulationManager.setup_simulation(dt=1.0 / 60.0)
        rs.play_and_warmup(simulation_app)
        art = rs.articulation()
        arm_sim.set_robot_configuration(simulation_app, art, (0.0, 0.0, 0.0))
        planning = arm_sim.ArmPlanning(art)
        link0_pos, link0_q = planning.arm_base_pose()
        result["arm_base_world"] = {"position": link0_pos.tolist(), "orientation_wxyz": link0_q.tolist()}
        result["base_obstacle"] = {
            "center_in_base": planning.base_box_center_local.tolist(),
            "size": planning.base_box_size.tolist(),
        }
        result["tracked_prims"] = planning.tracked_prims

        # ---- 衝突を考えた IK ソルバー ----
        world_view = planning.world_interface.world_view
        world_view.update()
        robot_description = planning.cumotion_robot.robot_description
        ik_config = cumotion.create_default_collision_free_ik_solver_config(
            robot_description, arm_sim.TOOL_FRAME, world_view
        )
        solver = cumotion.create_collision_free_ik_solver(ik_config)
        success = cumotion.CollisionFreeIkSolver.Results.Status.SUCCESS
        hand_down = isaac_sim_to_cumotion_pose(
            position_world_to_target=np.zeros(3),
            orientation_world_to_target=np.asarray(arm_sim.HAND_DOWN_WXYZ),
            position_world_to_base=link0_pos,
            orientation_world_to_base=link0_q,
        ).rotation

        def solve(point_world, with_orientation: bool) -> tuple[bool, float]:
            target_base = isaac_sim_to_cumotion_pose(
                position_world_to_target=np.asarray(point_world),
                orientation_world_to_target=np.asarray(arm_sim.HAND_DOWN_WXYZ),
                position_world_to_base=link0_pos,
                orientation_world_to_base=link0_q,
            )
            orientation = (
                cumotion.CollisionFreeIkSolver.OrientationConstraint.target(hand_down)
                if with_orientation
                else cumotion.CollisionFreeIkSolver.OrientationConstraint.none()
            )
            target = cumotion.CollisionFreeIkSolver.TaskSpaceTarget(
                cumotion.CollisionFreeIkSolver.TranslationConstraint.target(target_base.translation), orientation
            )
            t0 = time.perf_counter()
            res = solver.solve(target)
            return res.status() == success, time.perf_counter() - t0

        # ---- 格子点ごとに解く（台車は原点・向き 0 なので、台車座標 = ワールド座標）----
        t_all = time.perf_counter()
        for pt in reach_grid(DISTANCES, HEIGHTS, ANGLES):
            ok_down, dt_down = solve(pt.point_in_base, with_orientation=True)
            ok_pos, dt_pos = solve(pt.point_in_base, with_orientation=False)
            result["points"].append(
                {
                    "distance": pt.distance,
                    "height": pt.height,
                    "angle_deg": pt.angle_deg,
                    "hand_down_ok": ok_down,
                    "position_only_ok": ok_pos,
                    "time_down_s": dt_down,
                    "time_pos_s": dt_pos,
                }
            )
        result["total_time_s"] = time.perf_counter() - t_all
        times = [p["time_down_s"] for p in result["points"]]
        result["ik_time_down_median_s"] = float(np.median(times))
        _print_maps(result)
        result["status"] = "ok"
        exit_code = 0
    except Exception as exc:
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        (out_dir / "reach.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[reach] status={result.get('status')} -> {out_dir}", flush=True)
    simulation_app.close(exit_code=exit_code)
    return exit_code


def _print_maps(result: dict) -> None:
    """方向ごとに、距離（列）× 高さ（行）の到達可否を表示する（D=下向きで可、P=位置だけ可、.=不可）。"""
    pts = {(p["angle_deg"], p["distance"], p["height"]): p for p in result["points"]}
    for a in result["angles_deg"]:
        print(
            f"[reach] angle={a:+d}deg  rows=height, cols=distance " + " ".join(f"{d:.1f}" for d in result["distances"])
        )
        for h in reversed(result["heights"]):
            cells = []
            for d in result["distances"]:
                p = pts[(a, d, h)]
                cells.append(" D " if p["hand_down_ok"] else (" P " if p["position_only_ok"] else " . "))
            print(f"[reach]   h={h:.1f} " + "".join(cells))


if __name__ == "__main__":
    sys.exit(main())
