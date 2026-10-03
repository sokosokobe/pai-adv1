"""Issue #3：RidgebackFranka のアームを cuMotion で机の上の目標へ動かし、手段ごとの精度・時間を比べる。

手段（どちらも Motion Generation（Experimental）の枠組み上の cuMotion）:
    rmpflow : RmpFlowController（反応型。毎ステップ指令を計算）
    trajopt : TrajectoryOptimizer（軌道を一括で計画し、位置指令で追従）
    graph   : GraphBasedMotionPlanner（経路をグラフ探索で計画 → 最短時間の軌道に変換し、位置指令で追従）

段階:
    A_synced     : 台車を原点以外の姿勢 A に置き、根元の姿勢を cuMotion に渡して試行
    B_not_synced : 台車を速度指令で姿勢 B へ動かし、根元の姿勢を「更新せずに」試行（更新の必要性の確認）
    B_synced     : 根元の姿勢を更新して試行

使い方::

    .venv\\Scripts\\python.exe scripts\\experiments\\arm_motion.py

出力: outputs/m0_3/motion/ に summary.json、各試行の画像
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # experiments.* を import するため

from experiments.arm_common import is_success, quat_angle_error_wxyz, world_from_base  # noqa: E402
from experiments.ridgeback_common import settle_time, velocity_command  # noqa: E402

# ------------------------------------------------------------
# 条件
# ------------------------------------------------------------
BASE_A = (1.0, -0.5, math.radians(30))  # 台車の姿勢 A (x, y, yaw)
# 机（固定）：台車 A の前面（中心から 0.466m）から 8cm 空けて置く。天板の高さ 0.7m
TABLE_CENTER_IN_A = (0.85, 0.0, 0.35)
TABLE_SIZE = (0.6, 1.2, 0.7)  # (奥行き, 幅, 高さ) [m]
HAND_HEIGHT = 0.85  # 目標の手先（panda_hand）の高さ。指先は天板の約 5cm 上
# 目標（台車 A から見た距離 [m]、方向 [deg]）。到達範囲の調査（arm_reach.py）で楽に届く範囲から選んだ
TARGETS_IN_A = [(0.65, 0.0), (0.80, 0.0), (0.70, 25.0), (0.70, -25.0), (0.75, 35.0)]
# 待機姿勢：台車の上空に手先を下向きで置く（台車座標系）。Franka 単体の既定姿勢では手先が机（高さ 0.7m）に当たるため
HOME_HAND_IN_BASE = (0.4, 0.0, 0.95)
# 台車の姿勢 B：A から机に沿って左へ 0.3m、向きを -10°
BASE_B_OFFSET_IN_A = (0.0, 0.30)
BASE_B_DYAW = math.radians(-10)

POS_TOL, ANG_TOL = 0.01, math.radians(5)  # 成功の判定
RMP_DURATION_S = 5.0
TRAJ_EXTRA_S = 1.0  # 軌道の終了後に待つ時間
METHODS = ("rmpflow", "trajopt", "graph")
# グラフ計画の経路を軌道に変換するときの上限（cuMotion の graph_planner の例と同じ値）
GRAPH_MAX_VEL = [2.0, 2.0, 2.0, 2.0, 2.5, 2.5, 2.5]
GRAPH_MAX_ACC = [2.0] * 7


def main() -> int:
    out_dir = REPO_ROOT / "outputs" / "m0_3" / "motion"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    # ------------------------------------------------------------
    # SimulationApp を最初に起動（Isaac Sim のモジュールはこの後で import）
    # ------------------------------------------------------------
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    summary: dict = {"conditions": {"base_A": BASE_A, "targets_in_A": TARGETS_IN_A, "hand_height": HAND_HEIGHT}}
    summary["runs"] = []
    exit_code = 1
    try:
        import cumotion
        import numpy as np
        from isaacsim.core.experimental.objects import Cube
        from isaacsim.core.experimental.prims import GeomPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.robot_motion.cumotion import GraphBasedMotionPlanner, RmpFlowController, TrajectoryOptimizer
        from isaacsim.robot_motion.cumotion.impl.utils import isaac_sim_to_cumotion_pose
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools, PhysxSchema

        from experiments import arm_sim
        from experiments import ridgeback_sim as rs

        # ---- シーン：地面・ロボット・机・カメラ ----
        rs.build_scene(simulation_app, "multiphysics")
        arm_sim.zero_base_joint_friction()
        table_center = world_from_base(BASE_A, TABLE_CENTER_IN_A)
        table = Cube(
            "/World/table",
            sizes=1.0,
            scales=[list(TABLE_SIZE)],
            positions=[list(table_center)],
            orientations=[[math.cos(BASE_A[2] / 2), 0.0, 0.0, math.sin(BASE_A[2] / 2)]],
            colors="brown",
        )
        GeomPrim(table.paths, apply_collision_apis=True)
        targets_world = [
            world_from_base(BASE_A, (d * math.cos(math.radians(a)), d * math.sin(math.radians(a)), HAND_HEIGHT))
            for d, a in TARGETS_IN_A
        ]
        for i, p in enumerate(targets_world):  # 目標の目印（見た目だけ）
            Cube(f"/World/target_marker_{i}", sizes=0.03, positions=[list(p)], colors="green")
        summary["targets_world"] = targets_world
        eye = world_from_base(BASE_A, (-1.6, -2.4, 2.7))
        camera = rs.Camera("/World/arm_camera", eye=eye, target=[table_center[0], table_center[1], 0.7])

        # ---- 接触の検出：アームのリンクに接触レポートを付ける ----
        arm_links = [f"panda_link{i}" for i in range(1, 8)] + ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
        stage = rs.stage_utils.get_current_stage()
        for name in arm_links:
            api = PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(rs.link_path(name)))
            api.CreateThresholdAttr().Set(0.0)
        # 接触ペアごとに、最小のめり込み量 [m]（負＝めり込み）と最大の力積 [N·s] を記録する。
        # 接触レポートは接触判定のマージン内の「近接」も報告するので、実際の接触とは区別する
        contacts: dict[tuple[str, str], dict] = {}

        def on_contact(headers, data) -> None:
            for h in headers:
                a0 = str(PhysicsSchemaTools.intToSdfPath(h.actor0))
                a1 = str(PhysicsSchemaTools.intToSdfPath(h.actor1))
                rec = contacts.setdefault(tuple(sorted((a0, a1))), {"min_separation": math.inf, "max_impulse": 0.0})
                for k in range(h.contact_data_offset, h.contact_data_offset + h.num_contact_data):
                    d = data[k]
                    rec["min_separation"] = min(rec["min_separation"], float(d.separation))
                    imp = d.impulse
                    rec["max_impulse"] = max(rec["max_impulse"], math.sqrt(imp[0] ** 2 + imp[1] ** 2 + imp[2] ** 2))

        contact_sub = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)  # noqa: F841

        # ---- 物理開始、台車を A に置く ----
        SimulationManager.setup_simulation(dt=1.0 / 60.0)
        rs.play_and_warmup(simulation_app)
        dt = SimulationManager.get_physics_dt()
        art = rs.articulation()
        names = list(art.dof_names)
        base_idx = [names.index(n) for n in rs.BASE_JOINTS]
        arm_sim.set_robot_configuration(simulation_app, art, BASE_A)
        planning = arm_sim.ArmPlanning(art)
        summary["tracked_prims"] = planning.tracked_prims
        controlled = planning.cumotion_robot.controlled_joint_names
        controlled_idx = [names.index(n) for n in controlled]
        summary["controlled_joints"] = controlled
        default_q = [arm_sim.FRANKA_DEFAULT[n] for n in controlled]
        home_q = planning.solve_ik(world_from_base(BASE_A, HOME_HAND_IN_BASE), arm_sim.HAND_DOWN_WXYZ, seed=default_q)
        if home_q is None:
            raise RuntimeError("待機姿勢の IK が解けませんでした")
        summary["home_q"] = dict(zip(controlled, home_q.tolist(), strict=True))

        def reset_arm() -> None:
            """アームを待機姿勢に戻す（台車はそのまま）。"""
            q = art.get_dof_positions().numpy()[0].copy()
            q[controlled_idx] = home_q
            art.set_dof_positions(q)
            art.set_dof_velocities(np.zeros_like(q))
            art.set_dof_position_targets(q)
            for _ in range(10):
                simulation_app.update()

        def errors_to(target) -> tuple[float, float]:
            hp, hq = planning.tool_pose()
            return float(np.linalg.norm(hp - np.asarray(target))), quat_angle_error_wxyz(hq, arm_sim.HAND_DOWN_WXYZ)

        def robot_contacts() -> tuple[list, list]:
            """机・台車本体とアームの (実際の接触, 近接だけ) のリスト。

            実際の接触：めり込み 1mm 以上、または力積が 0 でない。
            """
            touching, near = [], []
            for (a, b), rec in contacts.items():
                pair = {a, b}
                if not (any("/World/table" in p for p in pair) or any(p.endswith("/base_link") for p in pair)):
                    continue
                item = [a, b, round(rec["min_separation"], 4), round(rec["max_impulse"], 6)]
                (touching if rec["min_separation"] < -0.001 or rec["max_impulse"] > 1e-9 else near).append(item)
            return touching, near

        def execute(method: str, target, label: str) -> dict:
            reset_arm()
            contacts.clear()
            t_plan, traj, ctrl = 0.0, None, None
            setpoint = planning.target_state(target, arm_sim.HAND_DOWN_WXYZ)
            if method == "rmpflow":
                ctrl = RmpFlowController(
                    cumotion_robot=planning.cumotion_robot,
                    cumotion_world_interface=planning.world_interface,
                    robot_joint_space=planning.joint_space,
                    robot_site_space=[arm_sim.TOOL_FRAME],
                    tool_frame=arm_sim.TOOL_FRAME,
                )
                ctrl.reset(planning.robot_state(), setpoint, 0.0)
                duration = RMP_DURATION_S
            elif method == "graph":
                planner = GraphBasedMotionPlanner(
                    cumotion_robot=planning.cumotion_robot,
                    cumotion_world_interface=planning.world_interface,
                    tool_frame=arm_sim.TOOL_FRAME,  # 省略すると panda_leftfingertip になる
                )
                q0 = art.get_dof_positions().numpy()[0][controlled_idx].astype(np.float64)
                t0 = time.perf_counter()
                path = planner.plan_to_pose_target(
                    q_initial=q0, position=np.asarray(target), orientation=np.asarray(arm_sim.HAND_DOWN_WXYZ)
                )
                if path is not None:
                    traj = path.to_minimal_time_joint_trajectory(
                        max_velocities=np.asarray(GRAPH_MAX_VEL),
                        max_accelerations=np.asarray(GRAPH_MAX_ACC),
                        robot_joint_space=planning.joint_space,
                        active_joints=controlled,
                    )
                t_plan = time.perf_counter() - t0
                if traj is None:
                    return {"method": method, "label": label, "planned": False, "planning_time_s": t_plan}
                duration = traj.duration + TRAJ_EXTRA_S
            else:
                optimizer = TrajectoryOptimizer(
                    cumotion_robot=planning.cumotion_robot,
                    robot_joint_space=planning.joint_space,
                    cumotion_world_interface=planning.world_interface,
                    # 省略すると設定の先頭の tool frame（panda_leftfingertip）が使われる
                    tool_frame=arm_sim.TOOL_FRAME,
                )
                base_pos, base_q = planning.arm_base_pose()
                goal_base = isaac_sim_to_cumotion_pose(
                    position_world_to_target=np.asarray(target),
                    orientation_world_to_target=np.asarray(arm_sim.HAND_DOWN_WXYZ),
                    position_world_to_base=base_pos,
                    orientation_world_to_base=base_q,
                )
                goal = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
                    translation_constraint=cumotion.TrajectoryOptimizer.TranslationConstraint.target(
                        goal_base.translation
                    ),
                    orientation_constraint=cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target(
                        goal_base.rotation
                    ),
                )
                q0 = art.get_dof_positions().numpy()[0][controlled_idx].astype(np.float64)
                t0 = time.perf_counter()
                traj = optimizer.plan_to_goal(q0, goal)
                t_plan = time.perf_counter() - t0
                if traj is None:
                    return {"method": method, "label": label, "planned": False, "planning_time_s": t_plan}
                duration = traj.duration + TRAJ_EXTRA_S
            ts, pos_errs, ang_errs = [], [], []
            for step in range(int(duration / dt)):
                t = (step + 1) * dt
                if ctrl is not None:
                    planning.apply(ctrl.forward(planning.robot_state(), setpoint, t))
                else:
                    planning.apply(traj.get_target_state(min(t, traj.duration)))
                simulation_app.update()
                pe, ae = errors_to(target)
                ts.append(t)
                pos_errs.append(pe)
                ang_errs.append(ae)
            camera.save(simulation_app, out_dir / f"{label}_{method}.png")
            settled = settle_time(ts, pos_errs, ang_errs, pos_tol=POS_TOL, yaw_tol=ANG_TOL)
            touching, near = robot_contacts()
            result = {
                "method": method,
                "label": label,
                "planned": True,
                "planning_time_s": t_plan,
                "trajectory_duration_s": traj.duration if traj is not None else None,
                "final_pos_err_m": pos_errs[-1],
                "final_ang_err_deg": math.degrees(ang_errs[-1]),
                "settle_time_s": settled,
                "success": is_success(pos_errs[-1], ang_errs[-1], pos_tol=POS_TOL, ang_tol=ANG_TOL),
                "contacts": touching,
                "near_contacts": near,
            }
            print(f"[arm] {label:20s} {method:8s} " + json.dumps(_brief(result)), flush=True)
            return result

        def run_phase(phase: str) -> None:
            reset_arm()
            camera.save(simulation_app, out_dir / f"{phase}_start.png")
            for i, target in enumerate(targets_world):
                for method in METHODS:
                    r = execute(method, target, f"{phase}_t{i}")
                    r.update({"phase": phase, "target_index": i, "base_pose": list(rs.pose2d(planning.base))})
                    summary["runs"].append(r)
            reset_arm()

        # ---- 段階 A：根元の姿勢を渡して試行 ----
        run_phase("A_synced")

        # ---- 台車を速度指令で B へ動かす（#2 の推奨方法）----
        base_b = (
            *world_from_base(BASE_A, (*BASE_B_OFFSET_IN_A, 0.0))[:2],
            BASE_A[2] + BASE_B_DYAW,
        )
        art.switch_dof_control_mode("velocity", dof_indices=base_idx)
        for _ in range(int(6.0 / dt)):
            cur = tuple(float(v) for v in art.get_dof_positions().numpy()[0][base_idx])
            cmd = velocity_command(cur, base_b, kp_lin=2.0, kp_ang=2.0, v_max=0.5, w_max=1.0)
            art.set_dof_velocity_targets([list(cmd)], dof_indices=base_idx)
            simulation_app.update()
        art.set_dof_velocity_targets([[0.0, 0.0, 0.0]], dof_indices=base_idx)
        art.switch_dof_control_mode("position", dof_indices=base_idx)
        q_hold = art.get_dof_positions().numpy()[0]
        art.set_dof_position_targets([list(q_hold[base_idx])], dof_indices=base_idx)
        for _ in range(30):
            simulation_app.update()
        summary["base_B_target"] = base_b
        summary["base_B_actual"] = list(rs.pose2d(planning.base))

        # ---- 段階 B：根元の姿勢を更新しない → 更新する ----
        run_phase("B_not_synced")
        planning.sync()
        run_phase("B_synced")

        summary["stats"] = _stats(summary["runs"])
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
        print(f"[arm] status={summary.get('status')} -> {out_dir}", flush=True)
    simulation_app.close(exit_code=exit_code)
    return exit_code


def _brief(r: dict) -> dict:
    keys = (
        "success",
        "final_pos_err_m",
        "final_ang_err_deg",
        "settle_time_s",
        "planning_time_s",
        "trajectory_duration_s",
        "contacts",
    )
    return {k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in keys if k in r}


def _stats(runs: list[dict]) -> dict:
    """段階 × 手段ごとの成功率・誤差・時間の集計。"""
    import statistics

    out = {}
    for phase in sorted({r["phase"] for r in runs}):
        for method in METHODS:
            rs_ = [r for r in runs if r["phase"] == phase and r["method"] == method]
            ok = [r for r in rs_ if r.get("success")]
            planned = [r for r in rs_ if r.get("planned")]
            settles = [r["settle_time_s"] for r in ok if r.get("settle_time_s") is not None]
            out[f"{phase}/{method}"] = {
                "n": len(rs_),
                "success": len(ok),
                "pos_err_mean_mm": statistics.mean(r["final_pos_err_m"] * 1000 for r in planned) if planned else None,
                "pos_err_max_mm": max(r["final_pos_err_m"] * 1000 for r in planned) if planned else None,
                "ang_err_max_deg": max(r["final_ang_err_deg"] for r in planned) if planned else None,
                "settle_mean_s": statistics.mean(settles) if settles else None,
                "planning_mean_s": statistics.mean(r["planning_time_s"] for r in rs_),
                "runs_with_contact": sum(1 for r in planned if r["contacts"]),
                "runs_with_near_contact_only": sum(1 for r in planned if r["near_contacts"] and not r["contacts"]),
            }
    return out


if __name__ == "__main__":
    sys.exit(main())
