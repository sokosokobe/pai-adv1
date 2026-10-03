"""Issue #2：RidgebackFranka の台車を指令方法ごとに動かし、到達精度・所要時間・アームの揺れを計測する。

指令方法（台車の3つの仮想関節 x, y, yaw に対して）:
    position_step : 位置指令。目標値を一度に与える
    position_ramp : 位置指令。目標値を一定速度（v_max, w_max）で目標へ動かす
    velocity      : 速度指令。現在姿勢から比例制御で速度を作り、飽和させて与える（実機の cmd_vel に近い）

使い方::

    .venv\\Scripts\\python.exe scripts\\experiments\\ridgeback_base_motion.py
    .venv\\Scripts\\python.exe scripts\\experiments\\ridgeback_base_motion.py --obstacle light   # light|heavy|static

出力: outputs/m0_2/motion_<asset>[_obstacle_<kind>]/ に summary.json、各試行の時系列 CSV、前後の画像
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # experiments.* を import するため

from experiments.ridgeback_common import (  # noqa: E402  （Isaac Sim 非依存）
    classify_obstacle_outcome,
    max_deviation,
    pose_error,
    settle_time,
    step_toward,
    velocity_command,
)

# ------------------------------------------------------------
# 条件
# ------------------------------------------------------------
MODES = ("position_step", "position_ramp", "velocity")
PATTERNS = {  # 開始姿勢からの相対目標 (dx [m], dy [m], dyaw [rad])
    "forward_1m": (1.0, 0.0, 0.0),
    "rotate_90deg": (0.0, 0.0, math.pi / 2),
    "diagonal_1m_1m": (1.0, 1.0, 0.0),
}
OBSTACLE_GOAL = (2.0, 0.0, 0.0)  # 障害物試験：箱の向こうへ 2m 前進させる
BOX_SIZE = 0.4  # [m]
BOX_X = 1.2  # 箱の中心の x [m]
BOX_MASS = {"light": 5.0, "heavy": 50.0, "static": None}  # None = 固定（剛体にしない）

V_MAX, W_MAX = 0.5, 1.0  # [m/s], [rad/s]（position_ramp と velocity で共通）
KP_LIN, KP_ANG = 2.0, 2.0  # velocity の比例ゲイン
POS_TOL, YAW_TOL = 0.01, math.radians(1.0)  # 到達判定 1cm, 1°
DURATION_S = 8.0
SETTLE_STEPS = 30


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=["multiphysics", "legacy"], default="multiphysics")
    parser.add_argument("--obstacle", choices=["none", *BOX_MASS], default="none")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument(
        "--ground-contact",
        choices=["default", "filtered"],
        default="default",
        help="filtered: 台車本体（base_link）と地面の衝突を無効にする（摩擦の影響を調べるため）",
    )
    parser.add_argument(
        "--base-joint-friction",
        choices=["asset", "zero"],
        default="asset",
        help="zero: 台車の仮想関節の摩擦を 0 にする（アセットでは x, y に physxJoint:jointFriction=0.1）",
    )
    args = parser.parse_args()
    suffix = "" if args.obstacle == "none" else f"_obstacle_{args.obstacle}"
    suffix += "" if args.ground_contact == "default" else "_ground_filtered"
    suffix += "" if args.base_joint_friction == "asset" else "_nofriction"
    out_dir = REPO_ROOT / "outputs" / "m0_2" / f"motion_{args.asset}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    # ------------------------------------------------------------
    # SimulationApp を最初に起動（Isaac Sim のモジュールはこの後で import）
    # ------------------------------------------------------------
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
    summary: dict = {
        "asset": args.asset,
        "obstacle": args.obstacle,
        "ground_contact": args.ground_contact,
        "base_joint_friction": args.base_joint_friction,
        "conditions": {
            "v_max": V_MAX,
            "w_max": W_MAX,
            "kp_lin": KP_LIN,
            "kp_ang": KP_ANG,
            "pos_tol": POS_TOL,
            "yaw_tol_deg": math.degrees(YAW_TOL),
            "duration_s": DURATION_S,
        },
        "runs": [],
    }
    exit_code = 1
    try:
        import numpy as np
        from isaacsim.core.experimental.objects import Cube
        from isaacsim.core.experimental.prims import GeomPrim, RigidPrim, XformPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Usd, UsdGeom, UsdPhysics

        from experiments import ridgeback_sim as rs

        # ---- シーン構築：地面・ライト・ロボット・向き確認用の目印・（障害物）・カメラ ----
        rs.build_scene(simulation_app, args.asset)
        # 台車の仮想関節の関節摩擦（PhysX の係数 physxJoint:jointFriction）。物理開始前に USD 上で変更する
        summary["base_joint_friction_asset"] = {}
        for joint_name in rs.BASE_JOINTS:
            attr = rs.find_joint_prim(joint_name).GetAttribute("physxJoint:jointFriction")
            summary["base_joint_friction_asset"][joint_name] = attr.Get() if attr and attr.HasValue() else None
            if args.base_joint_friction == "zero" and attr:
                attr.Set(0.0)
        if args.ground_contact == "filtered":
            # 台車本体と地面の組だけ衝突判定から外す（箱など他の物体との衝突は残る）
            base_prim = rs.stage_utils.get_current_stage().GetPrimAtPath(rs.base_link_path())
            UsdPhysics.FilteredPairsAPI.Apply(base_prim).CreateFilteredPairsRel().AddTarget("/World/ground_plane")
        # 画像の向きの確認用（見た目だけ。衝突なし）：+x 側に赤、+y 側に緑
        Cube("/World/marker_plus_x", sizes=0.15, positions=[3.2, 0.0, 0.075], colors="red")
        Cube("/World/marker_plus_y", sizes=0.15, positions=[0.0, 2.2, 0.075], colors="green")
        box = None
        if args.obstacle != "none":
            box_shape = Cube(
                "/World/box",
                sizes=1.0,
                scales=[BOX_SIZE] * 3,
                positions=[BOX_X, 0.0, BOX_SIZE / 2 + 0.001],
                colors="blue",
            )
            GeomPrim(box_shape.paths, apply_collision_apis=True)
            if BOX_MASS[args.obstacle] is not None:
                box = RigidPrim(box_shape.paths)
                box.set_masses([BOX_MASS[args.obstacle]])
                box.set_default_state(positions=[[BOX_X, 0.0, BOX_SIZE / 2 + 0.001]], orientations=[[1, 0, 0, 0]])
        box_xform = XformPrim("/World/box") if args.obstacle != "none" else None
        camera = rs.Camera("/World/motion_camera", eye=[-4.5, -7.0, 8.0], target=[1.2, 0.8, 0.0])

        # ---- 物理開始 ----
        SimulationManager.setup_simulation(dt=1.0 / 60.0)
        rs.play_and_warmup(simulation_app)
        dt = SimulationManager.get_physics_dt()
        art = rs.articulation()
        names = list(art.dof_names)
        base_idx = [names.index(n) for n in rs.BASE_JOINTS]
        arm_idx = [names.index(f"panda_joint{i}") for i in range(1, 8)]
        home = art.get_dof_positions().numpy()[0].copy()
        base = XformPrim(rs.base_link_path())
        hand = XformPrim(rs.link_path("panda_hand"))

        # 台車の x 方向の半分の長さ（障害物にめり込んだかの判定用）
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"], useExtentsHint=True)
        stage = base.prims[0].GetStage()
        rng = bbox.ComputeWorldBound(stage.GetPrimAtPath(base.paths[0])).ComputeAlignedRange()
        base_half_len_x = float((rng.GetMax()[0] - rng.GetMin()[0]) / 2)
        summary["base_half_length_x_m"] = base_half_len_x

        def hand_in_base() -> np.ndarray:
            """台車から見た手先の位置（台車座標系）。"""
            pb, _ = rs.world_pose(base)
            ph, _ = rs.world_pose(hand)
            yaw = rs.pose2d(base)[2]
            c, s = math.cos(-yaw), math.sin(-yaw)
            d = ph - pb
            return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])

        def reset(mode: str) -> None:
            """ロボット（と箱）を初期状態に戻し、指令方法に合わせてドライブを切り替える。"""
            art.switch_dof_control_mode("position", dof_indices=base_idx)
            art.set_dof_positions(home)
            art.set_dof_velocities(np.zeros_like(home))
            art.set_dof_position_targets(home)
            art.set_dof_velocity_targets(np.zeros_like(home))
            if box is not None:
                box.reset_to_default_state()
            if mode == "velocity":
                art.switch_dof_control_mode("velocity", dof_indices=base_idx)
            for _ in range(SETTLE_STEPS):
                simulation_app.update()

        def run(name: str, mode: str, delta: tuple[float, float, float]) -> dict:
            reset(mode)
            q0 = art.get_dof_positions().numpy()[0][
                base_idx
            ]  # 関節座標（= ワールド基準。ridgeback_inspect で確認済み）
            start = tuple(float(v) for v in q0)
            goal = (start[0] + delta[0], start[1] + delta[1], start[2] + delta[2])
            p_link0 = rs.pose2d(base)
            goal_link = (p_link0[0] + delta[0], p_link0[1] + delta[1], p_link0[2] + delta[2])
            hand0 = hand_in_base()
            arm0 = art.get_dof_positions().numpy()[0][arm_idx]
            box0 = rs.world_pose(box_xform)[0] if box_xform is not None else None
            camera.save(simulation_app, out_dir / f"{name}_{mode}_before.png")

            ramp_target = start
            if mode == "position_step":
                art.set_dof_position_targets([list(goal)], dof_indices=base_idx)
            rows, hand_pts, arm_dev = [], [], 0.0
            for step in range(int(DURATION_S / dt)):
                q = art.get_dof_positions().numpy()[0]
                cur = tuple(float(v) for v in q[base_idx])
                if mode == "position_ramp":
                    ramp_target = step_toward(ramp_target, goal, V_MAX * dt, W_MAX * dt)
                    art.set_dof_position_targets([list(ramp_target)], dof_indices=base_idx)
                elif mode == "velocity":
                    cmd = velocity_command(cur, goal, kp_lin=KP_LIN, kp_ang=KP_ANG, v_max=V_MAX, w_max=W_MAX)
                    art.set_dof_velocity_targets([list(cmd)], dof_indices=base_idx)
                simulation_app.update()
                # 計測（真値：台車リンクのワールド姿勢）
                p = rs.pose2d(base)
                err = pose_error(p, goal_link)
                v = art.get_dof_velocities().numpy()[0][base_idx]
                hb = hand_in_base()
                hand_pts.append(hb)
                arm_dev = max(arm_dev, float(np.max(np.abs(art.get_dof_positions().numpy()[0][arm_idx] - arm0))))
                rows.append(
                    {
                        "t": (step + 1) * dt,
                        "x": p[0],
                        "y": p[1],
                        "yaw": p[2],
                        "pos_err": err.distance,
                        "yaw_err": err.dyaw,
                        "vx": float(v[0]),
                        "vy": float(v[1]),
                        "wz": float(v[2]),
                        "hand_dx": float(hb[0] - hand0[0]),
                        "hand_dy": float(hb[1] - hand0[1]),
                        "hand_dz": float(hb[2] - hand0[2]),
                    }
                )
            camera.save(simulation_app, out_dir / f"{name}_{mode}_after.png")

            # ---- 集計 ----
            with (out_dir / f"{name}_{mode}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            ts = [r["t"] for r in rows]
            speeds = [math.hypot(r["vx"], r["vy"]) for r in rows]
            result = {
                "pattern": name,
                "mode": mode,
                "goal_delta": delta,
                "final_pos_err_m": rows[-1]["pos_err"],
                "final_yaw_err_deg": math.degrees(rows[-1]["yaw_err"]),
                "settle_time_s": settle_time(
                    ts, [r["pos_err"] for r in rows], [r["yaw_err"] for r in rows], pos_tol=POS_TOL, yaw_tol=YAW_TOL
                ),
                "max_overshoot_m": _max_overshoot(rows, p_link0, goal_link),
                "peak_speed_mps": max(speeds),
                "peak_yaw_rate_radps": max(abs(r["wz"]) for r in rows),
                "peak_accel_mps2": max(abs(speeds[i] - speeds[i - 1]) / dt for i in range(1, len(speeds))),
                "hand_sway_max_m": max_deviation(np.array(hand_pts), hand0),
                "arm_joint_dev_max_rad": arm_dev,
            }
            if box_xform is not None:
                box1 = rs.world_pose(box_xform)[0]
                moved = float(np.linalg.norm(box1[:2] - box0[:2]))
                progress = (rows[-1]["x"] - p_link0[0]) / delta[0]
                result.update(
                    {
                        "box_moved_m": moved,
                        "base_progress_ratio": progress,
                        "base_final_x_m": rows[-1]["x"],
                        "box_final_x_m": float(box1[0]),
                        # 台車の前面 - 箱の後面（正ならめり込み）
                        "overlap_m": rows[-1]["x"] + base_half_len_x - (float(box1[0]) - BOX_SIZE / 2),
                        "outcome": classify_obstacle_outcome(box_moved=moved, base_progress=progress),
                    }
                )
            print(f"[motion] {name:16s} {mode:14s} " + json.dumps(_brief(result)), flush=True)
            return result

        patterns = {"obstacle_forward_2m": OBSTACLE_GOAL} if args.obstacle != "none" else PATTERNS
        for name, delta in patterns.items():
            for mode in args.modes:
                summary["runs"].append(run(name, mode, delta))
        summary["status"] = "ok"
        exit_code = 0
    except Exception as exc:
        traceback.print_exc()
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[motion] status={summary.get('status')} -> {out_dir}", flush=True)
    simulation_app.close(exit_code=exit_code)
    return exit_code


def _max_overshoot(rows: list[dict], start: tuple, goal: tuple) -> float:
    """開始→目標の方向に、目標をどれだけ行き過ぎたかの最大値 [m]（並進がないパターンは 0）。"""
    ux, uy = goal[0] - start[0], goal[1] - start[1]
    norm = math.hypot(ux, uy)
    if norm < 1e-9:
        return 0.0
    ux, uy = ux / norm, uy / norm
    return max(0.0, max((r["x"] - goal[0]) * ux + (r["y"] - goal[1]) * uy for r in rows))


def _brief(result: dict) -> dict:
    keys = (
        "final_pos_err_m",
        "final_yaw_err_deg",
        "settle_time_s",
        "max_overshoot_m",
        "peak_speed_mps",
        "peak_accel_mps2",
        "hand_sway_max_m",
        "outcome",
        "box_moved_m",
        "base_progress_ratio",
        "overlap_m",
    )
    return {k: (round(result[k], 4) if isinstance(result[k], float) else result[k]) for k in keys if k in result}


if __name__ == "__main__":
    sys.exit(main())
