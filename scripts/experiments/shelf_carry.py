"""Issue #15：低い棚（天板 0.45m）で掴んで置けるか、物を持ったまま台車を動かしてよいかを確かめる。

1. shelf：低い棚の正面（+x 側、隙間 10cm）に台車を止め、5 個の物体を棚の 3 か所から掴み、棚の別の位置に置く
2. carry：机の前（M1 の止め位置）で物体を掴み、待機姿勢のまま台車を速度指令で動かす
   （後退 0.8m → 左右へ 0.5m → ±90° 旋回 → 前進 0.8m）。手先に対する物体のずれと、落ちたかを真値で記録する

使い方:
    .venv\\Scripts\\python.exe scripts\\experiments\\shelf_carry.py                   # shelf と carry（指令は段階的）
    .venv\\Scripts\\python.exe scripts\\experiments\\shelf_carry.py --parts carry --profile ramp --speed 0.3

出力: outputs/m2_15/<run>/summary.json、frames/<試行>/*.png（各部の最初の試行）
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from restorebot.core.config import M1_BASE_POSE
from restorebot.core.grasp import TrialObservation, classify_trial, footprints, tilt_deg_from_quat_wxyz
from restorebot.core.navigation import facing_pose, in_reach
from restorebot.core.scene import HOME_V2, Placement
from restorebot.core.types import Pose2D
from restorebot.sim.app import start_app

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENE = HOME_V2
SIDE = SCENE.surfaces["side_table"]
SHELF_BASE = facing_pose(SIDE, SIDE.front, lateral=SIDE.center[1])  # (-0.73, -0.30)、-x を向く
SHELF_POSITIONS = [(-1.45, -0.55), (-1.45, -0.30), (-1.45, -0.05)]  # 棚の作業範囲の左右の端と中央
CARRY_START = (0.75, 0.65)  # 机の上、M1 の止め位置の正面
# 台車の動き（台車座標の速度 vx, vy [m/s]、wz [rad/s] を、上限の速さに対する割合で）と、進む量
CARRY_MOVES = [
    ("back", (-1.0, 0.0, 0.0), 0.8),  # 後退 0.8m
    ("left", (0.0, 1.0, 0.0), 0.5),  # 左へ 0.5m
    ("right", (0.0, -1.0, 0.0), 0.5),  # 右へ 0.5m（元の位置）
    ("turn_left", (0.0, 0.0, 1.0), math.pi / 2),  # 左に 90°
    ("turn_right", (0.0, 0.0, -1.0), math.pi / 2),  # 右に 90°（元の向き）
    ("forward", (1.0, 0.0, 0.0), 0.8),  # 前進 0.8m
]
FRAME_EVERY = 15


def park_placement(i: int) -> Placement:
    return Placement("floor", 0.3 + 0.25 * i, -1.3, 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="+", choices=["shelf", "carry"], default=["shelf", "carry"])
    ap.add_argument("--objects", nargs="+", default=[o.key for o in SCENE.objects])
    ap.add_argument("--profile", choices=["step", "ramp"], default="step", help="速度指令の変え方（段階的・台形）")
    ap.add_argument("--speed", type=float, default=0.5, help="並進の速さの上限 [m/s]（旋回は 2 倍の rad/s）")
    ap.add_argument("--accel", type=float, default=0.5, help="ramp のときの加速度の上限 [m/s²]（旋回は 2 倍）")
    ap.add_argument("--run", default=None)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    run = args.run or f"{'_'.join(args.parts)}_{args.profile}_{args.speed:g}"
    out = REPO_ROOT / "outputs" / "m2_15" / run
    out.mkdir(parents=True, exist_ok=True)

    simulation_app = start_app(headless=not args.gui)
    summary: dict = {"args": vars(args), "shelf_base": asdict(SHELF_BASE), "shelf": [], "carry": []}
    exit_code = 1
    try:
        import numpy as np
        from isaacsim.core.rendering_manager import ViewportManager
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
        from PIL import Image

        from restorebot.core.types import Status
        from restorebot.sim import robot as rb
        from restorebot.sim.arm import ArmPlanning
        from restorebot.sim.manipulation import ArmContext, pick_action, place_action, run_action
        from restorebot.sim.scene_builder import PHYSICS_DT, build_scene, object_path, set_robot_pose
        from restorebot.sim.world import SimWorld

        parked = {o.key: park_placement(i) for i, o in enumerate(SCENE.objects)}
        sim = build_scene(simulation_app, SCENE, SCENE.tidy, SHELF_BASE)
        world = SimWorld(sim)
        ctx = ArmContext(ArmPlanning(sim.robot))
        base_idx = sim.dof_indices(rb.BASE_JOINTS)

        rtx = RtxCamera("/World/check_camera")
        rtx.camera.set_focal_lengths([1.4])
        rtx.camera.set_apertures(horizontal_apertures=[2.0955])
        cam = CameraSensor(rtx, resolution=(480, 640), annotators=["rgb"])

        def look(eye, target):
            ViewportManager.set_camera_view("/World/check_camera", eye=list(eye), target=list(target))

        def save_frame(d: Path, i: int) -> None:
            rgb, _ = cam.get_data("rgb")
            if rgb is not None:
                d.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.ascontiguousarray(rgb.numpy()[:, :, :3], dtype=np.uint8)).save(d / f"{i:05d}.png")

        def reset(base: Pose2D, placements: dict[str, Placement]) -> None:
            set_robot_pose(simulation_app, sim, base, rb.STANDBY_ARM)
            full = {**parked, **placements}
            for key, p in full.items():
                if p.surface == "floor":
                    sim.objects[key].set_world_poses(
                        positions=[[p.x, p.y, SCENE.object(key).height / 2 + 0.003]], orientations=[[1.0, 0, 0, 0]]
                    )
                    sim.objects[key].set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
            world.set_placements({k: p for k, p in full.items() if p.surface != "floor"})
            for _ in range(30):
                simulation_app.update()
            ctx.planning.sync()

        def neighbors(key):
            return footprints(SCENE, {k: world.object_pose(k)[0] for k in SCENE.tidy if k != key}, key)

        def pick(key, base, frames_dir=None, counter=None):
            state = world.object_state(key)
            action = pick_action(
                ctx, SCENE, state, base, lambda: float(world.object_pose(key)[0][2]), object_path(key), neighbors(key)
            )

            def on_step(i):
                if frames_dir is not None and i % FRAME_EVERY == 0:
                    save_frame(frames_dir, counter[0] + i)

            r = run_action(simulation_app, action, PHYSICS_DT, on_step=on_step)
            if counter is not None:
                counter[0] += int(action.t / PHYSICS_DT)
            return r, action, np.asarray(state.position)

        # ---------------- 1. 低い棚 ----------------
        if "shelf" in args.parts:
            look((-0.55, -1.25, 1.35), (-1.5, -0.3, 0.45))
            for key in args.objects:
                for i, (x, y) in enumerate(SHELF_POSITIONS):
                    tx, ty = SHELF_POSITIONS[(i + 1) % len(SHELF_POSITIONS)]
                    reset(SHELF_BASE, {key: Placement("side_table", x, y, 0.4 * i)})
                    if not (in_reach(SHELF_BASE, (x, y)) and in_reach(SHELF_BASE, (tx, ty))):
                        raise RuntimeError(f"{key} {i}：届く範囲の外です")
                    frames_dir = out / "frames" / f"shelf_{key}_{i}" if i == 0 and key == args.objects[0] else None
                    counter = [0]
                    t0 = time.perf_counter()
                    r_pick, pick_act, p0 = pick(key, SHELF_BASE, frames_dir, counter)
                    target = SCENE.object_center(key, Placement("side_table", tx, ty))
                    rec = {
                        "key": key,
                        "start": [x, y],
                        "target": [tx, ty],
                        "pick": [r_pick.status.value, r_pick.reason],
                    }
                    max_lift = float(world.object_pose(key)[0][2] - p0[2])
                    place_done = False
                    if r_pick.status is Status.SUCCEEDED:
                        place = place_action(
                            ctx,
                            SCENE.object(key).shape,
                            target,
                            pick_act.hand_yaw,
                            SHELF_BASE,
                            object_path(key),
                            neighbors(key),
                        )
                        r_place = run_action(simulation_app, place, PHYSICS_DT)
                        rec["place"] = [r_place.status.value, r_place.reason]
                        place_done = r_place.status is Status.SUCCEEDED
                    for _ in range(60):
                        simulation_app.update()
                    p1, q1 = world.object_pose(key)
                    o = TrialObservation(
                        controller_done=place_done,
                        moved_before_grasp_m=0.0,
                        max_lift_m=max_lift,
                        dropped=False,
                        place_xy_err_m=float(math.hypot(p1[0] - target[0], p1[1] - target[1])),
                        tilt_deg=tilt_deg_from_quat_wxyz(q1),
                    )
                    rec.update(observation=asdict(o), outcome=classify_trial(o), wall_s=time.perf_counter() - t0)
                    summary["shelf"].append(rec)
                    print(
                        f"[shelf] {key:10s} {i} {rec['outcome']:20s} err={o.place_xy_err_m * 1000:5.1f}mm "
                        f"{r_pick.reason or ''} {(rec.get('place') or [None, ''])[1] or ''}",
                        flush=True,
                    )

        # ---------------- 2. 持ったまま台車を動かす ----------------
        if "carry" in args.parts:
            look((-1.6, -1.35, 2.2), (0.6, -0.4, 0.5))
            v_max, w_max = args.speed, 2 * args.speed
            a_max, aw_max = args.accel, 2 * args.accel
            for n, key in enumerate(args.objects):
                reset(M1_BASE_POSE, {key: Placement("desk", *CARRY_START, 0.0)})
                frames_dir = out / "frames" / f"carry_{key}" if n == 0 else None
                counter = [0]
                r_pick, _, _ = pick(key, M1_BASE_POSE, frames_dir, counter)
                rec = {"key": key, "pick": [r_pick.status.value, r_pick.reason], "moves": []}
                if r_pick.status is not Status.SUCCEEDED:
                    summary["carry"].append(rec)
                    print(f"[carry] {key}: 掴めなかった {r_pick.reason}", flush=True)
                    continue

                def rel_in_base(key=key):
                    """手先から見た物体の位置（台車座標。台車と一緒に回る）。"""
                    q = sim.robot.get_dof_positions().numpy()[0]
                    yaw = float(q[base_idx[2]])
                    d = world.object_pose(key)[0] - ctx.planning.tool_pose()[0]
                    c, s = math.cos(yaw), math.sin(yaw)
                    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]])

                rel0 = rel_in_base()
                sim.robot.switch_dof_control_mode("velocity", dof_indices=base_idx)
                step = 0
                for name, direction, amount in CARRY_MOVES:
                    is_turn = direction[2] != 0.0
                    vmax, amax = (w_max, aw_max) if is_turn else (v_max, a_max)
                    travelled, speed, max_dev = 0.0, 0.0, 0.0
                    while travelled < amount:
                        remaining = amount - travelled
                        if args.profile == "step":
                            speed = vmax
                        else:  # 台形：加速度の上限で加速し、止まれる距離に入ったら減速する
                            stop = speed * speed / (2 * amax)
                            target_speed = vmax if remaining > stop else 0.0
                            speed = (
                                min(target_speed, speed + amax * PHYSICS_DT)
                                if target_speed > speed
                                else max(target_speed, speed - amax * PHYSICS_DT)
                            )
                            speed = max(speed, 0.05 * vmax)  # 止まりきらないよう最小の速さ
                        speed = min(speed, remaining / PHYSICS_DT)
                        q = sim.robot.get_dof_positions().numpy()[0]
                        yaw = float(q[base_idx[2]])
                        c, s = math.cos(yaw), math.sin(yaw)
                        bx, by, bw = (d * speed for d in direction)
                        cmd = [c * bx - s * by, s * bx + c * by, bw]
                        sim.robot.set_dof_velocity_targets([cmd], dof_indices=base_idx)
                        simulation_app.update()
                        step += 1
                        travelled += speed * PHYSICS_DT
                        max_dev = max(max_dev, float(np.linalg.norm(rel_in_base() - rel0)))
                        if frames_dir is not None and step % FRAME_EVERY == 0:
                            save_frame(frames_dir, counter[0] + step)
                    sim.robot.set_dof_velocity_targets([[0.0, 0.0, 0.0]], dof_indices=base_idx)
                    for _ in range(30):  # 止まった後の揺れ
                        simulation_app.update()
                        step += 1
                        max_dev = max(max_dev, float(np.linalg.norm(rel_in_base() - rel0)))
                    rec["moves"].append({"move": name, "max_dev_m": max_dev})
                final_dev = float(np.linalg.norm(rel_in_base() - rel0))
                q = sim.robot.get_dof_positions().numpy()[0]
                rec.update(
                    final_dev_m=final_dev,
                    max_dev_m=max(m["max_dev_m"] for m in rec["moves"]),
                    dropped=final_dev > 0.05,
                    base_end=[float(q[i]) for i in base_idx],
                )
                summary["carry"].append(rec)
                print(
                    f"[carry] {key:10s} max_dev={rec['max_dev_m'] * 1000:6.1f}mm "
                    f"final_dev={final_dev * 1000:6.1f}mm dropped={rec['dropped']} "
                    f"per_move={[round(m['max_dev_m'] * 1000, 1) for m in rec['moves']]}",
                    flush=True,
                )
        summary["status"] = "ok"
        exit_code = 0
    except Exception as exc:
        traceback.print_exc()
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), "utf-8")
        print(f"[shelf_carry] status={summary.get('status')} -> {out}", flush=True)
    simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
