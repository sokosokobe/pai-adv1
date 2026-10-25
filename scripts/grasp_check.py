"""#13：M1 の止め位置から、机の物体を掴んで置けるかを確かめる（シーンの構築・天井カメラ・掴む・置く）。

使い方:
    .venv\\Scripts\\python.exe scripts\\grasp_check.py            # すべて（作業範囲の 5 か所、隣の物体、散らかし方）
    .venv\\Scripts\\python.exe scripts\\grasp_check.py --quick    # 統合テスト用：天井カメラと 1 回だけ
    .venv\\Scripts\\python.exe scripts\\grasp_check.py --gui

出力（<out>、既定は outputs/m1_13/）:
    summary.json            試行ごとの結果（成功・失敗の分類、配置の誤差、隣の物体の動き、段階の記録）
    ceiling_tidy.png        天井カメラの画像（いつもの配置。画像の上がワールドの +y、右が +x のはず）
    frames/<試行>/*.png      斜め上からの連番画像（各区分の最初の試行）
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from restorebot.core.config import M1_BASE_POSE
from restorebot.core.grasp import TrialObservation, classify_trial, footprints, tilt_deg_from_quat_wxyz
from restorebot.core.navigation import in_reach
from restorebot.core.scene import HOME_V2, Placement, generate_scenario
from restorebot.sim.app import start_app

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE = HOME_V2
BASE = M1_BASE_POSE
DESK_KEYS = [k for k, p in SCENE.tidy.items() if p.surface == "desk"]
# 作業範囲の端を含む 5 か所（M1 の止め位置から楽に届く範囲の左右の端・奥・手前）と、そこでの物体の向き
GRID = {
    "left": (0.40, 0.62, 0.0),
    "right": (1.10, 0.62, 0.6),
    "far_left": (0.55, 0.80, -0.9),
    "far_right": (0.95, 0.80, 1.4),
    "near": (0.75, 0.62, 2.5),
}
CLEARANCE = 0.05  # 散らかし方の隙間（外接円どうし。#11）
FRAME_EVERY = 15  # 連番画像：4Hz
NEIGHBOR_TOL = 0.01  # 隣の物体が動いてよい量 [m]


def park_placement(i: int) -> Placement:
    """使わない物体を置いておく場所（床の上、ロボットと家具から離れた所。天板の上には置かない）。"""
    return Placement("floor", -0.4 + 0.25 * i, -1.25, 0.0)


@dataclass
class Trial:
    name: str
    key: str
    start: Placement
    target: Placement
    others: dict[str, Placement] = field(default_factory=dict)  # 机・棚の上に置く、ほかの物体
    frames: bool = False


def neighbor_along_fingers(key: str, p: Placement, neighbor: str, hand_yaw: float, sign: float = 1.0) -> Placement:
    """物体 key（位置 p）から、指が閉じる方向（手先の y 方向）に隙間 CLEARANCE をあけた隣の物体の置き方。"""
    # 真下向きの手先の y 方向は、ワールドで (sin yaw, -cos yaw)
    d = SCENE.object(key).footprint_radius + SCENE.object(neighbor).footprint_radius + CLEARANCE
    return Placement(p.surface, p.x + sign * d * math.sin(hand_yaw), p.y - sign * d * math.cos(hand_yaw), 0.0)


def build_trials(quick: bool) -> list[Trial]:
    from restorebot.core.grasp import grasp_yaw

    home = SCENE.tidy
    if quick:
        x, y, yaw = GRID["left"]
        return [Trial("quick_cup_left", "cup", Placement("desk", x, y, yaw), home["cup"], frames=True)]
    trials = []
    # 1. 作業範囲の 5 か所 × 机の 3 個 → いつもの位置
    for key in DESK_KEYS:
        for i, (pos_name, (x, y, yaw)) in enumerate(GRID.items()):
            trials.append(Trial(f"grid_{key}_{pos_name}", key, Placement("desk", x, y, yaw), home[key], frames=i == 0))
    # 2. 隣の物体との隙間（指が閉じる方向に 5cm）：掴むときと置くときの両方
    cases = [("cup", "pen_stand", 0.0), ("remote", "cup", 0.4), ("pen_stand", "remote", 0.0)]
    for n, (key, neighbor, yaw) in enumerate(cases):
        start = Placement("desk", 0.75, 0.64, yaw)
        hy = grasp_yaw(SCENE.object(key).shape, yaw, BASE.yaw)
        nb = neighbor_along_fingers(key, start, neighbor, hy)
        trials.append(Trial(f"neighbor_pick_{key}", key, start, home[key], {neighbor: nb}, frames=n == 0))
        nb_place = neighbor_along_fingers(key, home[key], neighbor, hy, sign=-1.0)
        start2 = Placement("desk", 0.75, 0.62, yaw)
        trials.append(Trial(f"neighbor_place_{key}", key, start2, home[key], {neighbor: nb_place}))
    return trials


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--no-frames", action="store_true", help="斜め上のカメラ（連番画像）を作らない")
    ap.add_argument("--only", nargs="*", default=None, help="名前にこの文字列を含む試行だけを実行する")
    ap.add_argument("--scenario-frames", action="store_true", help="散らかし方の試行でも連番画像を保存する")
    ap.add_argument("--seeds", type=int, default=4, help="M1 の散らかし方で試すシードの数（--quick では 0）")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "m1_13")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    simulation_app = start_app(headless=not args.gui)
    summary: dict = {"base_pose": asdict(BASE), "quick": args.quick, "trials": [], "scenarios": []}
    exit_code = 1
    t_wall = time.perf_counter()
    try:
        import numpy as np
        from isaacsim.core.rendering_manager import ViewportManager
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
        from PIL import Image

        from restorebot.core.tasks import plan_tasks
        from restorebot.core.types import MoveInstruction, Status
        from restorebot.sim import robot as rb
        from restorebot.sim.arm import ArmPlanning
        from restorebot.sim.manipulation import ArmContext, pick_action, place_action, run_action
        from restorebot.sim.scene_builder import PHYSICS_DT, build_scene, object_path, set_robot_pose
        from restorebot.sim.world import SimCeilingCamera, SimWorld

        parked = {o.key: park_placement(i) for i, o in enumerate(SCENE.objects)}
        sim = build_scene(simulation_app, SCENE, SCENE.tidy, BASE)
        world = SimWorld(sim)
        ceiling = SimCeilingCamera(sim, simulation_app)
        obs = ceiling.capture()
        Image.fromarray(obs.rgb).save(args.out / "ceiling_tidy.png")
        summary["ceiling"] = {"shape": list(obs.rgb.shape), "depth_min": float(np.nanmin(obs.depth))}

        # 斜め上から机を見るカメラ（連番画像用）
        side = None
        if not args.no_frames:
            side_path = "/World/side_camera"
            side = CameraSensor(RtxCamera(side_path), resolution=(480, 640), annotators=["rgb"])
            ViewportManager.set_camera_view(side_path, eye=[0.75 - 1.2, -1.0, 1.8], target=[0.75, 0.75, 0.7])

        ctx = ArmContext(ArmPlanning(sim.robot))
        summary["tracked_prims"] = ctx.planning.tracked_prims

        def save_frame(dir_: Path, i: int) -> None:
            if side is None:
                return
            rgb, _ = side.get_data("rgb")
            if rgb is not None:
                dir_.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.ascontiguousarray(rgb.numpy()[:, :, :3], dtype=np.uint8)).save(dir_ / f"{i:05d}.png")

        def reset(placements: dict[str, Placement]) -> None:
            set_robot_pose(simulation_app, sim, BASE, rb.STANDBY_ARM)
            full = {k: parked[k] for k in parked}
            full.update(placements)
            # 床の上に置く場所は SceneSpec の天板ではないので、ここで中心の高さを直接与える
            for key, p in full.items():
                if p.surface == "floor":
                    body = sim.objects[key]
                    body.set_world_poses(
                        positions=[[p.x, p.y, SCENE.object(key).height / 2 + 0.003]], orientations=[[1.0, 0, 0, 0]]
                    )
                    body.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)))
            world.set_placements({k: p for k, p in full.items() if p.surface != "floor"})
            for _ in range(30):
                simulation_app.update()

        def pick_and_place(key: str, target_center, frames_dir: Path | None) -> dict:
            """物体 key を真値で掴み、target_center に置く。ほかの物体の動きも記録する。"""
            others0 = {k: world.object_pose(k)[0] for k in SCENE.tidy if k != key}
            state = world.object_state(key)
            p0 = np.asarray(state.position)
            rec = {"moved_before_grasp": None, "max_lift": 0.0, "lifted": False, "dropped": False}
            counter = {"i": 0}
            tilts: list[tuple[str, float, float]] = []  # (動作:段階, 時刻, 物体の傾き [deg])。段階が変わるたびに記録
            trace: list[list] = []  # [動作:段階, 時刻, 物体 x, y, z, 手先 x, y, z]（5 ステップごと）

            def monitor(action):
                def on_step(_):
                    counter["i"] += 1
                    p, q = world.object_pose(key)
                    lift = p[2] - p0[2]
                    label = f"{action.name.split(':')[0]}:{action.seq.stage.name}"
                    if not tilts or tilts[-1][0] != label:
                        tilts.append((label, round(action.t, 3), round(tilt_deg_from_quat_wxyz(q), 1)))
                    if (
                        counter["i"] % 5 == 0
                    ):  # 物体と手先の位置の記録（運んでいる間に滑ったか、何かに当たったかを見分ける）
                        hand = ctx.planning.tool_pose()[0]
                        trace.append([label, round(action.t, 3), *[round(float(c), 4) for c in (*p, *hand)]])
                    if (
                        rec["moved_before_grasp"] is None
                        and action.name.startswith("pick")
                        and (action.seq.stage.name == "grasp")
                    ):
                        rec["moved_before_grasp"] = float(np.linalg.norm(p[:2] - p0[:2]))
                    rec["max_lift"] = max(rec["max_lift"], float(lift))
                    rec["lifted"] = rec["lifted"] or lift > 0.03
                    # 落としたかどうかは、持ち上げてから置く場所の真上に着くまで（place の下降より前）だけで判定する
                    stage = action.seq.stage.name
                    carrying = stage in ("lift", "standby") if action.name.startswith("pick") else stage == "approach"
                    if rec["lifted"] and carrying and lift < 0.01:
                        rec["dropped"] = True
                    if frames_dir is not None and counter["i"] % FRAME_EVERY == 0:
                        save_frame(frames_dir, counter["i"])

                return on_step

            t0 = time.perf_counter()
            neighbors = footprints(SCENE, others0, key)
            pick = pick_action(
                ctx, SCENE, state, BASE, lambda: float(world.object_pose(key)[0][2]), object_path(key), neighbors
            )
            r_pick = run_action(simulation_app, pick, PHYSICS_DT, on_step=monitor(pick))
            out = {
                "key": key,
                "start": [float(c) for c in p0],
                "target": [float(c) for c in target_center],
                "hand_yaw": pick.hand_yaw,
                "pick": {"status": r_pick.status.value, "reason": r_pick.reason, "stages": pick.log, "t": pick.t},
            }
            place = None
            if r_pick.status is Status.SUCCEEDED:
                shape = SCENE.object(key).shape
                place = place_action(ctx, shape, target_center, pick.hand_yaw, BASE, object_path(key), neighbors)
                r_place = run_action(simulation_app, place, PHYSICS_DT, on_step=monitor(place))
                out["place"] = {"status": r_place.status.value, "reason": r_place.reason, "stages": place.log}
                out["place"]["t"] = place.t
                out["place_hand_yaw"] = place.hand_yaw
            for _ in range(int(1.0 / PHYSICS_DT)):  # 置いた後に落ち着くのを待つ
                simulation_app.update()
            p1, q1 = world.object_pose(key)
            out["tilts"] = tilts + [("settled", 0.0, round(tilt_deg_from_quat_wxyz(q1), 1))]
            out["trace"] = trace
            o = TrialObservation(
                controller_done=place is not None and out["place"]["status"] == "succeeded",
                moved_before_grasp_m=rec["moved_before_grasp"] or 0.0,
                max_lift_m=rec["max_lift"],
                dropped=rec["dropped"],
                place_xy_err_m=float(math.hypot(p1[0] - target_center[0], p1[1] - target_center[1])),
                tilt_deg=tilt_deg_from_quat_wxyz(q1),
            )
            out["observation"] = asdict(o)
            out["outcome"] = classify_trial(o)
            out["others_moved_m"] = {k: float(np.linalg.norm(world.object_pose(k)[0] - p)) for k, p in others0.items()}
            out["wall_s"] = time.perf_counter() - t0
            return out

        # ---- 1・2. 決めた位置からの試行 ----
        trials_to_run = [t for t in build_trials(args.quick) if not args.only or any(s in t.name for s in args.only)]
        for trial in trials_to_run:
            reset({trial.key: trial.start, **trial.others})
            if not in_reach(BASE, (trial.start.x, trial.start.y)):
                raise RuntimeError(f"{trial.name}：開始位置が届く範囲の外です")
            target = SCENE.object_center(trial.key, trial.target)
            frames_dir = args.out / "frames" / trial.name if trial.frames else None
            r = pick_and_place(trial.key, target, frames_dir)
            r["name"] = trial.name
            r["neighbors"] = {k: asdict(p) for k, p in trial.others.items()}
            neighbor_moves = [r["others_moved_m"][k] for k in trial.others]
            r["neighbor_ok"] = all(m < NEIGHBOR_TOL for m in neighbor_moves)
            summary["trials"].append(r)
            print(
                f"[grasp_check] {trial.name:28s} {r['outcome']:22s} "
                f"err={r['observation']['place_xy_err_m'] * 1000:5.1f}mm "
                f"neighbors={[round(m * 1000, 1) for m in neighbor_moves]}mm wall={r['wall_s']:.0f}s "
                f"{r['pick']['reason'] or ''} {r.get('place', {}).get('reason') or ''}",
                flush=True,
            )

        # ---- 3. M1 の散らかし方：動かした物体を真値で、近い順に戻す ----
        for seed in range(0 if args.quick else args.seeds):
            sc = generate_scenario(SCENE, seed, surfaces=("desk",), reach_from=BASE)
            reset(dict(sc.placements))
            instructions = [
                MoveInstruction(k, (0.0, 0.0), (0.0, 0.0), SCENE.object_center(k, SCENE.tidy[k]), "desk")
                for k in sc.moved
            ]
            tasks = plan_tasks(instructions, world, SCENE, BASE)
            frames_root = args.out / "frames" / f"scenario_{seed}" if args.scenario_frames else None
            results = [
                pick_and_place(t.key, t.place_position, frames_root / t.key if frames_root else None) for t in tasks
            ]
            final = {k: world.object_pose(k)[0] for k in SCENE.tidy}
            errors = {
                k: float(math.hypot(final[k][0] - SCENE.tidy[k].x, final[k][1] - SCENE.tidy[k].y)) for k in sc.moved
            }
            untouched = {
                k: float(np.linalg.norm(final[k][:2] - np.array([sc.placements[k].x, sc.placements[k].y])))
                for k in SCENE.tidy
                if k not in sc.moved
            }
            ok = all(r["outcome"] == "success" for r in results) and all(m < NEIGHBOR_TOL for m in untouched.values())
            summary["scenarios"].append(
                {
                    "seed": seed,
                    "moved": list(sc.moved),
                    "order": [t.key for t in tasks],
                    "results": results,
                    "home_err_m": errors,
                    "untouched_moved_m": untouched,
                    "ok": ok,
                }
            )
            print(
                f"[grasp_check] scenario seed={seed} moved={list(sc.moved)} ok={ok} "
                f"err_mm={ {k: round(v * 1000, 1) for k, v in errors.items()} } "
                f"untouched_mm={ {k: round(v * 1000, 1) for k, v in untouched.items()} }",
                flush=True,
            )

        trials = summary["trials"]
        summary["stats"] = {
            "trials": len(trials),
            "success": sum(t["outcome"] == "success" for t in trials),
            "neighbor_ok": sum(t["neighbor_ok"] for t in trials),
            "place_err_max_mm": max((t["observation"]["place_xy_err_m"] * 1000 for t in trials), default=None),
            "scenarios_ok": sum(s["ok"] for s in summary["scenarios"]),
            "scenarios": len(summary["scenarios"]),
        }
        summary["status"] = "ok"
        exit_code = 0
    except Exception as exc:
        traceback.print_exc()
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        summary["wall_s"] = time.perf_counter() - t_wall
        (args.out / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        print(f"[grasp_check] status={summary.get('status')} stats={summary.get('stats')} -> {args.out}", flush=True)
    simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
