"""1 回の実行（エピソード）を Isaac Sim で動かす（設計書 3.2）。

準備（動画には入れない）：
    1. ロボットを部屋の外に置き、いつもの配置で「いつもの写真」を撮る
    2. 台車を M1 の止め位置に置き、シードで決まる散らかし方に物体を置く
本番：core.episode.Episode（撮影姿勢で今の写真 → VLM → 掴んで置く → 片付けた後の写真）

使い方:
    .venv\\Scripts\\python.exe scripts\\run_episode.py --seed 0     # VLM は真値から作る応答（API を呼ばない）
    .venv\\Scripts\\python.exe scripts\\run_episode.py --seed 0 --vlm claude --record
    .venv\\Scripts\\python.exe scripts\\run_episode.py --seed 0 --gui

出力（outputs/runs/<日時>_m1_seed<シード>_<VLM>/）:
    log.json                      記録（VLM の応答、タスクごとの結果、時間と費用）
    reference_* / current_* / after_*   写真（RGB の PNG、深度の npy）
    result.json                   真値での結果（物体ごとの戻し先からの誤差、ほかの物体の動き）
    overview.mp4 / follow.mp4 / ceiling.mp4   動画（--record のとき）
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from restorebot.core.config import M1_BASE_POSE
from restorebot.core.scene import HOME_V2, generate_scenario
from restorebot.core.types import Pose2D
from restorebot.sim.app import start_app

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE = HOME_V2
PARKED = Pose2D(6.0, 0.0, 0.0)  # いつもの写真を撮るときのロボットの置き場所（部屋の外。天井カメラに写らない）
SUCCESS_TOL = 0.03  # 成功の基準：戻し先から水平 3cm 以内（設計書 13 章）


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--milestone", choices=["m1"], default="m1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vlm", choices=["truth", "claude"], default="truth")
    ap.add_argument("--record", action="store_true", help="撮影用のカメラの動画を作る")
    ap.add_argument("--keep-frames", action="store_true", help="動画にした後も連番画像を残す")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.out or REPO_ROOT / "outputs" / "runs" / f"{stamp}_{args.milestone}_seed{args.seed}_{args.vlm}"
    out.mkdir(parents=True, exist_ok=True)

    simulation_app = start_app(headless=not args.gui)
    result: dict = {"seed": args.seed, "vlm": args.vlm, "milestone": args.milestone}
    exit_code = 1
    t_wall = time.perf_counter()
    try:
        from restorebot.core.episode import Episode, EpisodeConfig
        from restorebot.core.vlm.client import DummyVlmClient, make_client, truth_response
        from restorebot.sim import robot as rb
        from restorebot.sim.arm import ArmPlanning
        from restorebot.sim.manipulation import ArmContext
        from restorebot.sim.robot_adapter import SimRobot
        from restorebot.sim.scene_builder import PHYSICS_DT, build_scene, set_robot_pose
        from restorebot.sim.world import SimCeilingCamera, SimWorld

        scenario = generate_scenario(SCENE, args.seed, surfaces=("desk",), reach_from=M1_BASE_POSE)
        result["moved"] = list(scenario.moved)

        # ---- 準備 1：いつもの写真（ロボットは部屋の外）----
        sim = build_scene(simulation_app, SCENE, SCENE.tidy, PARKED)
        world = SimWorld(sim)
        camera = SimCeilingCamera(sim, simulation_app)
        reference = camera.capture()

        # ---- 準備 2：台車を止め位置に置き、物体を散らかす ----
        set_robot_pose(simulation_app, sim, M1_BASE_POSE, rb.STANDBY_ARM)
        world.set_placements(dict(scenario.placements))
        for _ in range(60):
            simulation_app.update()
        start = {k: world.object_pose(k)[0] for k in SCENE.tidy}
        ctx = ArmContext(ArmPlanning(sim.robot))
        robot = SimRobot(SCENE, sim, world, ctx)

        if args.vlm == "truth":
            current = {k: tuple(start[k]) for k in scenario.moved}
            target = {k: SCENE.object_center(k, SCENE.tidy[k]) for k in scenario.moved}
            vlm = DummyVlmClient(truth_response(SCENE, current, target), name="truth")
        else:
            vlm = make_client(args.vlm)

        recorder = None
        if args.record:
            from restorebot.sim.recorder import Recorder

            recorder = Recorder(sim, out, world.robot_pose)

        class Stepper:
            def step(self) -> float:
                simulation_app.update()
                if recorder is not None:
                    recorder.on_step()
                return PHYSICS_DT

        episode = Episode(
            SCENE,
            world,
            camera,
            robot,
            vlm,
            Stepper(),
            EpisodeConfig(milestone=args.milestone),
            meta={"seed": args.seed, "vlm": vlm.name, "moved_truth": list(scenario.moved)},
        )
        episode.reference = reference
        log = episode.run()
        log.save(out)

        # ---- 真値での結果（評価の簡易版。本格的な評価は #18）----
        final = {k: world.object_pose(k)[0] for k in SCENE.tidy}
        errors = {
            k: float(math.hypot(final[k][0] - SCENE.tidy[k].x, final[k][1] - SCENE.tidy[k].y)) for k in scenario.moved
        }
        untouched = {k: float(math.dist(final[k][:2], start[k][:2])) for k in SCENE.tidy if k not in scenario.moved}
        result.update(
            {
                "status": log.status,
                "reason": log.reason,
                "home_err_m": errors,
                "all_returned": all(e <= SUCCESS_TOL for e in errors.values()),
                "untouched_moved_m": untouched,
                "sim_time_s": log.sim_time_s,
                "cost_usd": log.to_dict()["cost_usd"],
            }
        )
        if recorder is not None:
            result["videos"] = {k: str(v) for k, v in recorder.finalize(keep_frames=args.keep_frames).items()}
        exit_code = 0
    except Exception as exc:
        traceback.print_exc()
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["wall_s"] = time.perf_counter() - t_wall
        (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"[run_episode] seed={args.seed} vlm={args.vlm} status={result.get('status')} "
            f"all_returned={result.get('all_returned')} "
            f"err_mm={ {k: round(v * 1000, 1) for k, v in result.get('home_err_m', {}).items()} } "
            f"sim={result.get('sim_time_s', 0):.1f}s wall={result['wall_s']:.0f}s -> {out}",
            flush=True,
        )
    simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
