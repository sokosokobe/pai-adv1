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
    report.html / report.json     報告の画面と報告文（--report のとき）
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

from restorebot.core.config import M1_BASE_POSE, M2_START_POSE
from restorebot.core.navigation import robot_clearance
from restorebot.core.perception import EstimatedWorld
from restorebot.core.scene import HOME_V2, generate_scenario
from restorebot.core.types import Pose2D
from restorebot.sim.app import start_app

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE = HOME_V2
PARKED = Pose2D(6.0, 0.0, 0.0)  # いつもの写真を撮るときのロボットの置き場所（部屋の外。天井カメラに写らない）
SUCCESS_TOL = 0.03  # 成功の基準：戻し先から水平 3cm 以内（設計書 13 章）


class PerturbPlaceRobot:
    """Robot の包み：物体 key の最初の 1 回だけ、置く位置を offset ずらす（わざと戻し損ねる。#19）。"""

    def __init__(self, robot, key: str, offset: tuple[float, float, float]) -> None:
        self.robot, self.key, self.offset, self.used = robot, key, offset, False

    def place(self, position, yaw):
        if self.robot.holding() == self.key and not self.used:
            self.used = True
            position = tuple(p + d for p, d in zip(position, self.offset, strict=True))
        return self.robot.place(position, yaw)

    def __getattr__(self, name):
        return getattr(self.robot, name)


def _report(out: Path, vlm: str) -> None:
    """報告の画面を作る（失敗しても実行の結果は変えない）。"""
    try:
        from restorebot.core.report_page import make_report
        from restorebot.core.vlm.client import make_client

        path, report = make_report(out, SCENE, make_client("claude") if vlm == "claude" else None)
        print(f"[run_episode] report: {path} ({report.source})", flush=True)
    except Exception:
        traceback.print_exc()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--milestone", choices=["m1", "m2"], default="m1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vlm", choices=["truth", "claude", "replay"], default="truth")
    ap.add_argument(
        "--replay-log",
        type=Path,
        default=None,
        help="--vlm replay：応答を再生する過去の実行の log.json（API を呼ばない）",
    )
    ap.add_argument(
        "--perception",
        choices=["truth", "estimated"],
        default="truth",
        help="掴む物体の位置・向き：シミュレータの真値か、今の写真と VLM の答えから推定するか",
    )
    ap.add_argument("--record", action="store_true", help="撮影用のカメラの動画を作る")
    ap.add_argument("--keep-frames", action="store_true", help="動画にした後も連番画像を残す")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--no-verify", action="store_true", help="片付けた後の確認とやり直しをしない")
    ap.add_argument(
        "--perturb-place", default=None, metavar="KEY:DX:DY", help="物体 KEY の 1 回目の置く位置を (DX, DY) [m] ずらす"
    )
    ap.add_argument(
        "--report", action="store_true", help="報告の画面（report.html）を作る。--vlm claude なら報告文も Claude が書く"
    )
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
        from restorebot.core.vlm.client import DummyVlmClient, TruthVlmClient, make_client
        from restorebot.sim import robot as rb
        from restorebot.sim.arm import ArmPlanning
        from restorebot.sim.manipulation import ArmContext
        from restorebot.sim.robot_adapter import SimRobot
        from restorebot.sim.scene_builder import PHYSICS_DT, build_scene, set_robot_pose
        from restorebot.sim.world import SimCeilingCamera, SimWorld

        if args.milestone == "m1":  # 机の上だけ、M1 の止め位置から届く範囲
            scenario = generate_scenario(SCENE, args.seed, surfaces=("desk",), reach_from=M1_BASE_POSE)
            start_pose = M1_BASE_POSE
        else:  # 机と低い棚の両方
            scenario = generate_scenario(SCENE, args.seed)
            start_pose = M2_START_POSE
        result["placements"] = {k: [p.surface, p.x, p.y, p.yaw] for k, p in scenario.placements.items()}
        result["moved"] = list(scenario.moved)

        # ---- 準備 1：いつもの写真（ロボットは部屋の外）----
        sim = build_scene(simulation_app, SCENE, SCENE.tidy, PARKED)
        world = SimWorld(sim)
        camera = SimCeilingCamera(sim, simulation_app)
        reference = camera.capture()

        # ---- 準備 2：台車を止め位置に置き、物体を散らかす ----
        set_robot_pose(simulation_app, sim, start_pose, rb.STANDBY_ARM)
        world.set_placements(dict(scenario.placements))
        for _ in range(60):
            simulation_app.update()
        start = {k: world.object_pose(k)[0] for k in SCENE.tidy}
        ctx = ArmContext(ArmPlanning(sim.robot))
        # 物体の位置・向き：真値か、今の写真と VLM の答えから推定するか（#23）。台車の位置はどちらも関節の値
        perceived = EstimatedWorld(SCENE, world) if args.perception == "estimated" else world
        robot = SimRobot(SCENE, sim, world, ctx, objects=perceived)

        if args.vlm == "replay":  # 過去の実行の VLM の応答を順に再生する（同じ条件で再現するため）
            past = json.loads(args.replay_log.read_text(encoding="utf-8"))
            detections = [past["detection"]] + ([past["verify"]["detection"]] if past.get("verify") else [])
            texts = [a["text"] for d in detections for a in d["attempts"] if a.get("text") is not None]
            vlm = DummyVlmClient(texts, name=f"replay:{args.replay_log.parent.name}")
        elif args.vlm == "truth":  # 問い合わせのたびに、その時点の真値から答える（確認でも使える）
            vlm = TruthVlmClient(SCENE, lambda: {k: tuple(world.object_pose(k)[0]) for k in SCENE.tidy})
        else:
            vlm = make_client(args.vlm)
        if args.perturb_place:  # わざと戻し損ねる：指定の物体の 1 回目の置く位置だけをずらす（確認とやり直しの確認用）
            key, dx, dy = args.perturb_place.split(":")
            robot = PerturbPlaceRobot(robot, key, (float(dx), float(dy), 0.0))
            result["perturb_place"] = [key, float(dx), float(dy)]

        recorder = None
        if args.record:
            from restorebot.sim.recorder import Recorder

            recorder = Recorder(sim, out, world.robot_pose)

        track = {"steps": 0, "min_clearance": float("inf"), "at": None, "trajectory": [], "objects": []}

        class Stepper:
            def step(self) -> float:
                # 台車の外形と家具・壁の距離（当たっていないかの確認）と、軌跡（10Hz）
                pose = world.robot_pose()
                c = robot_clearance(pose, SCENE)
                if c < track["min_clearance"]:
                    track["min_clearance"], track["at"] = c, [pose.x, pose.y, pose.yaw]
                if track["steps"] % 6 == 0:
                    track["trajectory"].append([round(pose.x, 4), round(pose.y, 4), round(pose.yaw, 4)])
                    # 物体の位置（倒した・押したときに、いつ起きたかを調べるため）
                    track["objects"].append([round(c, 4) for k in SCENE.tidy for c in world.object_pose(k)[0]])
                track["steps"] += 1
                simulation_app.update()
                if recorder is not None:
                    recorder.on_step()
                return PHYSICS_DT

        episode = Episode(
            SCENE,
            perceived,
            camera,
            robot,
            vlm,
            Stepper(),
            EpisodeConfig(milestone=args.milestone, verify=not args.no_verify),
            meta={
                "seed": args.seed,
                "vlm": vlm.name,
                "perception": args.perception,
                "moved_truth": list(scenario.moved),
            },
            truth=world,
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
        # 箱の向きの誤差（いつもの向きとの差。180° の違いは同じとみなす。#25）
        yaw_errors = {}
        for k in scenario.moved:
            if SCENE.object(k).shape == "cube":
                d = (world.object_state(k).yaw - SCENE.tidy[k].yaw) % math.pi
                yaw_errors[k] = math.degrees(min(d, math.pi - d))
        result.update(
            {
                "status": log.status,
                "reason": log.reason,
                "home_err_m": errors,
                "all_returned": all(e <= SUCCESS_TOL for e in errors.values()),
                "home_yaw_err_deg": yaw_errors,
                "untouched_moved_m": untouched,
                "min_clearance_m": track["min_clearance"],
                "min_clearance_at": track["at"],
                "trajectory": track["trajectory"],
                "object_keys": list(SCENE.tidy),
                "object_trajectory": track["objects"],  # 10Hz。物体ごとに x, y, z を object_keys の順に並べた行
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
        if args.report and (out / "log.json").exists():
            _report(out, args.vlm)
        print(
            f"[run_episode] seed={args.seed} vlm={args.vlm} status={result.get('status')} "
            f"all_returned={result.get('all_returned')} "
            f"err_mm={ {k: round(v * 1000, 1) for k, v in result.get('home_err_m', {}).items()} } "
            f"yaw_err_deg={ {k: round(v, 1) for k, v in result.get('home_yaw_err_deg', {}).items()} } "
            f"sim={result.get('sim_time_s', 0):.1f}s wall={result['wall_s']:.0f}s -> {out}",
            flush=True,
        )
    simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
