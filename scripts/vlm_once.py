"""保存済みの天井カメラの画像で VLM を 1 回呼び、応答 → 目標 → タスクの列までを確かめる（#12）。

Isaac Sim は使わない。実際の API を呼ぶので費用がかかる（Claude Sonnet 5.5 で 1 回約 $0.01）。

使い方:
    .venv\\Scripts\\python.exe scripts\\vlm_once.py --dataset outputs\\m9\\dataset --scenario 0

入力: <dataset>/tidy_rgb.png・tidy_depth.npy（いつもの写真）、scene_XX_rgb.png（今の写真）、truth.json（正解）
出力: <out>/scene_XX.json（プロンプト、応答の全文、予測、目標、捨てた予測、タスク、費用、正解との誤差）
      <out>/scene_XX_overlay.png（今の写真に、予測した今の位置 = 橙、戻し先 = 水色、正解の戻し先 = 緑の円）
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from restorebot.core.scene import HOME_V2, Placement
from restorebot.core.tasks import plan_tasks, to_instructions
from restorebot.core.types import ObjectState, Observation, Pose2D
from restorebot.core.vlm.client import make_client
from restorebot.core.vlm.detect import detect_moves

REPO_ROOT = Path(__file__).resolve().parents[1]


class TruthWorld:
    """truth.json の今の配置を真値として返す WorldState。"""

    def __init__(self, placements: dict[str, dict]):
        self.placements = {k: Placement(v["surface"], v["x"], v["y"], v.get("yaw", 0.0)) for k, v in placements.items()}

    def object_state(self, key: str) -> ObjectState:
        p = self.placements[key]
        return ObjectState(key, HOME_V2.object_center(key, p), p.yaw, p.surface)

    def robot_pose(self) -> Pose2D:
        return Pose2D(0.0, 0.0, 0.0)


def _draw_overlay(image: Image.Image, detection, truth_target_px: dict[str, tuple[float, float]]) -> Image.Image:
    out = image.convert("RGB").copy()
    d = ImageDraw.Draw(out)

    def cross(px, r, color):
        d.line([px[0] - r, px[1] - r, px[0] + r, px[1] + r], fill=color, width=3)
        d.line([px[0] - r, px[1] + r, px[0] + r, px[1] - r], fill=color, width=3)

    for px in truth_target_px.values():
        d.ellipse([px[0] - 14, px[1] - 14, px[0] + 14, px[1] + 14], outline=(40, 220, 40), width=3)
    for p in detection.moves:
        d.line([*p.current_px, *p.target_px], fill=(0, 200, 255), width=1)
        cross(p.current_px, 9, (255, 150, 0))
        cross(p.target_px, 9, (0, 200, 255))
        # PIL の既定のフォントは日本語を描けないので、物体の key（英字）を書く
        key = next((o.key for o in HOME_V2.objects if o.name == p.name), "?")
        d.text((p.current_px[0] + 12, p.current_px[1] - 8), key, fill=(255, 255, 255))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=REPO_ROOT / "outputs" / "m9" / "dataset")
    ap.add_argument("--scenario", type=int, default=0)
    ap.add_argument("--vlm", default="claude", choices=["claude", "gemini"])
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "m1_12")
    args = ap.parse_args()

    truth = json.loads((args.dataset / "truth.json").read_text(encoding="utf-8"))
    if truth["scene"] != HOME_V2.key:
        raise SystemExit(f"場面が違います: {truth['scene']}")
    scenario = next(s for s in truth["scenarios"] if s["index"] == args.scenario)
    reference_png = (args.dataset / "tidy_rgb.png").read_bytes()
    current_png = (args.dataset / scenario["image"]).read_bytes()
    depth = np.load(args.dataset / "tidy_depth.npy")
    cam = HOME_V2.camera
    if depth.shape != (cam.height, cam.width):
        raise SystemExit(f"深度の大きさが天井カメラと違います: {depth.shape}")
    reference = Observation(rgb=np.asarray(Image.open(args.dataset / "tidy_rgb.png")), depth=depth, time_s=0.0)

    client = make_client(args.vlm)
    detection = detect_moves(
        client,
        reference_png,
        current_png,
        object_names=[o.name for o in HOME_V2.objects],
        width=cam.width,
        height=cam.height,
    )
    instructions, dropped = to_instructions(detection.moves, HOME_V2, reference) if detection.ok else ([], [])
    world = TruthWorld(scenario["placements"])
    tasks = plan_tasks(instructions, world, HOME_V2, Pose2D(0.0, 0.0, 0.0))

    # 正解との誤差：#9 の「いつもの配置」の物体の中心と、逆投影で求めた戻し先
    tidy = {k: Placement(v["surface"], v["x"], v["y"], v.get("yaw", 0.0)) for k, v in truth["tidy"].items()}
    errors = {
        ins.key: math.dist(ins.target_position, HOME_V2.object_center(ins.key, tidy[ins.key])) for ins in instructions
    }
    record = {
        "dataset": str(args.dataset),
        "scenario": args.scenario,
        "moved_truth": scenario["moved"],
        "vlm": client.name,
        "ok": detection.ok,
        "prompt": detection.prompt,
        "attempts": [
            {"error": a.error, **(dataclasses.asdict(a.result) if a.result is not None else {})}
            for a in detection.attempts
        ],
        "moves": [dataclasses.asdict(m) for m in detection.moves],
        "dropped_static": [dataclasses.asdict(m) for m in detection.dropped_static],
        "instructions": [dataclasses.asdict(i) for i in instructions],
        "dropped": dropped,
        "tasks": [dataclasses.asdict(t) for t in tasks],
        "target_error_m": errors,
        "cost_usd": detection.cost_usd,
        "latency_s": detection.latency_s,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stem = Path(scenario["image"]).stem.removesuffix("_rgb")
    (args.out / f"{stem}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    truth_px = {k: tuple(v) for k, v in scenario["target_px"].items()}
    _draw_overlay(Image.open(args.dataset / scenario["image"]), detection, truth_px).save(
        args.out / f"{stem}_overlay.png"
    )

    print(f"[vlm_once] {stem}: ok={detection.ok} attempts={len(detection.attempts)} cost=${detection.cost_usd}")
    print(f"[vlm_once]   正解の動いた物体: {scenario['moved']}")
    print(f"[vlm_once]   予測: {[m.name for m in detection.moves]}（静止で除いた: {len(detection.dropped_static)}）")
    print(f"[vlm_once]   捨てた予測: {dropped}")
    for t in tasks:
        print(
            f"[vlm_once]   タスク {t.key}: 置く位置 {tuple(round(c, 3) for c in t.place_position)}"
            f"（正解との差 {errors[t.key] * 100:.1f}cm）"
        )
    print(f"[vlm_once]   -> {args.out / (stem + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
