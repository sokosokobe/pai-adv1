"""Issue #5：簡素な部屋を作り、天井カメラで「いつもの状態」と「散らかった状態」の画像のペアを作る。

部屋（床・壁・机・棚）と物体は単純な形状だけで作る（物理演算は使わず、置き場所に直接置く）。
天井カメラは真下向き（RGB＋深度）。配置と正解データは vlm_common.py で作る（シード固定）。

使い方::

    .venv\\Scripts\\python.exe scripts\\experiments\\vlm_scene.py                 # 10 組
    .venv\\Scripts\\python.exe scripts\\experiments\\vlm_scene.py --pairs 3 --seed 1
    .venv\\Scripts\\python.exe scripts\\experiments\\vlm_scene.py --difficulty small --seed 1 --out \\
        outputs\\m0_5\\dataset_small    # 実際は 1 行で入力する
    .venv\\Scripts\\python.exe scripts\\experiments\\vlm_scene.py --gui

出力（outputs/m0_5/dataset/）:
    tidy_rgb.png / tidy_depth.npy        いつもの状態（全ペアで共通）
    scene_XX_rgb.png / scene_XX_depth.npy  散らかった状態
    overlay/scene_XX.png                 正解の位置を重ねた確認用の画像
    depth_preview/*.png                  深度の確認用の画像
    truth.json                           カメラ・配置・正解データ・深度の確認結果
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # experiments.* を import するため

from experiments.arm_common import quat_from_yaw_wxyz  # noqa: E402
from experiments.vlm_common import (  # noqa: E402
    CEILING_CAMERA,
    OBJECTS,
    ORIENTATION_MARKER,
    ROOM_HALF,
    SURFACES,
    TIDY_LAYOUT,
    Placement,
    draw_overlay,
    generate_scenarios,
    object_center,
    scenario_truth,
)

# 散らかし方。easy：置き場所（面）を選び直して 40cm 以上離す。small：同じ面の上で 10〜25cm だけずらす
DIFFICULTIES = {
    "easy": {"min_shift": 0.4},
    "small": {"min_shift": 0.10, "max_shift": 0.25, "same_surface": True},
}
WALL_HEIGHT = 2.4
WALL_THICKNESS = 0.05
# 物体を動かした後、描画が落ち着くまで進めるフレーム数。RTX の時間方向の処理で前のシーンの物体の残像が残る。
# 30 では残像（いつもの画像との差が平均 2〜4 階調）が残った
WARMUP_FRAMES = 150


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "m0_5" / "dataset")
    parser.add_argument("--difficulty", choices=list(DIFFICULTIES), default="easy")
    parser.add_argument("--gui", action="store_true", help="GUI を表示する（既定は headless）")
    args = parser.parse_args()
    out_dir: Path = args.out
    (out_dir / "overlay").mkdir(parents=True, exist_ok=True)
    (out_dir / "depth_preview").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    # ------------------------------------------------------------
    # SimulationApp を最初に起動（Isaac Sim のモジュールはこの後で import）
    # ------------------------------------------------------------
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": not args.gui})
    exit_code = 1
    record: dict = {}
    try:
        import isaacsim.core.experimental.utils.app as app_utils
        import numpy as np
        from isaacsim.core.experimental.objects import Cone, Cube, Cylinder, DomeLight, Sphere
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
        from PIL import Image

        cam = CEILING_CAMERA
        hx, hy = ROOM_HALF

        # ---- 部屋：床・壁・机・棚・向きの目印 ----
        # 真下からの平行光源は床に光源の映り込み（白飛び）を作ったので、ドームライトだけにする
        DomeLight("/World/dome_light").set_intensities(1000)
        Cube(
            "/World/room/floor",
            sizes=1.0,
            scales=[[2 * hx, 2 * hy, 0.02]],
            positions=[[0, 0, -0.01]],
            colors=[[0.45, 0.45, 0.48]],
        )
        t = WALL_THICKNESS
        walls = {
            "north": ([0, hy + t / 2, WALL_HEIGHT / 2], [2 * hx + 2 * t, t, WALL_HEIGHT]),
            "south": ([0, -hy - t / 2, WALL_HEIGHT / 2], [2 * hx + 2 * t, t, WALL_HEIGHT]),
            "east": ([hx + t / 2, 0, WALL_HEIGHT / 2], [t, 2 * hy, WALL_HEIGHT]),
            "west": ([-hx - t / 2, 0, WALL_HEIGHT / 2], [t, 2 * hy, WALL_HEIGHT]),
        }
        for name, (pos, size) in walls.items():
            Cube(f"/World/room/wall_{name}", sizes=1.0, scales=[size], positions=[pos], colors=[[0.92, 0.9, 0.85]])
        furniture_colors = {"desk": [0.55, 0.36, 0.2], "shelf": [0.35, 0.25, 0.18]}
        for key, color in furniture_colors.items():
            s = SURFACES[key]
            Cube(
                f"/World/room/{key}",
                sizes=1.0,
                scales=[[s.size[0], s.size[1], s.z]],
                positions=[[s.center[0], s.center[1], s.z / 2]],
                colors=[color],
            )
        m = ORIENTATION_MARKER
        Cube(
            "/World/room/orientation_marker",
            sizes=1.0,
            scales=[[m.size[0], m.size[1], 0.005]],
            positions=[[m.center[0], m.center[1], 0.0025]],
            colors=[[0.05, 0.05, 0.05]],
        )

        # ---- 物体（いつもの状態に置く）----
        shapes = {}
        for o in OBJECTS:
            path = f"/World/objects/{o.key}"
            if o.shape == "cube":
                shapes[o.key] = Cube(path, sizes=1.0, scales=[list(o.size)], colors=[list(o.color)])
            elif o.shape == "cylinder":
                shapes[o.key] = Cylinder(path, radii=o.size[0], heights=o.size[1], axes="Z", colors=[list(o.color)])
            elif o.shape == "cone":
                shapes[o.key] = Cone(path, radii=o.size[0], heights=o.size[1], axes="Z", colors=[list(o.color)])
            elif o.shape == "sphere":
                shapes[o.key] = Sphere(path, radii=o.size[0], colors=[list(o.color)])
            else:
                raise ValueError(o.shape)

        def place_all(placements: dict[str, Placement]) -> None:
            for o in OBJECTS:
                p = placements[o.key]
                shapes[o.key].set_world_poses(
                    positions=[list(object_center(o, p))], orientations=[list(quat_from_yaw_wxyz(p.yaw))]
                )

        # ---- 天井カメラ（真下向き：カメラの -Z が視線、+Y が画像の上。回転なしでワールドの -Z を向く）----
        rtx = RtxCamera("/World/ceiling_camera", positions=[list(cam.position)], orientations=[[1.0, 0.0, 0.0, 0.0]])
        # Camera の API の単位は USD の値の 1/10（focal_length と aperture で同じなので比は変わらない）
        rtx.camera.set_focal_lengths([cam.focal_length / 10])
        rtx.camera.set_apertures(horizontal_apertures=[cam.horizontal_aperture / 10])
        rtx.camera.set_clipping_ranges(near_distances=[0.05], far_distances=[20.0])
        sensor = CameraSensor(rtx, resolution=(cam.height, cam.width), annotators=["rgb", "distance_to_image_plane"])
        focal = float(rtx.camera.get_focal_lengths().numpy()[0][0])
        h_ap, v_ap = (float(a.numpy()[0][0]) for a in rtx.camera.get_apertures())

        app_utils.play()
        for _ in range(WARMUP_FRAMES):
            simulation_app.update()

        def capture(stem: str) -> tuple[np.ndarray, np.ndarray]:
            for _ in range(WARMUP_FRAMES):
                simulation_app.update()
            for _ in range(120):
                rgb, _ = sensor.get_data("rgb")
                depth, _ = sensor.get_data("distance_to_image_plane")
                if rgb is not None and depth is not None:
                    break
                simulation_app.update()
            else:
                raise RuntimeError("カメラ画像を取得できませんでした")
            rgb_np = np.ascontiguousarray(rgb.numpy()[:, :, :3], dtype=np.uint8)
            depth_np = depth.numpy()[:, :, 0].astype(np.float32)
            Image.fromarray(rgb_np).save(out_dir / f"{stem}_rgb.png")
            np.save(out_dir / f"{stem}_depth.npy", depth_np)
            finite = np.isfinite(depth_np)
            lo, hi = float(depth_np[finite].min()), float(depth_np[finite].max())
            preview = np.zeros_like(depth_np)
            preview[finite] = (depth_np[finite] - lo) / max(hi - lo, 1e-6)
            Image.fromarray((255 * (1 - preview)).astype(np.uint8)).save(out_dir / "depth_preview" / f"{stem}.png")
            return rgb_np, depth_np

        def depth_checks(placements: dict[str, Placement], depth_np: np.ndarray) -> dict:
            """物体の上端の中心を投影した画素の深度と、「カメラの高さ − 上端の高さ」を比べる（投影と深度の確認）。

            上面が平らな物体（箱・円柱）では一致するはず。球は斜めから見ると視線が上端より手前の面に当たるので、
            測った深度が少し小さくなる。
            """
            out = {}
            for o in OBJECTS:
                x, y, z = object_center(o, placements[o.key])
                top = (x, y, z + o.height / 2)
                u, v = cam.project(top)
                measured = float(depth_np[int(round(v)), int(round(u))])
                expected = cam.position[2] - top[2]
                out[o.key] = {"pixel": [u, v], "measured_m": measured, "expected_m": expected}
            return out

        # ---- いつもの状態 ----
        place_all(TIDY_LAYOUT)
        _, tidy_depth = capture("tidy")
        record = {
            "seed": args.seed,
            "difficulty": args.difficulty,
            "warmup_frames": WARMUP_FRAMES,
            "scenario_params": DIFFICULTIES[args.difficulty],
            "camera": {
                **asdict(cam),
                "fx": cam.fx,
                "usd_focal_length": focal,
                "usd_apertures": [h_ap, v_ap],
                "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "objects": [asdict(o) for o in OBJECTS],
            "tidy": {k: asdict(p) for k, p in TIDY_LAYOUT.items()},
            "tidy_depth_check": depth_checks(TIDY_LAYOUT, tidy_depth),
            "scenarios": [],
        }

        # ---- 散らかった状態 ----
        for sc in generate_scenarios(args.pairs, seed=args.seed, **DIFFICULTIES[args.difficulty]):
            stem = f"scene_{sc.index:02d}"
            place_all(sc.placements)
            rgb_np, depth_np = capture(stem)
            truth = scenario_truth(sc, cam)
            draw_overlay(Image.fromarray(rgb_np), truth).save(out_dir / "overlay" / f"{stem}.png")
            record["scenarios"].append(
                {
                    "index": sc.index,
                    "image": f"{stem}_rgb.png",
                    "moved": list(sc.moved),
                    "placements": {k: asdict(p) for k, p in sc.placements.items()},
                    "current_px": truth.current_px,
                    "target_px": truth.target_px,
                    "depth_check": depth_checks(sc.placements, depth_np),
                }
            )
            print(f"[vlm_scene] {stem}: moved={list(sc.moved)}", flush=True)
        errs = [
            abs(c["measured_m"] - c["expected_m"])
            for s in [{"depth_check": record["tidy_depth_check"]}, *record["scenarios"]]
            for c in s["depth_check"].values()
        ]
        record["depth_check_max_abs_err_m"] = max(errs)
        record["status"] = "ok"
        exit_code = 0
    except Exception as exc:
        traceback.print_exc()
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        (out_dir / "truth.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
        )
        print(
            f"[vlm_scene] status={record.get('status')} "
            f"depth_err_max={record.get('depth_check_max_abs_err_m')} -> {out_dir}",
            flush=True,
        )
    simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
