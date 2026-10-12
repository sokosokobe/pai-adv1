"""Issue #10：深度画像で「ロボットに隠れた物体」を判定する方法（方法 B）を、データセットで評価する。Isaac Sim は不要。

ロボットの領域：今の深度が、ロボットのいない「いつもの写真」の深度より 15cm 以上近い画素（`robot_mask`）。
判定は 2 つ。
    物体ごと：物体のいつもの位置（いつもの写真での中心）に、ロボットの領域がかかるか
    撮り直し：ロボットの領域が、どれかの天板の物体を置ける範囲（縁から 4cm 内側。画像上）にかかるか
正解は、vlm_scene.py の depth_check（物体の上端の中心の深度が期待より近い＝物体の上に何かがある）。

使い方::

    .venv\\Scripts\\python.exe scripts\\experiments\\occlusion_check.py outputs\\m10\\dataset_robot \\
        outputs\\m10\\dataset_robot_shoot                                  # 実際は 1 行で入力する

出力：<データセット>/occlusion_check.json、<データセット>/robot_mask/scene_XX.png（ロボットの領域を赤く重ねた画像）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # experiments.* を import するため

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.vlm_common import (  # noqa: E402
    PLACEABLE_INSET,
    SCENES,
    DownCamera,
    hidden_truth,
    mask_overlaps_rect,
    points_hidden_by_mask,
    robot_mask,
    set_scores,
    surface_image_rect,
)

RADIUS_MARGIN_PX = 5.0  # 物体の外接円の半径（ピクセル）に足す余白


def check(dataset: Path) -> dict:
    record = json.loads((dataset / "truth.json").read_text(encoding="utf-8"))
    scene = SCENES[record["scene"]]
    cam = DownCamera(**{k: record["camera"][k] for k in ("width", "height", "focal_length", "horizontal_aperture")},
                     position=tuple(record["camera"]["position"]))  # fmt: skip
    specs = {o.key: o for o in scene.objects}
    tidy_depth = np.load(dataset / "tidy_depth.npy")
    tidy_px = {k: tuple(v["pixel"]) for k, v in record["tidy_depth_check"].items()}
    # 物体の外接円の半径（いつもの位置の高さで、ピクセルに換算）
    radius = {
        k: specs[k].footprint_radius / cam.meters_per_pixel(scene.surfaces[p.surface].z + specs[k].height)
        + RADIUS_MARGIN_PX
        for k, p in scene.tidy.items()
    }
    # 撮り直しの判定は、天板のうち物体を置ける範囲（縁から PLACEABLE_INSET 内側）にロボットがかかるか
    surface_rects = [surface_image_rect(s, cam, inset=PLACEABLE_INSET) for s in scene.surfaces.values()]
    (dataset / "robot_mask").mkdir(exist_ok=True)
    rows = []
    for s in record["scenarios"]:
        current = np.load(dataset / s["image"].replace("_rgb.png", "_depth.npy"))
        mask = robot_mask(current, tidy_depth)
        actual = hidden_truth(s["depth_check"])
        predicted = {k for k in tidy_px if points_hidden_by_mask(mask, {k: tidy_px[k]}, radius_px=radius[k])}
        retake = any(mask_overlaps_rect(mask, r) for r in surface_rects)
        rows.append(
            {
                "index": s["index"],
                "robot_pixels": int(mask.sum()),
                "hidden_actual": sorted(actual),
                "hidden_predicted": sorted(predicted),
                "retake": retake,
                "moved": s["moved"],
            }
        )
        img = np.asarray(Image.open(dataset / s["image"]).convert("RGB")).copy()
        img[mask] = (0.5 * img[mask] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
        Image.fromarray(img).save(dataset / "robot_mask" / f"scene_{s['index']:02d}.png")
    obj = set_scores(
        predicted={(r["index"], k) for r in rows for k in r["hidden_predicted"]},
        actual={(r["index"], k) for r in rows for k in r["hidden_actual"]},
    )
    scenes_with_hidden = {r["index"] for r in rows if r["hidden_actual"]}
    retakes = {r["index"] for r in rows if r["retake"]}
    summary = {
        "dataset": str(dataset),
        "n": len(rows),
        "objects_hidden_actual": sum(len(r["hidden_actual"]) for r in rows),
        "scenes_with_hidden": sorted(scenes_with_hidden),
        "object_level": obj,
        "retake": {
            "scenes": sorted(retakes),
            "missed_hidden_scenes": sorted(scenes_with_hidden - retakes),  # 撮り直すべきなのに撮り直さない
            "unneeded": sorted(retakes - scenes_with_hidden),  # 隠れた物体はないのに撮り直す
        },
        "rows": rows,
    }
    (dataset / "occlusion_check.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    args = parser.parse_args()
    for d in args.datasets:
        s = check(d)
        o, r = s["object_level"], s["retake"]
        print(
            f"[occlusion] {d.name}: n={s['n']} 隠れた物体={s['objects_hidden_actual']}（組 {s['scenes_with_hidden']}）"
            f" 物体ごと tp={o['tp']} fp={o['fp']} fn={o['fn']}"
            f" 撮り直し={len(r['scenes'])} 見逃し={r['missed_hidden_scenes']} 不要={len(r['unneeded'])}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
