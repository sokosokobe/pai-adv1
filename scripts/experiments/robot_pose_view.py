"""写真から推定した台車の位置（#26）を、確かめるための画像にする（Isaac Sim は起動しない）。

保存済みの実行の記録（log.json のイベント robot_pose と、写真の RGB・深度）から、写真ごとに次を重ねて描く：
    緑：推定に使った台車の上面の画素
    黄の長方形：当てはめた長方形（推定した位置と向き。前の辺を太く）
    水色の長方形：オドメトリの位置と向き（台車の上面の大きさ）

使い方:
    .venv\\Scripts\\python.exe scripts\\experiments\\robot_pose_view.py outputs\\m3_26\\truth_seed0 [ほかのフォルダ ...]
出力：<フォルダ>/robot_pose_<写真の名前>.png（台車のまわりを切り抜いて 2 倍に拡大）
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from restorebot.core.config import ROBOT
from restorebot.core.occlusion import robot_mask
from restorebot.core.scene import HOME_V2

CAM = HOME_V2.camera
TOP_Z = (ROBOT.top_z_min + ROBOT.top_z_max) / 2


def _corners(x, y, yaw, length, width):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [(length / 2, width / 2), (-length / 2, width / 2), (-length / 2, -width / 2), (length / 2, -width / 2)]
    return [CAM.project((x + c * a - s * b, y + s * a + c * b, TOP_Z)) for a, b in pts]


def _draw_box(draw, corners, color):
    for i in range(4):
        draw.line([corners[i], corners[(i + 1) % 4]], fill=color, width=6 if i == 3 else 3)  # 前の辺（i=3）を太く


def view(run_dir: Path) -> list[Path]:
    log = json.loads((run_dir / "log.json").read_text(encoding="utf-8"))
    ref = np.load(run_dir / "reference_depth.npy")
    outs = []
    for e in log["events"]:
        if e["kind"] != "robot_pose":
            continue
        name = e["photo"]
        depth = np.load(run_dir / f"{name}_depth.npy")
        rgb = np.asarray(Image.open(run_dir / f"{name}_rgb.png").convert("RGB")).copy()
        z = CAM.position[2] - depth
        sel = robot_mask(depth, ref) & (z >= ROBOT.top_z_min) & (z <= ROBOT.top_z_max)
        rgb[sel] = (0.5 * rgb[sel] + 0.5 * np.array([0, 200, 0])).astype(np.uint8)
        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img)
        ox, oy, oyaw = e["odometry"]
        _draw_box(draw, _corners(ox, oy, oyaw, ROBOT.top_length, ROBOT.top_width), (0, 200, 255))
        if e.get("estimate"):
            ex, ey, eyaw = e["estimate"]
            _draw_box(draw, _corners(ex, ey, eyaw, *e["size"]), (255, 220, 0))
            size = f"{e['size'][0]:.3f}x{e['size'][1]:.3f}"
            text = f"{name}: diff {e['diff_m'] * 100:.1f}cm {e['diff_deg']:.2f}deg size {size}"
        else:
            text = f"{name}: no estimate"
        u, v = CAM.project((ox, oy, TOP_Z))
        box = (max(int(u) - 200, 0), max(int(v) - 200, 0), min(int(u) + 200, CAM.width), min(int(v) + 200, CAM.height))
        crop = img.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.Resampling.NEAREST)
        ImageDraw.Draw(crop).text((10, 10), text, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        path = run_dir / f"robot_pose_{name}.png"
        crop.save(path)
        outs.append(path)
        print(f"[robot_pose_view] {path}  {text}")
    return outs


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        view(Path(arg))
