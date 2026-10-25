"""止め位置と経路（設計書 8 章）を上から見た図に描く（#16）。Isaac Sim は使わない。

使い方:
    .venv\\Scripts\\python.exe scripts\\plan_preview.py

出力: outputs/m2_16/routes.png
    灰色：膨張した障害物（台車の中心が入れない所）、茶色：家具、黒の枠：部屋の壁
    線：経路（色ごとに 1 本）、枠：台車の外形（出発・経由点・到着）、点：物体（掴む・置く位置）
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from restorebot.core.config import M1_BASE_POSE, NAV, ROBOT
from restorebot.core.navigation import approach_pose, build_grid, plan_route, robot_corners
from restorebot.core.scene import HOME_V2

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE = HOME_V2
SCALE = 200  # px / m
COLORS = [(220, 40, 40), (40, 120, 220), (30, 160, 60), (200, 120, 0)]


def main() -> int:
    hx, hy = SCENE.room_half
    grid = build_grid(SCENE, ROBOT.circumradius + NAV.inflation_margin, NAV.resolution)
    w, h = int(2 * hx * SCALE), int(2 * hy * SCALE)
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)

    def px(x, y):  # ワールド (x, y) → 画像（上が +y）
        return ((x + hx) * SCALE, (hy - y) * SCALE)

    rows, cols = grid.occupied.shape
    for i in range(rows):
        for j in range(cols):
            if grid.occupied[i, j]:
                x0, y0 = grid.origin[0] + j * grid.resolution, grid.origin[1] + i * grid.resolution
                d.rectangle([px(x0, y0 + grid.resolution), px(x0 + grid.resolution, y0)], fill=(215, 215, 215))
    for s in SCENE.surfaces.values():
        cx, cy = s.center
        d.rectangle(
            [px(cx - s.size[0] / 2, cy + s.size[1] / 2), px(cx + s.size[0] / 2, cy - s.size[1] / 2)],
            fill=(170, 120, 70),
        )
        d.text(px(cx - 0.15, cy), s.key, fill=(255, 255, 255))
    d.rectangle([px(-hx, hy), px(hx, -hy)], outline=(0, 0, 0), width=3)

    # 経路の例：M1 の止め位置 → 低い棚のスプレー缶 → 低い棚の小箱の戻し先 → 机の左側からコップ → M1 の止め位置
    targets = [
        ("spray", "side_table", (SCENE.tidy["spray"].x, SCENE.tidy["spray"].y)),
        ("small_box", "side_table", (SCENE.tidy["small_box"].x, SCENE.tidy["small_box"].y)),
        ("cup", "desk", (SCENE.tidy["cup"].x, SCENE.tidy["cup"].y)),
        ("pen_stand", "desk", (SCENE.tidy["pen_stand"].x, SCENE.tidy["pen_stand"].y)),
    ]
    pose = M1_BASE_POSE
    report = []
    for n, (key, surface, xy) in enumerate(targets):
        goal = approach_pose(xy, surface, SCENE, start=pose)
        route = plan_route(SCENE, grid, pose, goal) if goal is not None else None
        color = COLORS[n % len(COLORS)]
        report.append((key, goal, None if route is None else len(route)))
        if route is None:
            continue
        pts = [pose, *route]
        d.line([px(p.x, p.y) for p in pts], fill=color, width=3)
        for p in pts:
            corners = [px(*c) for c in robot_corners(p)]
            d.polygon(corners, outline=color)
        d.ellipse([px(xy[0] - 0.03, xy[1] + 0.03), px(xy[0] + 0.03, xy[1] - 0.03)], fill=color)
        d.text(px(goal.x + 0.05, goal.y + 0.05), f"{n + 1}:{key}", fill=color)
        pose = goal
    out = REPO_ROOT / "outputs" / "m2_16"
    out.mkdir(parents=True, exist_ok=True)
    img.save(out / "routes.png")
    for key, goal, n_wp in report:
        print(f"[plan_preview] {key:10s} goal={goal} waypoints={n_wp}")
    print(f"[plan_preview] -> {out / 'routes.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
