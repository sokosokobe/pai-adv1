"""VLM の予測 → ロボットの目標 → 運ぶタスクの列（設計書 6.3、2.1）。"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from restorebot.core.camera import DownCamera
from restorebot.core.scene import SceneSpec
from restorebot.core.types import MoveInstruction, Observation, Pose2D, TidyTask, WorldState
from restorebot.core.vlm.parse import PredictedMove, normalize_name


def _depth_at(depth: np.ndarray, uv: Sequence[float], window: int) -> float | None:
    """画素 uv のまわり window × window の深度の中央値（有限の値だけ）。なければ None。"""
    u, v = int(math.floor(uv[0])), int(math.floor(uv[1]))
    h = window // 2
    patch = depth[max(v - h, 0) : v + h + 1, max(u - h, 0) : u + h + 1]
    values = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(values)) if values.size else None


def to_instructions(
    preds: Sequence[PredictedMove],
    scene: SceneSpec,
    reference: Observation,
    camera: DownCamera | None = None,
    *,
    window: int = 5,
) -> tuple[list[MoveInstruction], list[str]]:
    """VLM の予測 → 物体と 3 次元の目標。2 つ目の戻り値は捨てた予測と理由（記録用）。

    1. 名前 → 物体：物体の表示名との一致（表記の揺れは無視）。一覧にない名前・2 回目以降の同じ名前は捨てる
    2. 戻し先：VLM の戻し先の画素は、物体の中心（高さの半分）を投影した点に当たる
       （#9 の正解の画素はこの点で、VLM の点との差は 0.4〜1.0px。#12 で確認）。
       まず、いつもの写真の深度（window × window の中央値）で逆投影し、その (x, y) が乗る天板を求める。
       次に、物体の中心の高さ「天板の高さ ＋ 物体の高さ / 2」で逆投影し直して、戻し先 (x, y, z) とする。
       深度の画素は物体の上面・側面・天板のどれにも当たりうるので、深度は天板を決めるためだけに使う。
       画像の外・天板の外・作業範囲の外は捨てる
    """
    camera = camera or scene.camera
    by_name = {normalize_name(o.name): o.key for o in scene.objects}
    instructions: list[MoveInstruction] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for p in preds:
        key = by_name.get(normalize_name(p.name))
        if key is None:
            dropped.append(f"{p.name}: 物体の一覧にない名前")
            continue
        if key in seen:
            dropped.append(f"{p.name}: 同じ物体の 2 回目以降の予測")
            continue
        seen.add(key)
        u, v = p.target_px
        if not (0 <= u < camera.width and 0 <= v < camera.height):
            dropped.append(f"{p.name}: 戻し先 {p.target_px} が画像の外")
            continue
        depth = _depth_at(reference.depth, p.target_px, window)
        if depth is None:
            dropped.append(f"{p.name}: 戻し先 {p.target_px} の深度が無効")
            continue
        x, y, _ = camera.unproject(p.target_px, depth)
        surface = scene.surface_at((x, y))
        if surface is None:
            dropped.append(f"{p.name}: 戻し先 ({x:.3f}, {y:.3f}) が天板の上にない")
            continue
        spec = scene.object(key)
        z = surface.z + spec.height / 2
        x, y, _ = camera.unproject(p.target_px, camera.position[2] - z)
        if not surface.in_work_area((x, y), spec.footprint_radius):
            dropped.append(f"{p.name}: 戻し先 ({x:.3f}, {y:.3f}) が {surface.key} の作業範囲の外")
            continue
        instructions.append(
            MoveInstruction(
                key=key,
                current_px=p.current_px,
                target_px=p.target_px,
                target_position=(x, y, z),
                target_surface=surface.key,
            )
        )
    return instructions, dropped


def plan_tasks(
    instructions: Sequence[MoveInstruction], world: WorldState, scene: SceneSpec, start: Pose2D
) -> list[TidyTask]:
    """掴む物体の状態（真値）を加え、今いる位置から近い順に並べる（設計書 2.1）。

    1 個目は台車の位置 start から、2 個目からは前のタスクで置く位置から、
    掴む物体までの水平距離が最も短いものを選ぶ（設計書では 2 個目からの基準は「前のタスクの止め位置」。
    止め位置の計算は #16 で作るので、それまでは置く位置で代用する）。
    物体の向きは戻さないので、置くときの向き place_yaw は掴んだときの物体の向きのまま。
    """
    remaining = [(ins, world.object_state(ins.key)) for ins in instructions]
    here = (start.x, start.y)
    tasks: list[TidyTask] = []
    while remaining:
        i = min(range(len(remaining)), key=lambda j: math.dist(here, remaining[j][1].position[:2]))
        ins, state = remaining.pop(i)
        tasks.append(
            TidyTask(
                key=ins.key,
                pick=state,
                place_position=ins.target_position,
                place_yaw=state.yaw,
                place_surface=ins.target_surface,
            )
        )
        here = ins.target_position[:2]
    return tasks
