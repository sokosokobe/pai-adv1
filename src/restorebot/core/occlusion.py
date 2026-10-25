"""ロボットに隠れた物体の判定：深度によるロボットの領域と、撮り直しの判定（設計書 7.2）。

#10 の scripts/experiments/vlm_common.py から移した。撮影位置の計算（choose_shoot_pose）は #16 で足す。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from restorebot.core.camera import DownCamera
from restorebot.core.config import PLACE_MARGIN
from restorebot.core.scene import Rect, SceneSpec, Surface

# 物体を置ける範囲：天板の縁から PLACE_MARGIN ＋ 最も細い物体の半径（2cm）だけ内側。撮り直しの判定に使う
PLACEABLE_INSET = PLACE_MARGIN + 0.02


def robot_mask(current_depth: np.ndarray, reference_depth: np.ndarray, *, margin: float = 0.15) -> np.ndarray:
    """今の深度が、ロボットのいない「いつもの写真」の深度より margin [m] 以上近い画素（＝ロボットがいる所）。

    片付けの対象物は高さ 10cm 以下なので、margin を 15cm にすれば物体は含まれない。
    """
    valid = np.isfinite(current_depth) & np.isfinite(reference_depth)
    return valid & (current_depth < reference_depth - margin)


def _bbox_rect(points: Sequence[tuple[float, float]]) -> Rect:
    us, vs = [p[0] for p in points], [p[1] for p in points]
    return Rect(center=((min(us) + max(us)) / 2, (min(vs) + max(vs)) / 2), size=(max(us) - min(us), max(vs) - min(vs)))


def surface_image_rect(surface: Surface, camera: DownCamera, *, inset: float = 0.0) -> Rect:
    """家具の天板が天井カメラの画像に写る範囲 [px]。inset [m] を与えると、天板の縁から内側へ縮めた範囲。"""
    hx, hy = surface.size[0] / 2 - inset, surface.size[1] / 2 - inset
    corners = [
        (surface.center[0] + sx * hx, surface.center[1] + sy * hy, surface.z) for sx in (-1, 1) for sy in (-1, 1)
    ]
    return _bbox_rect([camera.project(c) for c in corners])


def mask_overlaps_rect(mask: np.ndarray, rect: Rect) -> bool:
    """画像上の長方形 [px] の中に、ロボットの領域が 1 画素でもあるか。"""
    h, w = mask.shape
    j0 = max(int(math.floor(rect.center[0] - rect.size[0] / 2)), 0)
    j1 = min(int(math.ceil(rect.center[0] + rect.size[0] / 2)), w)
    i0 = max(int(math.floor(rect.center[1] - rect.size[1] / 2)), 0)
    i1 = min(int(math.ceil(rect.center[1] + rect.size[1] / 2)), h)
    return j0 < j1 and i0 < i1 and bool(mask[i0:i1, j0:j1].any())


def needs_retake(
    mask: np.ndarray, scene: SceneSpec, camera: DownCamera | None = None, *, inset: float = PLACEABLE_INSET
) -> bool:
    """ロボットの領域が、どれかの天板の物体を置ける範囲（縁から inset 内側）にかかるか（＝撮り直す）。

    #10：50 組で見逃し 0、撮影位置での不要な撮り直し 0。
    """
    camera = camera or scene.camera
    return any(mask_overlaps_rect(mask, surface_image_rect(s, camera, inset=inset)) for s in scene.surfaces.values())
