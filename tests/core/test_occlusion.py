"""restorebot.core.occlusion（深度によるロボットの領域と、撮り直しの判定）の単体テスト。"""

import numpy as np

from restorebot.core.occlusion import mask_overlaps_rect, needs_retake, robot_mask, surface_image_rect
from restorebot.core.scene import HOME_V2, Rect

CAM = HOME_V2.camera
DESK = HOME_V2.surfaces["desk"]


def _depth(value=2.6):
    return np.full((CAM.height, CAM.width), value, dtype=np.float32)


def test_robot_mask_marks_pixels_much_closer_than_reference():
    ref = _depth()
    cur = ref.copy()
    cur[10:20, 30:40] = 2.6 - 0.5  # 50cm 近い（ロボット）
    cur[50:60, 30:40] = 2.6 - 0.10  # 10cm 近い（物体程度）は含めない
    cur[0, 0] = np.nan
    mask = robot_mask(cur, ref)
    assert mask[10:20, 30:40].all()
    assert not mask[50:60, 30:40].any()
    assert mask.sum() == 100


def test_surface_image_rect_contains_projection_of_surface_center():
    r = surface_image_rect(DESK, CAM)
    u, v = CAM.project((*DESK.center, DESK.z))
    assert abs(u - r.center[0]) < 1e-6 and abs(v - r.center[1]) < 1e-6
    inner = surface_image_rect(DESK, CAM, inset=0.04)
    assert inner.size[0] < r.size[0] and inner.size[1] < r.size[1]


def test_mask_overlaps_rect():
    mask = np.zeros((100, 100), dtype=bool)
    mask[40, 40] = True
    assert mask_overlaps_rect(mask, Rect((40.5, 40.5), (2, 2)))
    assert not mask_overlaps_rect(mask, Rect((70, 70), (10, 10)))
    assert not mask_overlaps_rect(mask, Rect((-50, -50), (10, 10)))  # 画像の外


def _mask_at_world(x, y, z, half_px=3):
    mask = np.zeros((CAM.height, CAM.width), dtype=bool)
    u, v = CAM.project((x, y, z))
    mask[int(v) - half_px : int(v) + half_px, int(u) - half_px : int(u) + half_px] = True
    return mask


def test_needs_retake_when_robot_covers_desk_placeable_area():
    assert needs_retake(_mask_at_world(0.8, 0.75, DESK.z), HOME_V2)


def test_no_retake_when_robot_is_away_from_surfaces():
    assert not needs_retake(_mask_at_world(0.0, -1.0, 0.0), HOME_V2)
    assert not needs_retake(np.zeros((CAM.height, CAM.width), dtype=bool), HOME_V2)


def test_no_retake_when_robot_only_touches_the_edge_margin():
    # 天板の縁から 4cm 以内（物体を置かない所）にかかるだけなら撮り直さない
    edge_y = DESK.center[1] - DESK.size[1] / 2 + 0.01
    assert not needs_retake(_mask_at_world(0.8, edge_y, DESK.z, half_px=1), HOME_V2)
