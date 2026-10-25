"""restorebot.core.vlm.client.truth_response（真値から VLM の応答を作るダミー）の単体テスト。"""

import json

import pytest

from restorebot.core.scene import HOME_V2, Placement
from restorebot.core.vlm.client import truth_response
from restorebot.core.vlm.parse import parse_prediction


def test_truth_response_parses_back_to_projected_centers():
    cam = HOME_V2.camera
    current = {"cup": HOME_V2.object_center("cup", Placement("desk", 0.9, 0.62))}
    target = {"cup": HOME_V2.object_center("cup", HOME_V2.tidy["cup"])}
    text = truth_response(HOME_V2, current, target)
    data = json.loads(text)
    assert [m["name"] for m in data["moved_objects"]] == ["赤いコップ"]
    (p,) = parse_prediction(text, coord_format="pixel_xy", width=cam.width, height=cam.height)
    # 応答は 0.1px 単位に丸める
    assert p.current_px == pytest.approx(cam.project(current["cup"]), abs=0.05)
    assert p.target_px == pytest.approx(cam.project(target["cup"]), abs=0.05)


def test_truth_response_empty():
    assert json.loads(truth_response(HOME_V2, {}, {})) == {"moved_objects": []}
