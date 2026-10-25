"""restorebot.core.vlm.prompt / parse（プロンプト、座標の形式、応答の解析）の単体テスト。API は呼ばない。"""

import json

import pytest

from restorebot.core.vlm.parse import PredictedMove, drop_static_predictions, normalize_name, parse_prediction
from restorebot.core.vlm.prompt import COORD_FORMATS, RESPONSE_SCHEMA, build_prompt, to_pixels

# ---- 座標の形式 ----


@pytest.mark.parametrize(
    ("point", "fmt", "expected"),
    [
        ((100, 200), "pixel_xy", (100.0, 200.0)),
        ((500, 250), "norm1000_xy", (640.0, 240.0)),  # 1280 × 960 の画像
        ((250, 500), "norm1000_yx", (640.0, 240.0)),  # Gemini は [y, x]
    ],
)
def test_to_pixels(point, fmt, expected):
    assert to_pixels(point, fmt, 1280, 960) == pytest.approx(expected)


def test_to_pixels_rejects_unknown_format():
    with pytest.raises(ValueError):
        to_pixels((0, 0), "polar", 10, 10)


# ---- プロンプト ----


def test_prompt_lists_object_names_and_pixel_coordinates():
    prompt = build_prompt(coord_format="pixel_xy", width=1280, height=960, object_names=["赤いコップ", "黒いリモコン"])
    assert "- 赤いコップ" in prompt and "- 黒いリモコン" in prompt
    assert "1280x960" in prompt
    assert "REFERENCE" in prompt and "CURRENT" in prompt


@pytest.mark.parametrize("fmt", COORD_FORMATS)
def test_prompt_describes_each_coord_format(fmt):
    prompt = build_prompt(coord_format=fmt, width=1280, height=960, object_names=["赤いコップ"])
    assert ("[y, x]" in prompt) == (fmt == "norm1000_yx")


def test_prompt_has_no_robot_or_hidden_notes():
    # 設計書 6.2：ロボットの注意書きと hidden_objects は加えない（#10）
    prompt = build_prompt(coord_format="pixel_xy", width=1280, height=960, object_names=["赤いコップ"])
    assert "robot" not in prompt.lower()
    assert "hidden_objects" not in prompt


def test_response_schema_requires_all_fields():
    item = RESPONSE_SCHEMA["properties"]["moved_objects"]["items"]
    assert set(item["required"]) == {"name", "current_location", "current_point", "target_location", "target_point"}


# ---- 応答の解析 ----


def _response(*moves) -> str:
    return json.dumps({"moved_objects": list(moves)}, ensure_ascii=False)


MOVE = {
    "name": "赤いコップ",
    "current_location": "机の右",
    "current_point": [800, 500],
    "target_location": "机の左",
    "target_point": [400, 480],
}


def test_parse_prediction_pixel():
    (p,) = parse_prediction(_response(MOVE), coord_format="pixel_xy", width=1280, height=960)
    assert p == PredictedMove("赤いコップ", "机の右", (800.0, 500.0), "机の左", (400.0, 480.0))


def test_parse_prediction_converts_coordinates():
    move = {**MOVE, "current_point": [500, 500], "target_point": [250, 250]}
    (p,) = parse_prediction(_response(move), coord_format="norm1000_yx", width=1000, height=1000)
    assert p.current_px == (500.0, 500.0) and p.target_px == (250.0, 250.0)


def test_parse_prediction_accepts_code_fence_and_empty_list():
    text = "```json\n" + _response() + "\n```"
    assert parse_prediction(text, coord_format="pixel_xy", width=10, height=10) == []


@pytest.mark.parametrize(
    "text",
    [
        "",  # 拒否されたときなど
        "no json here",
        '{"moved": []}',
        '{"moved_objects": [{"current_point": [1, 2], "target_point": [3, 4]}]}',  # name がない
        _response({**MOVE, "target_point": [1]}),
        _response({**MOVE, "current_point": "100,200"}),
        '{"moved_objects": [',  # 途中で切れた JSON
    ],
)
def test_parse_prediction_rejects_bad_responses(text):
    with pytest.raises(ValueError):
        parse_prediction(text, coord_format="pixel_xy", width=1280, height=960)


def test_drop_static_predictions():
    moving = PredictedMove("a", "", (0.0, 0.0), "", (20.0, 0.0))
    static = PredictedMove("b", "", (100.0, 100.0), "", (110.0, 105.0))  # 約 11px
    assert drop_static_predictions([moving, static]) == [moving]
    assert drop_static_predictions([moving, static], min_move_px=5.0) == [moving, static]


@pytest.mark.parametrize(
    ("a", "b"), [("赤い コップ", "赤いコップ"), ("ＡＢＣ", "abc"), ("黒いリモコン\n", "黒いリモコン")]
)
def test_normalize_name(a, b):
    assert normalize_name(a) == normalize_name(b)
