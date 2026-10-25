"""差分検出のプロンプト、応答の JSON の形式、座標の形式（設計書 6.2）。

#5・#9 の scripts/experiments/vlm_common.py から移したもの。プロンプトの文面は #9 で検証したまま変えない。
ロボットの注意書きと hidden_objects は加えない（#10：注意書きでは誤検出を防げず、hidden_objects は使われなかった）。
"""

from __future__ import annotations

from collections.abc import Sequence

# モデルごとに得意な座標の形式が違うので、プロンプトで指定し、受け取ったらピクセルにそろえる
COORD_FORMATS = ("pixel_xy", "norm1000_xy", "norm1000_yx")


def to_pixels(point: Sequence[float], coord_format: str, width: int, height: int) -> tuple[float, float]:
    """VLM が返した点 → 画像上の座標 (u, v) [px]。"""
    a, b = float(point[0]), float(point[1])
    if coord_format == "pixel_xy":
        return (a, b)
    if coord_format == "norm1000_xy":
        return (a / 1000 * width, b / 1000 * height)
    if coord_format == "norm1000_yx":
        return (b / 1000 * width, a / 1000 * height)
    raise ValueError(f"未知の座標の形式です: {coord_format}")


def _coord_description(coord_format: str, width: int, height: int) -> str:
    if coord_format == "pixel_xy":
        return (
            f"Points are [x, y] in pixels of the {width}x{height} image: origin at the top-left corner, "
            "x increases to the right, y increases downward."
        )
    if coord_format == "norm1000_xy":
        return (
            "Points are [x, y] normalized to 0-1000: [0, 0] is the top-left corner and [1000, 1000] is the "
            "bottom-right corner of the image."
        )
    if coord_format == "norm1000_yx":
        return (
            "Points are [y, x] normalized to 0-1000: [0, 0] is the top-left corner and [1000, 1000] is the "
            "bottom-right corner of the image."
        )
    raise ValueError(f"未知の座標の形式です: {coord_format}")


RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "moved_objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "current_location": {"type": "string"},
                    "current_point": {"type": "array", "items": {"type": "number"}},
                    "target_location": {"type": "string"},
                    "target_point": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["name", "current_location", "current_point", "target_location", "target_point"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["moved_objects"],
    "additionalProperties": False,
}


def build_prompt(*, coord_format: str, width: int, height: int, object_names: Sequence[str]) -> str:
    """差分検出のプロンプト。画像は 1 枚目に「いつもの状態」、2 枚目に「今の状態」を渡す前提。"""
    names = "\n".join(f"- {n}" for n in object_names)
    name_rule = "The room contains these movable objects. Use exactly one of these names for `name`:\n" + names
    return f"""You are given two top-down images of the same room, taken by the same fixed ceiling camera.
Image 1 is the REFERENCE: the usual, tidy state of the room.
Image 2 is the CURRENT state. Some objects have been moved from their usual places.

Task: list every object whose position in the CURRENT image differs from its position in the REFERENCE image.
Do not list objects that stayed in place. Furniture (desk, shelf) and walls never move.

{name_rule}

For each moved object, report:
- current_location: where it is now (short Japanese description, e.g. 床の右下, 机の上の左端)
- current_point: the center of the object in the CURRENT image
- target_location: where it should be returned to (short Japanese description)
- target_point: the center of the object in the REFERENCE image (its usual place)

{_coord_description(coord_format, width, height)}

Respond with JSON only, in this format:
{{"moved_objects": [{{"name": "...", "current_location": "...", "current_point": [0, 0], \
"target_location": "...", "target_point": [0, 0]}}]}}
"""
