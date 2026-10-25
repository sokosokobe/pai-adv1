"""VLM の応答（JSON）の解析と、静止した予測の除去（設計書 6.2）。"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from restorebot.core.vlm.prompt import to_pixels


@dataclass(frozen=True)
class PredictedMove:
    """VLM が答えた「動いた物体」1 個。座標は画像上の (u, v) [px] にそろえたもの。"""

    name: str
    current_location: str
    current_px: tuple[float, float]
    target_location: str
    target_px: tuple[float, float]


def normalize_name(name: str) -> str:
    """名前の比較用：全角半角（NFKC）、空白、大文字小文字の違いを無くす。"""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", name)).lower()


def _extract_json(text: str) -> str:
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("応答に JSON がありません")
    return text[start : end + 1]


def _point(value, label: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2 or not all(isinstance(v, int | float) for v in value):
        raise ValueError(f"{label} は 2 つの数値の配列である必要があります: {value!r}")
    return (float(value[0]), float(value[1]))


def parse_prediction(text: str, *, coord_format: str, width: int, height: int) -> list[PredictedMove]:
    """VLM の応答（JSON）を解析し、座標をピクセルにそろえる。形式が正しくなければ ValueError。"""
    data = json.loads(_extract_json(text))  # json.JSONDecodeError は ValueError の一種
    if not isinstance(data, dict) or not isinstance(data.get("moved_objects"), list):
        raise ValueError("moved_objects（配列）がありません")
    moves = []
    for item in data["moved_objects"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError(f"name がありません: {item!r}")
        moves.append(
            PredictedMove(
                name=item["name"],
                current_location=str(item.get("current_location", "")),
                current_px=to_pixels(_point(item.get("current_point"), "current_point"), coord_format, width, height),
                target_location=str(item.get("target_location", "")),
                target_px=to_pixels(_point(item.get("target_point"), "target_point"), coord_format, width, height),
            )
        )
    return moves


def drop_static_predictions(preds: Sequence[PredictedMove], *, min_move_px: float = 15.0) -> list[PredictedMove]:
    """今の位置と戻し先がほぼ同じ（min_move_px 未満）予測を除く。

    #10：VLM が、動いていない物体を「動いた」と挙げ、今の位置と戻し先に同じ点を答えることがあった
    （ロボットが写った画像での誤検出 11 件すべて）。動いた物体なら、この 2 点は離れているはず。
    """
    return [p for p in preds if math.dist(p.current_px, p.target_px) >= min_move_px]
