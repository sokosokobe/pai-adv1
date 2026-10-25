"""VLM への問い合わせから、動いた物体の予測までの一連の処理と再試行（設計書 2.3、6 章）。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from restorebot.core.vlm.client import VlmApiError, VlmClient, VlmResult
from restorebot.core.vlm.parse import PredictedMove, drop_static_predictions, parse_prediction
from restorebot.core.vlm.prompt import build_prompt


@dataclass(frozen=True)
class Attempt:
    """1 回の問い合わせ。result は応答（API エラーのときは None）、error は失敗の理由（成功なら None）。"""

    result: VlmResult | None
    error: str | None


@dataclass(frozen=True)
class Detection:
    ok: bool
    moves: list[PredictedMove] = field(default_factory=list)  # 静止した予測を除いた後
    dropped_static: list[PredictedMove] = field(default_factory=list)
    attempts: tuple[Attempt, ...] = ()
    prompt: str = ""

    @property
    def cost_usd(self) -> float | None:
        """すべての問い合わせの料金の合計。1 回でも料金が分からなければ None。"""
        costs = [a.result.cost_usd for a in self.attempts if a.result is not None]
        return None if any(c is None for c in costs) else sum(costs)

    @property
    def latency_s(self) -> float:
        return sum(a.result.latency_s for a in self.attempts if a.result is not None)


def detect_moves(
    client: VlmClient,
    reference_png: bytes,
    current_png: bytes,
    *,
    object_names: Sequence[str],
    width: int,
    height: int,
    retries: int = 2,
    min_move_px: float = 15.0,
) -> Detection:
    """いつもの写真と今の写真を VLM で比べ、動いた物体の予測を返す。

    解析の失敗と拒否は retries 回まで問い合わせ直す。API エラーは SDK の中で再試行済みなので、ここでは重ねず失敗にする。
    """
    prompt = build_prompt(coord_format=client.coord_format, width=width, height=height, object_names=object_names)
    attempts: list[Attempt] = []
    for _ in range(retries + 1):
        try:
            result = client.detect(reference_png, current_png, prompt)
        except VlmApiError as e:
            attempts.append(Attempt(result=None, error=f"API エラー: {e}"))
            break
        if result.refused:
            attempts.append(Attempt(result=result, error="拒否された（refusal）"))
            continue
        try:
            preds = parse_prediction(result.text, coord_format=client.coord_format, width=width, height=height)
        except ValueError as e:
            attempts.append(Attempt(result=result, error=f"解析の失敗: {e}"))
            continue
        attempts.append(Attempt(result=result, error=None))
        moves = drop_static_predictions(preds, min_move_px=min_move_px)
        dropped = [p for p in preds if p not in moves]
        return Detection(ok=True, moves=moves, dropped_static=dropped, attempts=tuple(attempts), prompt=prompt)
    return Detection(ok=False, attempts=tuple(attempts), prompt=prompt)
