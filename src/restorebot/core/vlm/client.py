"""VLM のクライアント（設計書 6.1）。

- ClaudeClient：Claude Sonnet 5.5（anthropic 0.x）。構造化出力で JSON を受け取る。座標はピクセル
- GeminiClient：Gemini Robotics-ER 2（google-genai）。座標は [y, x] を 0〜1000
  （代わりの候補。#12 では動作を確認していない）
- DummyVlmClient：単体テスト用。決まった応答を順に返す

API キーはリポジトリ直下の .env から読み込む（make_client）。単体テストでは実際の API を呼ばない。
"""

from __future__ import annotations

import base64
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from restorebot.core.vlm.prompt import RESPONSE_SCHEMA

REFERENCE_LABEL = "Image 1 (REFERENCE, usual state):"
CURRENT_LABEL = "Image 2 (CURRENT state):"

# 100 万トークンあたりの料金（入力, 出力）[USD]。2026-10 時点の公式の値
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-fable-5-1": (10.0, 50.0),
    "gemini-robotics-er-2-preview": (1.0, 5.0),  # 有料枠（2026-12-31 まで）
}


def cost_usd(input_tokens: int, output_tokens: int, model: str) -> float | None:
    """料金 [USD]。料金の分からないモデルは None。"""
    if model not in PRICES_PER_MTOK:
        return None
    price_in, price_out = PRICES_PER_MTOK[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


class VlmApiError(Exception):
    """API の呼び出しの失敗（SDK の再試行の後）。"""


@dataclass(frozen=True)
class VlmResult:
    text: str
    input_tokens: int
    output_tokens: int  # 思考のトークンを含む（課金される出力の合計）
    latency_s: float
    model: str  # 実際に答えたモデル（フォールバックで替わることがある）
    refused: bool = False  # 安全のための分類器に断られた（text は空）
    cost_usd: float | None = None


class VlmClient(Protocol):
    name: str
    coord_format: str

    def detect(self, before_png: bytes, after_png: bytes, prompt: str) -> VlmResult:
        """1 枚目に「いつもの状態」、2 枚目に「今の状態」の画像を渡して差分を問い合わせる。"""
        ...


class ClaudeClient:
    coord_format = "pixel_xy"

    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        effort: str = "high",
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
        fallbacks: bool = True,
        client=None,
    ) -> None:
        """client は anthropic.Anthropic の代わり（テスト用）。省略すると環境変数の API キーで作る。

        API エラー（接続の失敗、429、5xx）の再試行は SDK に任せる（max_retries 回）。
        fallbacks=True のとき、安全のための分類器に断られたら、サーバー側で別のモデルに回して続ける（beta）。
        """
        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks
        self.name = f"{model}_{effort}"
        if client is None:
            import anthropic

            client = anthropic.Anthropic(timeout=timeout, max_retries=max_retries)
        self._client = client

    def detect(self, before_png: bytes, after_png: bytes, prompt: str) -> VlmResult:
        import anthropic

        def image(data: bytes) -> dict:
            b64 = base64.standard_b64encode(data).decode("ascii")
            return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}

        kwargs = {
            "model": self.model,
            "max_tokens": 16000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": REFERENCE_LABEL},
                        image(before_png),
                        {"type": "text", "text": CURRENT_LABEL},
                        image(after_png),
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
        }
        t0 = time.perf_counter()
        try:
            if self.fallbacks:
                resp = self._client.beta.messages.create(
                    **kwargs, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
                )
            else:
                resp = self._client.messages.create(**kwargs)
        except anthropic.APIError as e:
            raise VlmApiError(f"{type(e).__name__}: {e}") from e
        latency = time.perf_counter() - t0
        refused = resp.stop_reason == "refusal"
        # フォールバックがあった場合、断ったモデルの途中までの文章は fallback ブロックの前にある。
        # 最後の fallback ブロックより後だけを使う
        texts: list[str] = []
        for block in resp.content:
            if block.type == "fallback":
                texts = []
            elif block.type == "text":
                texts.append(block.text)
        return VlmResult(
            text="" if refused else "".join(texts),
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_s=latency,
            model=resp.model,
            refused=refused,
            cost_usd=cost_usd(resp.usage.input_tokens, resp.usage.output_tokens, resp.model),
        )


class GeminiClient:
    coord_format = "norm1000_yx"

    def __init__(self, model: str = "gemini-robotics-er-2-preview", thinking_level: str = "medium") -> None:
        from google import genai

        self.model = model
        self.thinking_level = thinking_level  # 公式の推奨：遅延と性能の釣り合いには medium
        self.name = f"gemini-robotics-er-2_{thinking_level}"
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def detect(self, before_png: bytes, after_png: bytes, prompt: str) -> VlmResult:
        from google.genai import errors, types

        contents = [
            REFERENCE_LABEL,
            types.Part.from_bytes(data=before_png, mime_type="image/png"),
            CURRENT_LABEL,
            types.Part.from_bytes(data=after_png, mime_type="image/png"),
            prompt,
        ]
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
            response_mime_type="application/json",
        )
        t0 = time.perf_counter()
        try:
            resp = self._client.models.generate_content(model=self.model, contents=contents, config=config)
        except errors.APIError as e:
            raise VlmApiError(f"{type(e).__name__}: {e}") from e
        latency = time.perf_counter() - t0
        um = resp.usage_metadata
        inp = um.prompt_token_count or 0
        out = (um.candidates_token_count or 0) + (um.thoughts_token_count or 0)
        return VlmResult(
            text=resp.text or "",
            input_tokens=inp,
            output_tokens=out,
            latency_s=latency,
            model=self.model,
            cost_usd=cost_usd(inp, out, self.model),
        )


class DummyVlmClient:
    """単体テスト用。responses を順に返し（最後のものは繰り返す）、受け取ったプロンプトを記録する。

    要素が str なら応答の文章、VlmResult ならそのまま返し、例外なら送出する。
    """

    coord_format = "pixel_xy"

    def __init__(self, responses: str | Sequence[str | VlmResult | Exception], *, name: str = "dummy") -> None:
        self.responses = [responses] if isinstance(responses, str) else list(responses)
        if not self.responses:
            raise ValueError("responses が空です")
        self.name = name
        self.prompts: list[str] = []

    def detect(self, before_png: bytes, after_png: bytes, prompt: str) -> VlmResult:
        item = self.responses[min(len(self.prompts), len(self.responses) - 1)]
        self.prompts.append(prompt)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, VlmResult):
            return item
        return VlmResult(text=item, input_tokens=0, output_tokens=0, latency_s=0.0, model=self.name, cost_usd=0.0)


def make_client(key: str) -> VlmClient:
    """名前 → クライアント。API キーはカレントディレクトリから上へたどって見つけた .env から読み込む。"""
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
    if key == "claude":
        return ClaudeClient()
    if key == "gemini":
        return GeminiClient()
    raise ValueError(f"未知の VLM です: {key}")
