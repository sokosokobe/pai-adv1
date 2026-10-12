"""Issue #5 VLM のクライアント（VlmClient の実装）。実際の API・ローカルサーバーを呼ぶので単体テストでは使わない。

- GeminiClient：Gemini Robotics-ER 2（google-genai）。座標は [y, x] を 0〜1000 に正規化（モデル本来の形式）
- ClaudeClient：Claude Sonnet 5.5（anthropic）。構造化出力で JSON を受け取る。座標はピクセル
- QwenOllamaClient：Qwen3-VL 8B（Ollama、ローカル）。座標は [x, y] を 0〜1000 に正規化

API キーはリポジトリ直下の .env から読み込む（GEMINI_API_KEY / ANTHROPIC_API_KEY）。
料金は 2026-10 時点の公式の値（100万トークンあたりの USD）。
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from experiments.vlm_common import RESPONSE_SCHEMA, VlmResult

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

REFERENCE_LABEL = "Image 1 (REFERENCE, usual state):"
CURRENT_LABEL = "Image 2 (CURRENT state):"


class GeminiClient:
    coord_format = "norm1000_yx"
    # 有料枠の料金（2026-12-31 まで）。出力は思考のトークンを含む。無料枠もある
    price_in_per_mtok = 1.0
    price_out_per_mtok = 5.0

    def __init__(self, model: str = "gemini-robotics-er-2-preview", thinking_level: str = "medium") -> None:
        from google import genai

        self.model = model
        self.thinking_level = thinking_level  # 公式の推奨：遅延と性能の釣り合いには medium
        self.name = f"gemini-robotics-er-2_{thinking_level}"
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def detect(self, before_png: bytes, after_png: bytes, prompt: str) -> VlmResult:
        from google.genai import types

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
        resp = self._client.models.generate_content(model=self.model, contents=contents, config=config)
        latency = time.perf_counter() - t0
        um = resp.usage_metadata
        return VlmResult(
            text=resp.text or "",
            input_tokens=um.prompt_token_count or 0,
            output_tokens=(um.candidates_token_count or 0) + (um.thoughts_token_count or 0),
            latency_s=latency,
        )


class ClaudeClient:
    coord_format = "pixel_xy"
    price_in_per_mtok = 2.0
    price_out_per_mtok = 10.0

    def __init__(self, model: str = "claude-sonnet-5-5", effort: str = "high") -> None:
        import anthropic

        self.model = model
        self.effort = effort  # Sonnet 5.5 の既定は high。思考は adaptive（既定）
        self.name = f"claude-sonnet-5-5_{effort}"
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def detect(self, before_png: bytes, after_png: bytes, prompt: str) -> VlmResult:
        def image(data: bytes) -> dict:
            b64 = base64.standard_b64encode(data).decode("ascii")
            return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}}

        content = [
            {"type": "text", "text": REFERENCE_LABEL},
            image(before_png),
            {"type": "text", "text": CURRENT_LABEL},
            image(after_png),
            {"type": "text", "text": prompt},
        ]
        t0 = time.perf_counter()
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=16000,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
        )
        latency = time.perf_counter() - t0
        # 拒否された場合（stop_reason == "refusal"）は空文字を返し、解析の失敗として数える
        refused = resp.stop_reason == "refusal"
        text = "" if refused else next((b.text for b in resp.content if b.type == "text"), "")
        return VlmResult(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_s=latency,
        )


class QwenOllamaClient:
    """Qwen3-VL 8B を Ollama（ローカル、署名付きの実行ファイル）の API で呼ぶ。

    llama.cpp の配布バイナリは、Windows のスマート アプリ コントロールに阻止されて起動できなかったので、Ollama を使う。
    """

    coord_format = "norm1000_xy"  # Qwen3-VL は 0〜1000 の相対座標で位置を答える
    price_in_per_mtok = 0.0
    price_out_per_mtok = 0.0

    def __init__(
        self,
        model: str = "qwen3-vl:8b-instruct-q8_0",
        base_url: str = "http://127.0.0.1:11434",
        name: str = "qwen3-vl-8b_q8_0",
        seed: int = 0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.seed = seed

    def detect(self, before_png: bytes, after_png: bytes, prompt: str) -> VlmResult:
        def b64(data: bytes) -> str:
            return base64.standard_b64encode(data).decode("ascii")

        payload = {
            "model": self.model,
            # 画像はメッセージごとに付くので、ラベルと画像を 1 枚ずつ別のメッセージにして順番を保つ
            "messages": [
                {"role": "user", "content": REFERENCE_LABEL, "images": [b64(before_png)]},
                {"role": "user", "content": CURRENT_LABEL, "images": [b64(after_png)]},
                {"role": "user", "content": prompt},
            ],
            "format": RESPONSE_SCHEMA,  # 構造化出力（Claude と同じ JSON スキーマ）
            "stream": False,
            # Qwen3-VL Instruct の推奨の設定（モデルカード）。シードを固定して再現できるようにする
            "options": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "presence_penalty": 1.5,
                "seed": self.seed,
                "num_ctx": 8192,
                "num_predict": 2048,
            },
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=600) as r:
            body = json.load(r)
        latency = time.perf_counter() - t0
        return VlmResult(
            text=body["message"]["content"] or "",
            input_tokens=int(body.get("prompt_eval_count", 0)),
            output_tokens=int(body.get("eval_count", 0)),
            latency_s=latency,
        )


def make_client(key: str):
    """コマンドラインの名前 → クライアント。"""
    if key == "gemini":
        return GeminiClient()
    if key == "claude":
        return ClaudeClient()
    if key == "qwen":
        return QwenOllamaClient()
    raise ValueError(f"未知のモデルです: {key}")
