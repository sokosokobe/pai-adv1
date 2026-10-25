"""restorebot.core.vlm.client（Claude のクライアント、ダミー、費用）の単体テスト。

実際の API は呼ばない。anthropic のクライアントの代わりに、送った引数を記録する偽物を渡す。
"""

import base64
from types import SimpleNamespace

import pytest

from restorebot.core.vlm.client import (
    ClaudeClient,
    DummyVlmClient,
    VlmApiError,
    VlmResult,
    cost_usd,
)
from restorebot.core.vlm.prompt import RESPONSE_SCHEMA


class _FakeMessages:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def _fake_anthropic(response=None, error=None):
    beta = _FakeMessages(response, error)
    plain = _FakeMessages(response, error)
    return SimpleNamespace(beta=SimpleNamespace(messages=beta), messages=plain), beta, plain


def _resp(content, *, stop_reason="end_turn", model="claude-sonnet-5-5", inp=1000, out=500):
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        model=model,
        usage=SimpleNamespace(input_tokens=inp, output_tokens=out),
    )


def _text(t):
    return SimpleNamespace(type="text", text=t)


def test_claude_request_uses_structured_output_effort_and_fallback():
    fake, beta, plain = _fake_anthropic(_resp([_text('{"moved_objects": []}')]))
    client = ClaudeClient(client=fake)
    client.detect(b"before", b"after", "PROMPT")
    assert plain.calls == []  # フォールバックは beta の API
    (kw,) = beta.calls
    assert kw["model"] == "claude-sonnet-5-5"
    assert kw["output_config"] == {"effort": "high", "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}}
    assert kw["fallbacks"] == "default"
    assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert "temperature" not in kw and "thinking" not in kw


def test_claude_request_puts_reference_image_first():
    fake, beta, _ = _fake_anthropic(_resp([_text("{}")]))
    ClaudeClient(client=fake).detect(b"before", b"after", "PROMPT")
    content = beta.calls[0]["messages"][0]["content"]
    images = [c for c in content if c["type"] == "image"]
    assert [base64.b64decode(i["source"]["data"]) for i in images] == [b"before", b"after"]
    assert content[-1] == {"type": "text", "text": "PROMPT"}
    # ラベル → 画像 の順（1 枚目がいつもの状態）
    assert "REFERENCE" in content[0]["text"] and "CURRENT" in content[2]["text"]


def test_claude_without_fallback_uses_plain_messages_api():
    fake, beta, plain = _fake_anthropic(_resp([_text("{}")]))
    ClaudeClient(client=fake, fallbacks=False).detect(b"a", b"b", "P")
    assert beta.calls == []
    assert "fallbacks" not in plain.calls[0] and "betas" not in plain.calls[0]


def test_claude_result_has_tokens_model_and_cost():
    fake, _, _ = _fake_anthropic(_resp([_text("{}")], inp=1_000_000, out=100_000))
    r = ClaudeClient(client=fake).detect(b"a", b"b", "P")
    assert (r.text, r.input_tokens, r.output_tokens, r.model, r.refused) == (
        "{}",
        1_000_000,
        100_000,
        "claude-sonnet-5-5",
        False,
    )
    assert r.cost_usd == pytest.approx(2.0 + 1.0)  # $2 / $10 per MTok


def test_claude_refusal_returns_empty_text():
    fake, _, _ = _fake_anthropic(_resp([_text("partial")], stop_reason="refusal"))
    r = ClaudeClient(client=fake).detect(b"a", b"b", "P")
    assert r.refused and r.text == ""


def test_claude_fallback_keeps_only_text_after_last_fallback_block():
    content = [_text("declined part"), SimpleNamespace(type="fallback"), _text('{"moved_objects": []}')]
    fake, _, _ = _fake_anthropic(_resp(content, model="claude-opus-5-5"))
    r = ClaudeClient(client=fake).detect(b"a", b"b", "P")
    assert r.text == '{"moved_objects": []}'
    assert r.model == "claude-opus-5-5"  # 答えたモデルの単価で費用を計算する
    assert r.cost_usd == pytest.approx(cost_usd(1000, 500, "claude-opus-5-5"))


def test_claude_api_error_is_wrapped():
    anthropic = pytest.importorskip("anthropic")
    error = anthropic.APIConnectionError(request=SimpleNamespace(method="POST", url="x"))
    fake, _, _ = _fake_anthropic(error=error)
    with pytest.raises(VlmApiError):
        ClaudeClient(client=fake).detect(b"a", b"b", "P")


def test_cost_usd_unknown_model_is_none():
    assert cost_usd(10, 10, "some-other-model") is None


# ---- ダミー ----


def test_dummy_returns_responses_in_order_and_repeats_last():
    dummy = DummyVlmClient(["first", "second"])
    texts = [dummy.detect(b"a", b"b", f"p{i}").text for i in range(3)]
    assert texts == ["first", "second", "second"]
    assert dummy.prompts == ["p0", "p1", "p2"]


def test_dummy_can_return_results_and_raise():
    refused = VlmResult(text="", input_tokens=0, output_tokens=0, latency_s=0.0, model="dummy", refused=True)
    dummy = DummyVlmClient([refused, VlmApiError("boom")])
    assert dummy.detect(b"a", b"b", "p").refused
    with pytest.raises(VlmApiError):
        dummy.detect(b"a", b"b", "p")
