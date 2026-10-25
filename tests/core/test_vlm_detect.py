"""restorebot.core.vlm.detect（問い合わせ・解析・再試行）の単体テスト。ダミーの VLM を使う。"""

import json

import pytest

from restorebot.core.vlm.client import DummyVlmClient, VlmApiError, VlmResult
from restorebot.core.vlm.detect import detect_moves

NAMES = ["赤いコップ", "黒いリモコン"]
GOOD = json.dumps(
    {
        "moved_objects": [
            {
                "name": "赤いコップ",
                "current_location": "机の右",
                "current_point": [800, 500],
                "target_location": "机の左",
                "target_point": [400, 480],
            },
            {  # 今の位置と戻し先がほぼ同じ（静止した予測）→ 除く
                "name": "黒いリモコン",
                "current_location": "机",
                "current_point": [600, 500],
                "target_location": "机",
                "target_point": [605, 503],
            },
        ]
    },
    ensure_ascii=False,
)


def _detect(dummy, **kw):
    return detect_moves(dummy, b"ref", b"cur", object_names=NAMES, width=1280, height=960, **kw)


def test_success_on_first_try_drops_static_predictions():
    dummy = DummyVlmClient([GOOD])
    d = _detect(dummy)
    assert d.ok
    assert [m.name for m in d.moves] == ["赤いコップ"]
    assert [m.name for m in d.dropped_static] == ["黒いリモコン"]
    assert len(d.attempts) == 1 and d.attempts[0].error is None
    assert "- 赤いコップ" in dummy.prompts[0]  # 物体名のリストをプロンプトに渡している


def test_retries_after_parse_failure():
    d = _detect(DummyVlmClient(["not json", GOOD]))
    assert d.ok and len(d.attempts) == 2
    assert d.attempts[0].error is not None and d.attempts[1].error is None


def test_retries_after_refusal():
    refused = VlmResult(text="", input_tokens=10, output_tokens=0, latency_s=0.1, model="dummy", refused=True)
    d = _detect(DummyVlmClient([refused, GOOD]))
    assert d.ok and len(d.attempts) == 2
    assert "拒否" in d.attempts[0].error


def test_gives_up_after_retries():
    d = _detect(DummyVlmClient(["bad"]), retries=2)
    assert not d.ok
    assert len(d.attempts) == 3  # 1 回目＋再試行 2 回
    assert d.moves == []


def test_api_error_is_not_retried_again():
    # API エラーの再試行は SDK の中で済んでいる（max_retries）ので、ここでは重ねて再試行しない
    d = _detect(DummyVlmClient([VlmApiError("timeout"), GOOD]))
    assert not d.ok and len(d.attempts) == 1
    assert "timeout" in d.attempts[0].error


def test_cost_and_latency_are_summed():
    a = VlmResult(text="bad", input_tokens=1, output_tokens=1, latency_s=1.0, model="m", cost_usd=0.01)
    b = VlmResult(text=GOOD, input_tokens=1, output_tokens=1, latency_s=2.0, model="m", cost_usd=0.02)
    d = _detect(DummyVlmClient([a, b]))
    assert d.cost_usd == pytest.approx(0.03)
    assert d.latency_s == pytest.approx(3.0)


def test_cost_is_none_if_any_attempt_cost_is_unknown():
    a = VlmResult(text=GOOD, input_tokens=1, output_tokens=1, latency_s=1.0, model="m", cost_usd=None)
    assert _detect(DummyVlmClient([a])).cost_usd is None
