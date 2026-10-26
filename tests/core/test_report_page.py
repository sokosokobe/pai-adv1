"""restorebot.core.report_page（報告の HTML）の単体テスト。"""

import base64
import io
import json
import re

import numpy as np
import pytest
from PIL import Image

from restorebot.core.report_page import (
    PALETTE,
    Mark,
    build_html,
    draw_marks,
    load_run,
    total_cost_usd,
)
from restorebot.core.scene import HOME_V2
from restorebot.core.vlm.summary import ReportText
from tests.core.test_vlm_summary import CUP, partial_log

SCENE = HOME_V2


def _photo(value=128, w=1280, h=960):
    return np.full((h, w, 3), value, dtype=np.uint8)


def _photos(names=("reference", "current", "after")):
    return {n: _photo() for n in names}


def test_draw_marks_arrow_and_rings():
    rgb = _photo()
    out = draw_marks(rgb, [Mark(1, (200.0, 500.0), (800.0, 500.0))])
    assert (rgb == 128).all()  # 元の画像は変えない
    color = PALETTE[0]
    assert tuple(out[500, 500]) == color  # 矢印の線（今の位置 → 戻し先）
    assert tuple(out[500, 800 - 40]) == color or tuple(out[500 - 40, 800]) == color  # 戻し先の輪
    assert (out[100, 100] == 128).all()  # 離れた所は変わらない


def test_draw_marks_target_only():
    out = draw_marks(_photo(), [Mark(2, None, (640.0, 480.0))])
    assert (out[480, 300] == 128).all()  # 矢印はない
    assert (out != 128).any(axis=2)[440:520, 600:680].any()  # 戻し先の印はある


def _images(html):
    return re.findall(r'<img [^>]*src="data:image/jpeg;base64,([^"]+)"', html)


def test_build_html_has_photos_table_and_report():
    log = partial_log()
    report = ReportText("コップを戻しました。<ペン立て>は戻せませんでした。", "dummy", 0.004)
    html = build_html(log, _photos(), SCENE, report)
    imgs = _images(html)
    assert len(imgs) == 3
    img = Image.open(io.BytesIO(base64.b64decode(imgs[1])))
    assert img.width == 800  # 縮めて埋め込む
    assert html.count('<tr class="task">') == 3  # 物体ごとに 1 行
    assert "コップを戻しました。&lt;ペン立て&gt;は" in html  # 報告文はエスケープする
    assert CUP in html and "場所が分からなかった" in html and "やり直し" in html
    assert "開発用" not in html  # 真値がなければ開発用の欄は出さない


def test_build_html_shows_final_photo_when_retried():
    html = build_html(partial_log(), _photos(("reference", "current", "after", "final")), SCENE, ReportText("x", "t"))
    assert len(_images(html)) == 4


def test_ground_truth_only_in_dev_section():
    result = {"home_err_m": {"cup": 0.0035}, "home_yaw_err_deg": {"remote": 0.3}, "all_returned": True}
    html = build_html(partial_log(), _photos(), SCENE, ReportText("x", "t"), result=result)
    main, dev = html.split("開発用", 1)
    assert "3.5" not in main and "3.5 mm" in dev


def test_total_cost():
    log = partial_log()
    assert total_cost_usd(log, ReportText("x", "m", 0.004)) == pytest.approx(0.024)
    assert total_cost_usd(log, ReportText("x", "m", None)) is None


def test_load_run_roundtrip(tmp_path):
    log = partial_log()
    (tmp_path / "log.json").write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")
    for name in log["photos"]:
        Image.fromarray(_photo(10)).save(tmp_path / f"{name}_rgb.png")
    (tmp_path / "result.json").write_text(json.dumps({"home_err_m": {}}), encoding="utf-8")
    loaded, photos, result = load_run(tmp_path)
    assert loaded == log and set(photos) == set(log["photos"]) and result == {"home_err_m": {}}
    assert photos["after"].shape == (960, 1280, 3)


def test_make_report_writes_html_and_text(tmp_path):
    from restorebot.core.report_page import make_report
    from restorebot.core.vlm.client import DummyVlmClient

    log = partial_log()
    (tmp_path / "log.json").write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")
    for name in log["photos"]:
        Image.fromarray(_photo()).save(tmp_path / f"{name}_rgb.png")
    path, report = make_report(tmp_path, SCENE, DummyVlmClient("2 個を戻しました。"))
    html = path.read_text(encoding="utf-8")
    assert path.name == "report.html" and "2 個を戻しました。" in html and len(_images(html)) == 3
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved["text"] == "2 個を戻しました。" and saved["source"] == "dummy"
