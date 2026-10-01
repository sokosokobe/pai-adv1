r"""smoke_test.py を実際に Isaac Sim で動かす統合テスト。

時間がかかる（初回はシェーダキャッシュ作成で数分）ため isaac マーカー付き。
実行: .venv\Scripts\python.exe -m pytest -m isaac
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_test.py"
STEPS = 100
TIMEOUT_S = 900

# result.json に必ず含まれるべきキー（セクション -> キー）
REQUIRED_KEYS = {
    "versions": ["isaacsim_app", "python", "driver"],
    "timing": ["steps", "wall_time_s", "sim_time_s", "realtime_factor", "avg_fps"],
    "resources": ["vram_peak_mib", "ram_peak_mib"],
}


@pytest.mark.isaac
def test_smoke_test_runs_headless(tmp_path):
    # ---- smoke_test.py をサブプロセスで実行（出力先は一時フォルダ）----
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    proc = subprocess.run(
        [sys.executable, str(SMOKE_SCRIPT), "--steps", str(STEPS), "--output-root", str(tmp_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT_S,
    )
    assert proc.returncode == 0, f"exit={proc.returncode}\n--- stdout (tail) ---\n{proc.stdout[-4000:]}"

    # ---- 出力フォルダ（<日時>）が1つだけ作られていること ----
    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1, run_dirs
    run_dir = run_dirs[0]

    # ---- result.json の必須キー ----
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "ok", result.get("error")
    for section, keys in REQUIRED_KEYS.items():
        assert section in result, f"missing section: {section}"
        for key in keys:
            assert key in result[section], f"missing key: {section}.{key}"
    assert result["timing"]["steps"] == STEPS

    # ---- カメラ画像 ----
    image_path = run_dir / result["camera"]["image"]
    assert image_path.is_file()
    assert image_path.stat().st_size > 0
