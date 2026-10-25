r"""run_episode.py を実際に Isaac Sim で動かす統合テスト（M1、真値から作る VLM の応答、動画なし）。

実行: .venv\Scripts\python.exe -m pytest -m isaac
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_episode.py"
TIMEOUT_S = 900


@pytest.mark.isaac
def test_m1_episode_with_truth_vlm(tmp_path):
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--seed", "1", "--out", str(tmp_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT_S,
    )
    assert proc.returncode == 0, f"exit={proc.returncode}\n--- stdout (tail) ---\n{proc.stdout[-4000:]}"
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "succeeded", result
    assert result["all_returned"], result["home_err_m"]
    assert all(m < 0.01 for m in result["untouched_moved_m"].values()), result["untouched_moved_m"]
    log = json.loads((tmp_path / "log.json").read_text(encoding="utf-8"))
    assert {"reference", "current", "after"} <= set(log["photos"])
    assert not any(e["kind"] == "arm_failed" for e in log["events"]), log["events"]
