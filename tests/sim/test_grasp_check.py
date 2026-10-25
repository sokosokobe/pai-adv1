r"""grasp_check.py --quick を実際に Isaac Sim で動かす統合テスト（シーンの作成、天井カメラ、掴む・置くを 1 回）。

実行: .venv\Scripts\python.exe -m pytest -m isaac
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "grasp_check.py"
TIMEOUT_S = 900


@pytest.mark.isaac
def test_grasp_check_quick(tmp_path):
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--quick", "--out", str(tmp_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT_S,
    )
    assert proc.returncode == 0, f"exit={proc.returncode}\n--- stdout (tail) ---\n{proc.stdout[-4000:]}"
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "ok", summary.get("error")

    # シーンの作成と天井カメラ（1280 × 960 の RGB）
    assert (tmp_path / "ceiling_tidy.png").is_file()
    assert summary["ceiling"]["shape"] == [960, 1280, 3]

    # 掴む・置くを 1 回：成功し、戻し先から 3cm 以内
    (trial,) = summary["trials"]
    assert trial["outcome"] == "success", trial
    assert trial["observation"]["place_xy_err_m"] < 0.03
    assert all(m < 0.01 for m in trial["others_moved_m"].values()), trial["others_moved_m"]
