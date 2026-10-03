"""Claude Code のフック（.claude/hooks/ruff_after_edit.py）のテスト（Isaac Sim 不要）。

フックは編集された .py ファイルに ruff format をかけ、ruff check の指摘があれば
終了コード 2 で Claude に知らせる（自動修正はしない）。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK = REPO_ROOT / ".claude" / "hooks" / "ruff_after_edit.py"


def run_hook(payload: dict | str, project_dir: Path) -> subprocess.CompletedProcess:
    stdin = payload if isinstance(payload, str) else json.dumps({"tool_name": "Edit", **payload})
    return subprocess.run(
        [sys.executable, str(HOOK)],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(project_dir)},
    )


def edit_payload(path: Path) -> dict:
    return {"tool_input": {"file_path": str(path)}}


def test_formats_python_file(tmp_path):
    target = tmp_path / "a.py"
    target.write_text('x = {  "a":1 }\n', encoding="utf-8")
    proc = run_hook(edit_payload(target), tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert target.read_text(encoding="utf-8") == 'x = {"a": 1}\n'


def test_reports_lint_error_without_fixing(tmp_path):
    target = tmp_path / "a.py"
    target.write_text("import os\n", encoding="utf-8")
    proc = run_hook(edit_payload(target), tmp_path)
    assert proc.returncode == 2  # Claude に指摘を返す
    assert "F401" in proc.stderr
    # 編集の途中で import を先に書くことがあるので、未使用の import を勝手に消さない
    assert target.read_text(encoding="utf-8") == "import os\n"


def test_ignores_non_python_file(tmp_path):
    target = tmp_path / "notes.md"
    target.write_text("x = {  'a':1 }\n", encoding="utf-8")
    proc = run_hook(edit_payload(target), tmp_path)
    assert proc.returncode == 0
    assert target.read_text(encoding="utf-8") == "x = {  'a':1 }\n"


def test_ignores_file_outside_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("import os\n", encoding="utf-8")
    proc = run_hook(edit_payload(outside), project)
    assert proc.returncode == 0
    assert outside.read_text(encoding="utf-8") == "import os\n"


def test_ignores_missing_file_path(tmp_path):
    proc = run_hook({"tool_input": {}}, tmp_path)
    assert proc.returncode == 0


def test_malformed_input_is_non_blocking_error(tmp_path):
    proc = run_hook("not json", tmp_path)
    assert proc.returncode == 1  # 1 = Claude を止めないエラー（ユーザーに表示される）
