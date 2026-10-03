"""Claude Code の PostToolUse フック：編集された .py ファイルに ruff をかける。

- ruff format で整形する（安全な自動修正なので適用する）
- ruff check の指摘があれば stderr に出して終了コード 2 で終わり、Claude に直させる
  （`--fix` は使わない。import を先に書いて後で使う編集の途中で、未使用扱いの import が
  消されてしまうため）

終了コード: 0 = 問題なし／対象外、1 = フック自体のエラー（Claude は止めない）、2 = lint 違反あり
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    sys.stderr.reconfigure(encoding="utf-8")

    # ---- 入力（stdin の JSON）から編集されたファイルのパスを取り出す ----
    try:
        payload = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        print(f"[ruff hook] フックへの入力を解釈できません: {exc}", file=sys.stderr)
        return 1
    file_path = (payload.get("tool_input") or {}).get("file_path")
    if not file_path or not file_path.endswith(".py"):
        return 0

    # ---- プロジェクト外のファイル（一時ファイルなど）は対象外 ----
    project_dir = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2]).resolve()
    target = Path(file_path).resolve()
    if not target.is_file() or not target.is_relative_to(project_dir):
        return 0

    # ---- ruff を実行（--force-exclude で pyproject.toml の除外設定を明示パスにも適用）----
    def ruff(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "ruff", *args, "--force-exclude", str(target)],
            cwd=project_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

    fmt = ruff("format", "--quiet")
    if fmt.returncode != 0:  # 構文エラーなどで整形できない
        print(f"[ruff hook] ruff format に失敗:\n{fmt.stderr or fmt.stdout}", file=sys.stderr)
        return 2
    check = ruff("check", "--output-format", "concise", "--quiet")
    if check.returncode != 0:
        print(f"[ruff hook] ruff check の指摘があります。修正してください:\n{check.stdout}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
