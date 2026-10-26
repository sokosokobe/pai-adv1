"""保存済みの 1 回の実行の記録から、報告の画面（report.html）を作る（設計書 11 章。#24）。Isaac Sim は起動しない。

使い方:
    .venv\\Scripts\\python.exe scripts\\make_report.py outputs\\runs\\<実行のフォルダ>              # 報告文は決まった型
    .venv\\Scripts\\python.exe scripts\\make_report.py outputs\\runs\\<実行のフォルダ> --writer claude

出力（実行のフォルダに）:
    report.html   報告の画面（画像を埋め込んだ 1 ファイル）
    report.json   報告文（書いたモデル、費用）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from restorebot.core.report_page import make_report
from restorebot.core.scene import HOME_V2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--writer", choices=["template", "claude"], default="template", help="報告文を書くもの")
    args = ap.parse_args()
    writer = None
    if args.writer == "claude":
        from restorebot.core.vlm.client import make_client

        writer = make_client("claude")
    path, report = make_report(args.run_dir, HOME_V2, writer)
    print(f"[make_report] {path}  source={report.source} cost={report.cost_usd} error={report.error}")
    print(report.text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
