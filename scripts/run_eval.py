"""定量評価（設計書 13 章。#18）：複数のシードでエピソードを実行し、結果を集計する。

シードごとに scripts/run_episode.py を別のプロセスで実行する（1 つのプロセスでシーンをリセットして繰り返すと、
物理の状態が残って結果が変わることがあるため）。M2 の条件（移動あり）、物体の位置は画像から推定、確認とやり直しあり。

使い方:
    .venv\\Scripts\\python.exe scripts\\run_eval.py --vlm claude --seeds 100-109 --out outputs\\m3_18\\claude --report
    .venv\\Scripts\\python.exe scripts\\run_eval.py --vlm truth --seeds 100-109 --out outputs\\m3_18\\truth
    .venv\\Scripts\\python.exe scripts\\run_eval.py --out outputs\\m3_18\\claude --aggregate-only   # 集計だけやり直す

出力（<out>）:
    seed<シード>/        run_episode.py の出力（log.json・result.json・写真、--report なら report.html）
    summary.json         エピソードごとの指標と全体の集計
    summary.md           表（docs/evaluation.md に貼る）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_seeds(text: str) -> list[int]:
    """ "100-109" や "0,3,5" → シードの一覧。"""
    seeds: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        else:
            seeds.append(int(part))
    return seeds


def run_seed(seed: int, args) -> int:
    out = args.out / f"seed{seed}"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_episode.py"),
        "--milestone",
        "m2",
        "--seed",
        str(seed),
        "--vlm",
        args.vlm,
        "--perception",
        "estimated",
        "--out",
        str(out),
    ]
    if args.report:
        cmd.append("--report")
    env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
    t0 = time.perf_counter()
    with open(args.out / f"seed{seed}.log", "w", encoding="utf-8") as f:
        code = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=REPO_ROOT).returncode
    print(f"[run_eval] seed={seed} exit={code} wall={time.perf_counter() - t0:.0f}s", flush=True)
    return code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vlm", choices=["truth", "claude"], default="truth")
    ap.add_argument("--seeds", default="100-109")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", action="store_true", help="エピソードごとに報告の画面も作る")
    ap.add_argument("--aggregate-only", action="store_true", help="実行せず、保存済みの結果を集計する")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        for seed in parse_seeds(args.seeds):
            run_seed(seed, args)
    return aggregate(args.out)


def aggregate(out: Path) -> int:
    from restorebot.core.evaluation import evaluate_run, markdown_tables, summarize
    from restorebot.core.scene import HOME_V2

    runs = []
    for d in sorted(out.glob("seed*"), key=lambda p: int(p.name[4:]) if p.name[4:].isdigit() else -1):
        if not d.is_dir():
            continue
        result_path, log_path = d / "result.json", d / "log.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {"seed": d.name}
        log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else None
        runs.append(evaluate_run(result, log, HOME_V2))
    summary = summarize(runs)
    (out / "summary.json").write_text(
        json.dumps({"runs": runs, "summary": summary}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out / "summary.md").write_text(markdown_tables(runs, summary), encoding="utf-8")
    print(f"[run_eval] {json.dumps(summary, ensure_ascii=False)} -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
