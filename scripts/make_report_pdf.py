"""提出用のレポート（PDF）を作る（#21）。Isaac Sim は起動しない。

本文は docs/report/report.html（A4 に印刷する前提の HTML）。図は、録画した実行（make_video.py と同じ）と
評価の出力から作って <out>/fig に置き、Edge（headless）で PDF に印刷する。

使い方:
    .venv\\Scripts\\python.exe scripts\\make_report_pdf.py --run outputs\\m4_20\\seed104 --out outputs\\m4_21\\report
出力：<out>/report.pdf（<out>/report.html と <out>/fig も残す）
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from restorebot.core.report_page import draw_marks, photo_marks
from restorebot.core.scene import HOME_V2
from restorebot.core.vlm.summary import report_facts

REPO_ROOT = Path(__file__).resolve().parents[1]
EDGE = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")


def ffmpeg_frame(video: Path, at: float, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{at}", "-i", str(video), "-frames:v", "1",
         str(out)],
        check=True,
    )  # fmt: skip


def edge(*args: str, out: Path) -> None:
    """Edge（headless）を別のプロフィールで動かし、out ができるまで待つ。"""
    out.unlink(missing_ok=True)
    profile = out.parent / "edge_profile"
    subprocess.run([str(EDGE), "--headless=new", "--disable-gpu", f"--user-data-dir={profile}", *args], check=True)
    for _ in range(200):
        if out.exists() and out.stat().st_size > 0:
            return
        time.sleep(0.1)
    raise RuntimeError(f"{out} ができなかった")


def photo(run_dir: Path, name: str) -> np.ndarray:
    return np.asarray(Image.open(run_dir / f"{name}_rgb.png").convert("RGB"))


def crop_room(img: np.ndarray) -> Image.Image:
    """天井カメラの写真から、部屋の外（壁の上）を少し落とす。"""
    return Image.fromarray(img).crop((40, 30, 1240, 930))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="run_episode.py --record の出力（図に使う）")
    ap.add_argument(
        "--robot-pose", type=Path, default=REPO_ROOT / "outputs" / "m3_26" / "truth_seed2" / "robot_pose_after.png"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    fig = args.out / "fig"
    fig.mkdir(parents=True, exist_ok=True)

    # 図：部屋（俯瞰）、掴む様子（追従）
    ffmpeg_frame(args.run / "overview.mp4", 1.0, fig / "scene.png")
    log = json.loads((args.run / "log.json").read_text(encoding="utf-8"))
    first = sorted(log["tasks"], key=lambda t: t["started_s"])[0]
    ffmpeg_frame(args.run / "follow.mp4", first["started_s"] + 9.5, fig / "grasp.png")
    # 図：VLM の答え、確認
    facts = report_facts(log, HOME_V2)
    numbers = {o["name"]: o["no"] for o in facts["objects"]}
    marks = photo_marks(log, numbers)
    crop_room(draw_marks(photo(args.run, "current"), marks["current"])).save(fig / "vlm.png")
    crop_room(draw_marks(photo(args.run, "after"), marks["after"])).save(fig / "verify.png")
    # 図：台車の位置の推定（#26 の確認用の画像）、報告の画面
    shutil.copyfile(args.robot_pose, fig / "robot_pose.png")
    shot = fig / "report_full.png"
    edge("--hide-scrollbars", "--window-size=1100,1500", f"--screenshot={shot.resolve()}",
         (args.run / "report.html").resolve().as_uri(), out=shot.resolve())  # fmt: skip
    Image.open(shot).crop((0, 0, 1100, 1100)).save(fig / "report.png")

    html = args.out / "report.html"
    shutil.copyfile(REPO_ROOT / "docs" / "report" / "report.html", html)
    pdf = (args.out / "report.pdf").resolve()
    edge("--no-pdf-header-footer", f"--print-to-pdf={pdf}", html.resolve().as_uri(), out=pdf)
    print(f"[make_report_pdf] {pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
