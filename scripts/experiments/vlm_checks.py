"""Issue #5 の補助的な確認（Isaac Sim は不要）。

ghost：前のシーンで物体があった所（今のシーンにも、いつもの状態にも物体がない所）の、いつもの画像との差を測る。
       RTX の時間方向の処理による残像が残っていないかの確認。比較用に、物体のない床の差も出す。
qwen-single：Qwen3-VL（Ollama）に今の画像を 1 枚だけ渡して物体を指させ、位置の誤差を測る。
       2 枚を比べさせたときの「今の位置」の誤差（評価の結果）と比べる。

使い方::

    .venv\\Scripts\\python.exe scripts\\experiments\\vlm_checks.py ghost outputs\\m0_5\\dataset
    .venv\\Scripts\\python.exe scripts\\experiments\\vlm_checks.py qwen-single outputs\\m0_5\\dataset --limit 5
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import statistics
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # experiments.* を import するため

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.vlm_common import CEILING_CAMERA, OBJECTS, TIDY_LAYOUT, Placement, object_center  # noqa: E402

PATCH = 8  # 差を測る正方形の半幅 [px]
FLOOR_PATCH_CENTER = (508, 608)  # どのシーンでも物体のない床の点 (u, v)


def _patch_diff(a: np.ndarray, b: np.ndarray, u: float, v: float) -> float:
    u, v = int(u), int(v)
    sl = (slice(v - PATCH, v + PATCH), slice(u - PATCH, u + PATCH))
    return float(np.abs(a[sl] - b[sl]).mean())


def ghost(dataset: Path) -> None:
    record = json.loads((dataset / "truth.json").read_text(encoding="utf-8"))
    specs = {o.key: o for o in OBJECTS}
    tidy = np.asarray(Image.open(dataset / "tidy_rgb.png"), dtype=float)
    prev = dict(TIDY_LAYOUT)
    for s in record["scenarios"]:
        cur = {k: Placement(**v) for k, v in s["placements"].items()}
        img = np.asarray(Image.open(dataset / s["image"]), dtype=float)
        occupied = [
            CEILING_CAMERA.project(object_center(specs[k], p)) for d in (cur, TIDY_LAYOUT) for k, p in d.items()
        ]
        diffs = []
        for k, p in prev.items():
            u, v = CEILING_CAMERA.project(object_center(specs[k], p))
            if min(math.dist((u, v), o) for o in occupied) < 60:  # 今も物体がある所は除く
                continue
            diffs.append(round(_patch_diff(img, tidy, u, v), 2))
        floor = round(_patch_diff(img, tidy, *FLOOR_PATCH_CENTER), 2)
        print(f"{s['image']}: 前のシーンの物体の跡 {diffs}  物体のない床 {floor}  （平均の差、階調）")
        prev = cur


def qwen_single(dataset: Path, limit: int, model: str, base_url: str) -> None:
    record = json.loads((dataset / "truth.json").read_text(encoding="utf-8"))
    names = {o.key: o.name for o in OBJECTS}
    width, height = record["camera"]["width"], record["camera"]["height"]
    prompt = (
        "Point to the center of each of these objects in the image: "
        + ", ".join(names.values())
        + '. Answer JSON only: {"points": [{"name": "...", "point": [x, y]}]} with [x, y] normalized to 0-1000.'
    )
    errs = []
    for s in record["scenarios"][:limit]:
        img = base64.standard_b64encode((dataset / s["image"]).read_bytes()).decode("ascii")
        payload = {
            "model": model,
            "stream": False,
            "format": "json",
            "messages": [{"role": "user", "content": prompt, "images": [img]}],
            "options": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": 0, "num_ctx": 8192},
        }
        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            answer = json.loads(json.load(r)["message"]["content"])
        points = {p["name"]: p["point"] for p in answer.get("points", [])}
        row = {}
        for key, name in names.items():
            if name in points:
                u, v = points[name][0] / 1000 * width, points[name][1] / 1000 * height
                row[key] = round(math.dist((u, v), s["current_px"][key]), 1)
                errs.append(row[key])
        print(f"{s['image']}: 誤差 [px] {row}")
    print(f"1 枚だけ渡した場合の誤差：中央値 {statistics.median(errs):.1f}px、最大 {max(errs):.1f}px（{len(errs)} 個）")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("check", choices=["ghost", "qwen-single"])
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model", default="qwen3-vl:8b-instruct-q8_0")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    if args.check == "ghost":
        ghost(args.dataset)
    else:
        qwen_single(args.dataset, args.limit, args.model, args.base_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
