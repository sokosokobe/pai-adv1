"""Issue #5：VLM に「いつもの写真」と「今の写真」を渡し、動いた物体と戻し先の検出を評価する。

vlm_scene.py で作ったデータセット（outputs/m0_5/dataset/）を使う。Isaac Sim は不要。
Qwen を使う場合は、Ollama を起動し、qwen3-vl:8b-instruct-q8_0 を取得しておく（docs/experiments/5_vlm_diff.md を参照）。

使い方::

    .venv\\Scripts\\python.exe scripts\\experiments\\vlm_eval.py --models gemini claude qwen
    .venv\\Scripts\\python.exe scripts\\experiments\\vlm_eval.py --models claude --modes names --limit 1   # 動作確認

出力（outputs/m0_5/eval/<run>/）:
    results.jsonl   1 回の問い合わせごとの結果（応答の全文、トークン数、時間、照合の結果）。途中から再開できる
    summary.json    モデル × プロンプトごとの集計
    summary.md      集計の表
    overlay/        正解と予測を重ねた画像
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # experiments.* を import するため

from PIL import Image  # noqa: E402

from experiments.vlm_clients import make_client  # noqa: E402
from experiments.vlm_common import (  # noqa: E402
    SCENES,
    ScenarioTruth,
    aggregate,
    build_prompt,
    cost_usd,
    draw_overlay,
    evaluate,
    parse_prediction,
)

MODES = {"names": True, "nonames": False}  # 物体名のリストを渡す／渡さない


def load_dataset(dataset: Path) -> tuple[dict, list[tuple[dict, ScenarioTruth]]]:
    record = json.loads((dataset / "truth.json").read_text(encoding="utf-8"))
    if record.get("status") != "ok":
        raise RuntimeError(f"データセットが不完全です: {record.get('error')}")
    objects = SCENES[record.get("scene", "room_v1")].objects  # #5 のデータセットには scene の記録がない
    names = {o.key: o.name for o in objects}
    items = []
    for s in record["scenarios"]:
        truth = ScenarioTruth(
            moved=tuple(s["moved"]),
            names=names,
            current_px={k: tuple(v) for k, v in s["current_px"].items()},
            target_px={k: tuple(v) for k, v in s["target_px"].items()},
        )
        items.append((s, truth))
    return record, items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", choices=["gemini", "claude", "qwen"], default=["gemini", "claude", "qwen"]
    )
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "outputs" / "m0_5" / "dataset")
    parser.add_argument("--run", default="main", help="出力のフォルダ名")
    parser.add_argument(
        "--eval-root", type=Path, default=REPO_ROOT / "outputs" / "m0_5" / "eval", help="出力の親フォルダ"
    )
    parser.add_argument("--limit", type=int, default=None, help="使うシナリオの数（動作確認用）")
    parser.add_argument("--match-tol-px", type=float, default=40.0)
    parser.add_argument("--target-tol-px", type=float, default=40.0)
    parser.add_argument(
        "--robot-note",
        action="store_true",
        help="プロンプトに「ロボットが写ることがあるが片付けの対象ではない」を加える",
    )
    args = parser.parse_args()

    record, items = load_dataset(args.dataset)
    items = items[: args.limit] if args.limit else items
    cam = record["camera"]
    width, height = cam["width"], cam["height"]
    before_png = (args.dataset / "tidy_rgb.png").read_bytes()
    out_dir = args.eval_root / args.run
    (out_dir / "overlay").mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["model_key"], r["mode"], r["scenario"]))

    for model_key in args.models:
        client = make_client(model_key)
        for mode in args.modes:
            objects = SCENES[record.get("scene", "room_v1")].objects
            names = [o.name for o in objects] if MODES[mode] else None
            prompt = build_prompt(
                coord_format=client.coord_format,
                width=width,
                height=height,
                object_names=names,
                robot_note=args.robot_note,
            )
            (out_dir / f"prompt_{client.coord_format}_{mode}.txt").write_text(prompt, encoding="utf-8")
            for s, truth in items:
                if (model_key, mode, s["index"]) in done:
                    continue
                after_png = (args.dataset / s["image"]).read_bytes()
                row: dict = {"model_key": model_key, "model": client.name, "mode": mode, "scenario": s["index"]}
                preds = None
                try:
                    res = client.detect(before_png, after_png, prompt)
                    row.update(
                        text=res.text,
                        input_tokens=res.input_tokens,
                        output_tokens=res.output_tokens,
                        latency_s=res.latency_s,
                        cost_usd=cost_usd(
                            res.input_tokens,
                            res.output_tokens,
                            price_in_per_mtok=client.price_in_per_mtok,
                            price_out_per_mtok=client.price_out_per_mtok,
                        ),
                    )
                    try:
                        preds = parse_prediction(res.text, coord_format=client.coord_format, width=width, height=height)
                    except ValueError as exc:
                        row["parse_error"] = str(exc)
                except Exception as exc:  # API のエラーも記録して続ける
                    traceback.print_exc()
                    row["api_error"] = f"{type(exc).__name__}: {exc}"
                score = evaluate(
                    preds,
                    truth,
                    use_names=MODES[mode],
                    match_tol_px=args.match_tol_px,
                    target_tol_px=args.target_tol_px,
                )
                row["predictions"] = [asdict(p) for p in preds] if preds is not None else None
                row["score"] = asdict(score)
                with results_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                after_img = Image.open(args.dataset / s["image"])
                draw_overlay(after_img, truth, preds).save(
                    out_dir / "overlay" / f"{model_key}_{mode}_scene_{s['index']:02d}.png"
                )
                print(
                    f"[vlm_eval] {client.name:28s} {mode:8s} scene {s['index']:02d}: "
                    f"tp={score.tp} fp={score.fp} fn={score.fn} target_ok={score.target_ok} "
                    f"t={row.get('latency_s', 0):.1f}s ${row.get('cost_usd', 0):.4f} "
                    f"{row.get('parse_error', '') or row.get('api_error', '')}",
                    flush=True,
                )

    summarize(results_path, out_dir)
    return 0


def summarize(results_path: Path, out_dir: Path) -> None:
    from experiments.vlm_common import Match, ScenarioScore

    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["model"], r["mode"]), []).append(r)
    summary = {}
    for (model, mode), rs in sorted(groups.items()):
        scores = [
            ScenarioScore(**{**r["score"], "matches": tuple(Match(**m) for m in r["score"]["matches"])}) for r in rs
        ]
        lat = [r["latency_s"] for r in rs if "latency_s" in r]
        costs = [r["cost_usd"] for r in rs if "cost_usd" in r]
        summary[f"{model}|{mode}"] = {
            "model": model,
            "mode": mode,
            **aggregate(scores),
            "api_errors": sum("api_error" in r for r in rs),
            "latency_s_median": statistics.median(lat) if lat else None,
            "latency_s_max": max(lat) if lat else None,
            "cost_usd_mean": statistics.mean(costs) if costs else None,
            "input_tokens_mean": statistics.mean(r["input_tokens"] for r in rs if "input_tokens" in r) if lat else None,
            "output_tokens_mean": statistics.mean(r["output_tokens"] for r in rs if "output_tokens" in r)
            if lat
            else None,
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    def f(v, fmt="{:.2f}"):
        return "-" if v is None else fmt.format(v)

    header = [
        "モデル",
        "名前のリスト",
        "適合率",
        "再現率",
        "完全一致",
        "戻し先OK",
        "戻し先の誤差 [px]",
        "解析失敗",
        "応答時間（中央値）[s]",
        "費用/回 [$]",
    ]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for v in summary.values():
        cells = [
            v["model"],
            "あり" if v["mode"] == "names" else "なし",
            f(v["precision"]),
            f(v["recall"]),
            f(v["exact_match_rate"]),
            f(v["target_ok_rate"]),
            f(v["target_err_px_mean"], "{:.0f}"),
            f(v["parse_failure_rate"]),
            f(v["latency_s_median"], "{:.1f}"),
            f(v["cost_usd_mean"], "{:.4f}"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    sys.exit(main())
