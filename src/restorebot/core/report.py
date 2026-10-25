"""1 回の実行（エピソード）の記録と保存（設計書 11 章）。

outputs/runs/<日時>/ に、写真（RGB の PNG、深度の npy）と log.json（出来事、VLM の応答、
タスクごとの結果、時間と費用）を保存する。
評価（設計書 13 章）・動画・レポートの素材になる。
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from restorebot.core.types import Observation


@dataclass
class TaskRecord:
    """1 個の物体のタスクの結果。status は succeeded・failed（置くのに失敗）・skipped（掴めない・届かない）。"""

    key: str
    place_position: tuple[float, float, float]
    pick_position: tuple[float, float, float] | None = None
    status: str = "pending"
    reason: str | None = None
    pick_attempts: int = 0
    final_position: tuple[float, float, float] | None = None
    error_m: float | None = None  # 置いた後の物体と置く位置の水平距離（真値）
    started_s: float = 0.0  # シミュレーションの時刻
    duration_s: float = 0.0


def detection_to_dict(detection) -> dict:
    """core.vlm.detect.Detection → 記録用の dict。"""
    return {
        "ok": detection.ok,
        "prompt": detection.prompt,
        "attempts": [
            {"error": a.error, **(dataclasses.asdict(a.result) if a.result is not None else {})}
            for a in detection.attempts
        ],
        "moves": [dataclasses.asdict(m) for m in detection.moves],
        "dropped_static": [dataclasses.asdict(m) for m in detection.dropped_static],
        "cost_usd": detection.cost_usd,
        "latency_s": detection.latency_s,
    }


@dataclass
class EpisodeLog:
    meta: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    photos: dict[str, Observation] = field(default_factory=dict)
    detection: dict | None = None
    instructions: list[dict] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    tasks: list[TaskRecord] = field(default_factory=list)
    status: str = "running"  # succeeded・partial・failed
    reason: str | None = None
    sim_time_s: float = 0.0

    def event(self, t: float, kind: str, message: str = "", **data) -> None:
        self.events.append({"t": round(t, 3), "kind": kind, "message": message, **data})

    def finish(self, status: str, reason: str | None = None) -> None:
        self.status, self.reason = status, reason

    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "status": self.status,
            "reason": self.reason,
            "sim_time_s": self.sim_time_s,
            "cost_usd": None if self.detection is None else self.detection.get("cost_usd"),
            "events": self.events,
            "detection": self.detection,
            "instructions": self.instructions,
            "dropped": self.dropped,
            "tasks": [dataclasses.asdict(t) for t in self.tasks],
            "photos": sorted(self.photos),
        }

    def save(self, out_dir: Path) -> Path:
        """log.json と写真（<名前>_rgb.png・<名前>_depth.npy）を out_dir に保存する。"""
        from PIL import Image

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, obs in self.photos.items():
            Image.fromarray(obs.rgb).save(out_dir / f"{name}_rgb.png")
            np.save(out_dir / f"{name}_depth.npy", obs.depth)
        path = out_dir / "log.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        return path
