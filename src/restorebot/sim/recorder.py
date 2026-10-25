"""撮影用のカメラの連番画像と mp4（設計書 12 章）。

※ SimulationApp を生成した後で import すること。#6 の方式：headless で、制御ループの中で
interval ステップごとに PNG を保存し、最後に ffmpeg（libx264）で mp4 にする。mp4 にしたら PNG は消す。
カメラ：overview（部屋の隅の上から見下ろす、固定）、follow（台車の後ろ上方から追従）、ceiling（天井カメラ）。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
from PIL import Image

from restorebot.core.types import Pose2D
from restorebot.core.video import capture_interval, clamp_to_room, encode_command, follow_camera_view, smooth_point
from restorebot.sim.scene_builder import PHYSICS_DT, SimScene

RESOLUTION = (720, 1280)  # (高さ, 幅)
# 部屋の南西の隅の上から、机と低い棚の両方を見下ろす
OVERVIEW_EYE, OVERVIEW_TARGET = (-1.85, -1.35, 2.3), (0.3, 0.35, 0.5)
FOLLOW = {"back": 1.3, "side": 0.4, "height": 1.9, "look_ahead": 0.7, "look_height": 0.7}
FOLLOW_ALPHA = 0.15  # 追従カメラのなめらかさ（1 フレームで目標に近づく割合）
# 焦点距離 [mm]（水平アパーチャ 20.955mm のとき。#6 の値）。既定のままでは画角が狭く、近くしか写らなかった（#14）
FOCAL_MM = {"overview": 14.0, "follow": 16.0}


def _video_camera(name: str) -> CameraSensor:
    rtx = RtxCamera(f"/World/video/{name}")
    rtx.camera.set_focal_lengths([FOCAL_MM[name] / 10])  # API の単位は USD の値の 1/10（#5）
    rtx.camera.set_apertures(horizontal_apertures=[2.0955])
    rtx.camera.set_clipping_ranges(near_distances=[0.05], far_distances=[30.0])
    return CameraSensor(rtx, resolution=RESOLUTION, annotators=["rgb"])


class Recorder:
    def __init__(self, sim: SimScene, out_dir: Path, robot_pose, *, fps: int = 30) -> None:
        self.sim = sim
        self.out_dir = Path(out_dir)
        self.robot_pose = robot_pose  # () -> Pose2D（追従カメラ用）
        self.fps = fps
        self.interval = capture_interval(physics_hz=round(1 / PHYSICS_DT), fps=fps)
        self.steps = 0
        self.frames = {"overview": 0, "follow": 0, "ceiling": 0}
        self.sensors = {name: _video_camera(name) for name in ("overview", "follow")}
        self.sensors["ceiling"] = sim.ceiling
        ViewportManager.set_camera_view("/World/video/overview", eye=list(OVERVIEW_EYE), target=list(OVERVIEW_TARGET))
        self._eye = self._target = None
        self._update_follow()

    def _update_follow(self) -> None:
        pose: Pose2D = self.robot_pose()
        eye, target = follow_camera_view(pose, **FOLLOW)
        eye = clamp_to_room(eye, half=self.sim.scene.room_half, margin=0.1)
        self._eye = smooth_point(self._eye, eye, alpha=FOLLOW_ALPHA)
        self._target = smooth_point(self._target, target, alpha=FOLLOW_ALPHA)
        ViewportManager.set_camera_view("/World/video/follow", eye=list(self._eye), target=list(self._target))

    def on_step(self) -> None:
        """物理の 1 ステップごとに呼ぶ（simulation_app.update() の後）。interval ステップごとに画像を保存する。"""
        self.steps += 1
        if self.steps % self.interval:
            return
        self._update_follow()
        for name, sensor in self.sensors.items():
            rgb, _ = sensor.get_data("rgb")
            if rgb is None:
                continue
            d = self.out_dir / "frames" / name
            d.mkdir(parents=True, exist_ok=True)
            img = np.ascontiguousarray(rgb.numpy()[:, :, :3], dtype=np.uint8)
            Image.fromarray(img).save(d / f"{self.frames[name]:05d}.png")
            self.frames[name] += 1

    def finalize(self, *, keep_frames: bool = False) -> dict[str, Path]:
        """カメラごとに mp4 を作る。keep_frames=False なら PNG を消す。"""
        videos = {}
        for name, n in self.frames.items():
            if n == 0:
                continue
            out = self.out_dir / f"{name}.mp4"
            pattern = str(self.out_dir / "frames" / name / "%05d.png")
            subprocess.run(encode_command(pattern, str(out), fps=self.fps), check=True)
            videos[name] = out
        if not keep_frames:
            shutil.rmtree(self.out_dir / "frames", ignore_errors=True)
        return videos
