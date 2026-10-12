"""Issue #6：headless で撮影用のカメラの画像を連番で保存し、ffmpeg で mp4 にする。カメラの台数による負荷も測る。

場面：#5 の部屋に RidgebackFranka を置き、#2 の推奨（速度指令）で台車を経由点に沿って動かす（アームは既定姿勢）。
カメラ：overview（部屋の隅の天井付近から見下ろす）、follow（台車の後ろ上方から追従）、ceiling（#5 の天井カメラ）。

使い方::

    .venv\\Scripts\\python.exe scripts\\experiments\\video_record.py                          # 3 台、10 秒、保存＋mp4
    .venv\\Scripts\\python.exe scripts\\experiments\\video_record.py --cameras overview --run cam1
    .venv\\Scripts\\python.exe scripts\\experiments\\video_record.py --cameras --run cam0     # カメラなし（負荷の基準）
    .venv\\Scripts\\python.exe scripts\\experiments\\video_record.py --no-save --run nosave   # 保存せずに取得だけ

出力（outputs/m0_6/<run>/）:
    frames/<カメラ>/%05d.png   連番画像
    <カメラ>_libx264.mp4 / <カメラ>_nvenc.mp4   動画
    overview_follow.mp4        俯瞰と追従を横に並べた動画
    summary.json               時間の内訳・GPU のメモリ・エンコードの時間とファイルの大きさ
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # experiments.* を import するため

from experiments.video_common import capture_interval, encode_command, hstack_command  # noqa: E402

PHYSICS_HZ = 60
RESOLUTION = (720, 1280)  # (高さ, 幅)
CAMERAS = ("overview", "follow", "ceiling")  # video_scene.CAMERAS と同じ（SimulationApp の前に引数を解析するため）


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", nargs="*", choices=CAMERAS, default=list(CAMERAS))
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--no-save", action="store_true", help="画像を取得するだけで保存しない（保存の負荷を除く）")
    parser.add_argument("--no-encode", action="store_true")
    parser.add_argument("--run", default="main")
    parser.add_argument("--gui", action="store_true", help="GUI を表示する（既定は headless）")
    args = parser.parse_args()
    out_dir = REPO_ROOT / "outputs" / "m0_6" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    t_start = time.perf_counter()
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": not args.gui})
    summary: dict = {"args": vars(args), "resolution": RESOLUTION, "startup_s": time.perf_counter() - t_start}
    exit_code = 1
    try:
        import numpy as np
        import pynvml
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.sensors.experimental.rtx import CameraSensor
        from PIL import Image

        from experiments import ridgeback_sim as rs
        from experiments import video_scene as vs

        pynvml.nvmlInit()
        gpu = pynvml.nvmlDeviceGetHandleByIndex(0)

        def gpu_mem_mb() -> float:
            return pynvml.nvmlDeviceGetMemoryInfo(gpu).used / 2**20

        summary["gpu_mem_before_mb"] = gpu_mem_mb()

        # ---- シーン（#5 の部屋＋ロボット）と撮影用のカメラ ----
        vs.build_scene(simulation_app)
        sensors: dict[str, CameraSensor] = {
            name: CameraSensor(vs.create_camera(name), resolution=RESOLUTION, annotators=["rgb"])
            for name in args.cameras
        }

        # ---- 物理開始、台車を START に置いて速度制御にする ----
        SimulationManager.setup_simulation(dt=1.0 / PHYSICS_HZ)
        rs.play_and_warmup(simulation_app)
        driver = vs.BaseDriver(simulation_app)
        follow = vs.FollowCamera(driver) if "follow" in sensors else None
        if follow is not None:
            follow.update()
        for _ in range(60):  # 描画を落ち着かせる（#5：RTX の時間方向の処理の残像）
            simulation_app.update()

        # ---- 撮影ループ ----
        interval = capture_interval(physics_hz=PHYSICS_HZ, fps=args.fps)
        n_steps = int(round(args.duration * PHYSICS_HZ))
        frames_dir = {name: out_dir / "frames" / name for name in sensors}
        for d in frames_dir.values():
            d.mkdir(parents=True, exist_ok=True)
        t_update = t_fetch = t_save = 0.0
        frame, missing = 0, 0
        trajectory = []
        t_loop = time.perf_counter()
        for step in range(n_steps):
            cur = driver.step()
            if follow is not None:
                follow.update()
            t0 = time.perf_counter()
            simulation_app.update()
            t_update += time.perf_counter() - t0
            if step % interval:
                continue
            trajectory.append([step / PHYSICS_HZ, *cur, driver.wp_index])
            for name, sensor in sensors.items():
                t0 = time.perf_counter()
                rgb, _ = sensor.get_data("rgb")
                img = None if rgb is None else np.ascontiguousarray(rgb.numpy()[:, :, :3], dtype=np.uint8)
                t_fetch += time.perf_counter() - t0
                if img is None:
                    missing += 1
                    continue
                if not args.no_save:
                    t0 = time.perf_counter()
                    Image.fromarray(img).save(frames_dir[name] / f"{frame:05d}.png", compress_level=1)
                    t_save += time.perf_counter() - t0
            frame += 1
        loop_s = time.perf_counter() - t_loop
        summary.update(
            {
                "cameras": list(sensors),
                "sim_duration_s": n_steps / PHYSICS_HZ,
                "frames": frame,
                "missing_images": missing,
                "loop_wall_s": loop_s,
                "realtime_factor": (n_steps / PHYSICS_HZ) / loop_s,
                "update_s": t_update,
                "fetch_s": t_fetch,
                "save_s": t_save,
                "update_ms_per_step": 1000 * t_update / n_steps,
                "gpu_mem_after_mb": gpu_mem_mb(),
                "final_waypoint_index": driver.wp_index,
                "trajectory": trajectory,
            }
        )
        print(
            f"[video] cameras={list(sensors)} loop={loop_s:.1f}s (x{summary['realtime_factor']:.2f} realtime) "
            f"update={t_update:.1f}s fetch={t_fetch:.1f}s save={t_save:.1f}s gpu={summary['gpu_mem_after_mb']:.0f}MB",
            flush=True,
        )
        # ---- mp4 への変換 ----
        if not args.no_save and not args.no_encode and sensors:
            summary["encode"] = encode_all(out_dir, list(sensors), args.fps)
            summary["encode"].update(encode_with_kit(out_dir, list(sensors)[0], args.fps, simulation_app))
        summary["status"] = "ok"
        exit_code = 0
    except Exception as exc:
        traceback.print_exc()
        summary["status"] = "failed"
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    simulation_app.close(exit_code=exit_code)
    return exit_code


def encode_all(out_dir: Path, cameras: list[str], fps: int) -> dict:
    results = {}

    def run(key: str, cmd: list[str], output: Path) -> None:
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        results[key] = {
            "seconds": time.perf_counter() - t0,
            "returncode": proc.returncode,
            "size_mb": output.stat().st_size / 2**20 if output.exists() else None,
            "stderr": proc.stderr[-500:],
        }
        print(f"[video] encode {key}: {results[key]['seconds']:.1f}s {results[key]['size_mb']} MB", flush=True)

    for name in cameras:
        pattern = str(out_dir / "frames" / name / "%05d.png")
        for codec, quality, suffix in (("libx264", 18, "libx264"), ("h264_nvenc", 19, "nvenc")):
            output = out_dir / f"{name}_{suffix}.mp4"
            run(f"{name}_{suffix}", encode_command(pattern, str(output), fps=fps, codec=codec, quality=quality), output)
    if {"overview", "follow"} <= set(cameras):
        output = out_dir / "overview_follow.mp4"
        inputs = [str(out_dir / f"{n}_libx264.mp4") for n in ("overview", "follow")]
        run("overview_follow", hstack_command(inputs, str(output), quality=20), output)
    return results


def encode_with_kit(out_dir: Path, camera: str, fps: int, simulation_app) -> dict:
    """Isaac Sim 同梱の omni.videoencoding（NVENC）で連番画像を mp4 にする（外部の ffmpeg を使わない方法）。"""
    import isaacsim.core.experimental.utils.app as app_utils

    app_utils.enable_extension("omni.videoencoding")
    for _ in range(3):
        simulation_app.update()
    from video_encoding import encode_image_file_sequence  # 拡張機能 omni.videoencoding の Python モジュール名

    output = out_dir / f"{camera}_kit.mp4"
    t0 = time.perf_counter()
    ok = encode_image_file_sequence(str(out_dir / "frames" / camera / "%05d.png"), 0, fps, str(output), True)
    result = {
        "seconds": time.perf_counter() - t0,
        "returncode": 0 if ok else 1,
        "size_mb": output.stat().st_size / 2**20 if output.exists() else None,
        "stderr": "",
    }
    print(f"[video] encode {camera}_kit: {result['seconds']:.1f}s {result['size_mb']} MB ok={ok}", flush=True)
    return {f"{camera}_kit": result}


if __name__ == "__main__":
    sys.exit(main())
