"""Issue #6：GUI の録画機能（Isaac Sim の Movie Capture）で mp4 を作り、headless の連番画像と比べる。

Movie Capture の実体は omni.kit.capture.viewport（エンコードは omni.videoencoding）。

video_record.py と同じ場面（#5 の部屋＋台車の往復）を、俯瞰カメラから録画する。
Movie Capture はタイムラインを進めながら 1 フレームずつ描画して mp4 にする（外部の ffmpeg は使わない）。
台車の速度指令は、物理の 1 ステップごとに呼ばれるコールバック（PHYSICS_PRE_STEP）で与える。
参考：Isaac Sim 同梱の skills/isaac-sim-remote/scripts/viewport_video.py

使い方::

    .venv\\Scripts\\python.exe scripts\\experiments\\video_gui_capture.py               # GUI を表示して録画
    .venv\\Scripts\\python.exe scripts\\experiments\\video_gui_capture.py --headless    # 表示せずに同じ処理

出力：outputs/m0_6/gui/gui_overview.mp4、summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # experiments.* を import するため

PHYSICS_HZ = 60


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--run", default="gui")
    args = parser.parse_args()
    out_dir = REPO_ROOT / "outputs" / "m0_6" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    t_start = time.perf_counter()
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})
    summary: dict = {"args": vars(args), "startup_s": time.perf_counter() - t_start}
    exit_code = 1
    try:
        import isaacsim.core.experimental.utils.app as app_utils
        from isaacsim.core.simulation_manager import SimulationEvent, SimulationManager

        app_utils.enable_extension("omni.videoencoding")
        app_utils.enable_extension("omni.kit.capture.viewport")
        for _ in range(5):
            simulation_app.update()
        from omni.kit.capture.viewport import (
            CaptureExtension,
            CaptureMovieType,
            CaptureOptions,
            CaptureRangeType,
            CaptureRenderPreset,
        )
        from omni.kit.capture.viewport import extension as capture_ext

        from experiments import ridgeback_sim as rs
        from experiments import video_scene as vs

        vs.build_scene(simulation_app)
        vs.create_camera("overview")
        SimulationManager.setup_simulation(dt=1.0 / PHYSICS_HZ)
        rs.play_and_warmup(simulation_app)
        driver = vs.BaseDriver(simulation_app)
        trajectory: list[list[float]] = []

        reinit_count = 0

        def on_pre_step(dt, context) -> None:
            nonlocal reinit_count
            try:
                pose = driver.step()
            except Exception:
                # Movie Capture の開始時にタイムラインが巻き戻され、物理の状態（Articulation のビュー）が消える。
                # 作り直して START の状態に戻す
                driver.reinit()
                reinit_count += 1
                trajectory.clear()
                pose = driver.step()
            if len(trajectory) % (PHYSICS_HZ // args.fps) == 0:
                trajectory.append([SimulationManager.get_simulation_time(), *pose, driver.wp_index])

        SimulationManager.register_callback(on_pre_step, event=SimulationEvent.PHYSICS_PRE_STEP)
        steps_before = SimulationManager.get_num_physics_steps()

        capture = getattr(capture_ext, "capture_instance", None)
        if capture is None:
            capture = CaptureExtension()
            capture.on_startup()
        n_frames = int(round(args.duration * args.fps))
        opts = CaptureOptions()
        opts.range_type = CaptureRangeType.FRAMES
        opts.start_frame = 1
        opts.end_frame = n_frames
        opts.fps = args.fps
        opts.animation_fps = args.fps  # タイムラインを進める間隔（再生が実時間になるように）
        opts.res_width, opts.res_height = 1280, 720
        opts.output_folder = str(out_dir)
        opts.file_name = "gui_overview"
        opts.file_type = ".mp4"
        opts.movie_type = CaptureMovieType.SEQUENCE
        opts.render_preset = CaptureRenderPreset.RAY_TRACE  # リアルタイムの RTX（headless の CameraSensor と同じ系統）
        opts.overwrite_existing_frames = True
        opts.camera = vs.camera_path("overview")
        capture.options = opts
        capture.show_default_progress_window = False
        t0 = time.perf_counter()
        if not capture.start():
            raise RuntimeError("CaptureExtension.start() が失敗しました")
        updates = 0
        while not capture.done:
            simulation_app.update()
            updates += 1
        capture_s = time.perf_counter() - t0
        output = out_dir / "gui_overview.mp4"
        summary.update(
            {
                "frames": n_frames,
                "capture_wall_s": capture_s,
                "app_updates": updates,
                "physics_steps": SimulationManager.get_num_physics_steps() - steps_before,
                "sim_time_end_s": SimulationManager.get_simulation_time(),
                "final_waypoint_index": driver.wp_index,
                "driver_reinit_count": reinit_count,
                "output": str(output),
                "size_mb": output.stat().st_size / 2**20 if output.exists() else None,
                "trajectory": trajectory,
            }
        )
        print(
            f"[gui] capture {n_frames} frames: {capture_s:.1f}s, physics steps={summary['physics_steps']}, "
            f"updates={updates}, size={summary['size_mb']} MB",
            flush=True,
        )
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


if __name__ == "__main__":
    sys.exit(main())
