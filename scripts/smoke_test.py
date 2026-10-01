"""Isaac Sim 6.1 headless 動作検証（スモークテスト）スクリプト。

地面・Franka・立方体1個・RGBカメラ1台を配置し、物理 60Hz・レンダリングありで
N ステップ進めて、性能とリソース使用量を ``outputs/smoke/<日時>/result.json`` に記録する。

使い方（リポジトリ直下で実行）::

    $env:OMNI_KIT_ACCEPT_EULA = "YES"
    .venv\\Scripts\\python.exe scripts\\smoke_test.py --steps 1000
    .venv\\Scripts\\python.exe scripts\\smoke_test.py --no-camera
    .venv\\Scripts\\python.exe scripts\\smoke_test.py --resolution 1280x720

API は Isaac Sim 6.1 同梱ソースに合わせている:
    - 起動〜ステップ: skills/isaac-sim-headless-deployment/scripts/batch_simulation.py
    - シーン構築:     isaacsim.robot_motion.examples の manipulation/scenario.py, pick_place_task.py
    - カメラ:         isaacsim.sensors.experimental.rtx の RtxCamera / CameraSensor
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import psutil
import pynvml

# ============================================================
# 定数
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
PHYSICS_DT = 1.0 / 60.0  # 物理 60Hz
WARMUP_STEPS = 30  # 計測前の慣らし（シェーダコンパイル・アセット読込の影響を除く）
CAMERA_MAX_WAIT_STEPS = 60  # カメラ画像が届くまで追加で待つ最大ステップ数
CUBE_SIZE = 0.0515  # Pick and Place サンプルと同じ立方体サイズ [m]
CUBE_DROP_HEIGHT = 0.30  # 立方体を落下させる初期高さ [m]（物理が動いているかの確認用）

# Franka アセット（6.1 同梱の Pick and Place サンプル robots.py と同じパス・バリアント）
FRANKA_ASSET = "/Isaac/Robots_Multiphysics/FrankaRobotics/FrankaPanda/franka/franka.usda"
FRANKA_VARIANTS = [("Gripper", "alternatefinger"), ("Mesh", "performance")]

EXIT_OK = 0
EXIT_FAILED = 1


# ============================================================
# 引数解析
# ============================================================
def parse_resolution(text: str) -> tuple[int, int]:
    """'640x480' 形式の文字列を (width, height) に変換する。"""
    try:
        w_str, h_str = text.lower().split("x")
        width, height = int(w_str), int(h_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"解像度は WIDTHxHEIGHT 形式で指定してください: {text!r}") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"解像度は正の値にしてください: {text!r}")
    return width, height


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isaac Sim headless smoke test")
    parser.add_argument("--steps", type=int, default=1000, help="計測するステップ数（既定 1000）")
    parser.add_argument("--no-camera", action="store_true", help="RGB カメラを配置しない")
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        default=(640, 480),
        help="カメラ解像度 WIDTHxHEIGHT（既定 640x480）",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "smoke",
        help="結果の出力先ルート（既定 outputs/smoke）",
    )
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("--steps は 1 以上にしてください")
    return args


# ============================================================
# リソース監視（別スレッドで VRAM / RAM をポーリングしピークを記録）
# ============================================================
class ResourceMonitor:
    """nvidia-ml-py と psutil で VRAM / RAM のピークを記録する。

    VRAM はデバイス全体の使用量（他プロセス分も含む）を測る。Windows (WDDM) では
    プロセス単位の VRAM が取得できないことが多いため、起動前の値をベースラインとして
    差分も記録する。
    """

    def __init__(self, interval_s: float = 0.1, gpu_index: int = 0) -> None:
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._process = psutil.Process(os.getpid())
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.gpu_name = pynvml.nvmlDeviceGetName(self._handle)
        self.driver_version = pynvml.nvmlSystemGetDriverVersion()
        cuda = pynvml.nvmlSystemGetCudaDriverVersion()
        self.cuda_driver_version = f"{cuda // 1000}.{(cuda % 1000) // 10}"
        self.vram_total_mib = pynvml.nvmlDeviceGetMemoryInfo(self._handle).total / 2**20
        # 起動前のベースライン
        self.vram_baseline_mib = self._vram_used_mib()
        self.vram_peak_mib = self.vram_baseline_mib
        self.vram_process_peak_mib: float | None = None
        self.ram_peak_mib = self._process.memory_info().rss / 2**20  # ポーリングで得た RSS のピーク
        self.ram_peak_wset_mib: float | None = None  # Windows が記録するピークワーキングセット
        self.samples = 0

    def _vram_used_mib(self) -> float:
        return pynvml.nvmlDeviceGetMemoryInfo(self._handle).used / 2**20

    def _vram_process_mib(self) -> float | None:
        """自プロセスの VRAM 使用量（取得できない環境では None）。"""
        try:
            procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(
                self._handle
            ) + pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
        except pynvml.NVMLError:
            return None
        used = [p.usedGpuMemory for p in procs if p.pid == self._process.pid and p.usedGpuMemory]
        return max(used) / 2**20 if used else None

    def sample(self) -> None:
        self.vram_peak_mib = max(self.vram_peak_mib, self._vram_used_mib())
        proc_vram = self._vram_process_mib()
        if proc_vram is not None:
            self.vram_process_peak_mib = max(self.vram_process_peak_mib or 0.0, proc_vram)
        self.ram_peak_mib = max(self.ram_peak_mib, self._process.memory_info().rss / 2**20)
        self.samples += 1

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            with contextlib.suppress(Exception):  # 監視の失敗で本体を止めない
                self.sample()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.sample()  # 最後に1回取り直す
        # Windows ではプロセスのピークワーキングセットも取れる（ポーリング間隔の取りこぼしが無い）
        peak_wset = getattr(self._process.memory_info(), "peak_wset", None)
        if peak_wset:
            self.ram_peak_wset_mib = peak_wset / 2**20
        pynvml.nvmlShutdown()

    def to_dict(self) -> dict:
        return {
            "vram_peak_mib": round(self.vram_peak_mib, 1),
            "vram_baseline_mib": round(self.vram_baseline_mib, 1),
            "vram_peak_delta_mib": round(self.vram_peak_mib - self.vram_baseline_mib, 1),
            "vram_process_peak_mib": (
                round(self.vram_process_peak_mib, 1) if self.vram_process_peak_mib is not None else None
            ),
            "vram_total_mib": round(self.vram_total_mib, 1),
            # ram_peak_mib はポーリング値と Windows のピークワーキングセットの大きい方
            "ram_peak_mib": round(max(self.ram_peak_mib, self.ram_peak_wset_mib or 0.0), 1),
            "ram_rss_peak_polled_mib": round(self.ram_peak_mib, 1),
            "ram_peak_wset_mib": round(self.ram_peak_wset_mib, 1) if self.ram_peak_wset_mib is not None else None,
            "ram_total_mib": round(psutil.virtual_memory().total / 2**20, 1),
            "monitor_samples": self.samples,
        }


# ============================================================
# 結果の書き出し
# ============================================================
def write_result(out_dir: Path, result: dict) -> Path:
    path = out_dir / "result.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    width, height = args.resolution
    use_camera = not args.no_camera

    # ------------------------------------------------------------
    # 出力フォルダ作成と、Isaac Sim 起動前に取れる情報の記録
    # ------------------------------------------------------------
    out_dir = args.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")  # EULA は環境変数で承諾

    monitor = ResourceMonitor()
    monitor.start()
    result: dict = {
        "status": "running",
        "error": None,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "args": {
            "steps": args.steps,
            "camera": use_camera,
            "resolution": [width, height] if use_camera else None,
        },
        "versions": {
            "isaacsim_pip": importlib.metadata.version("isaacsim"),
            "isaacsim_app": None,  # 起動後に埋める
            "python": platform.python_version(),
            "torch": _safe_version("torch"),
            "gpu": monitor.gpu_name,
            "driver": monitor.driver_version,
            "cuda_driver": monitor.cuda_driver_version,
            "os": platform.platform(),
        },
    }
    print(f"[smoke] 出力先: {out_dir}", flush=True)

    simulation_app = None
    exit_code = EXIT_FAILED
    try:
        # ------------------------------------------------------------
        # 1. SimulationApp を headless で起動
        #    ※ 他の Isaac Sim / omni / pxr モジュールの import は必ずこの後
        # ------------------------------------------------------------
        t_launch = time.perf_counter()
        from isaacsim import SimulationApp

        simulation_app = SimulationApp({"headless": True, "width": width, "height": height})
        launch_s = time.perf_counter() - t_launch
        print(f"[smoke] SimulationApp 起動 {launch_s:.1f}s", flush=True)

        import carb
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        import numpy as np
        from isaacsim.core.experimental.objects import Cube, DomeLight, GroundPlane
        from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim, XformPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.version import get_version
        from isaacsim.storage.native import get_assets_root_path

        result["versions"]["isaacsim_app"] = get_version()[0]
        result["versions"]["kit"] = simulation_app.app.get_build_version()
        result["renderer"] = carb.settings.get_settings().get_as_string("/rtx/rendermode")

        # ------------------------------------------------------------
        # 2. シーン構築：地面・ライト・Franka・立方体・カメラ
        # ------------------------------------------------------------
        t_scene = time.perf_counter()
        GroundPlane("/World/ground_plane")
        DomeLight("/World/DomeLight").set_intensities(1000)

        # Franka（アセットルートは Pick and Place サンプルと同様に解決）
        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("Isaac Sim のアセットルートを解決できませんでした（ネットワーク/Nucleus を確認）")
        stage_utils.add_reference_to_stage(assets_root + FRANKA_ASSET, "/World/franka", variants=FRANKA_VARIANTS)
        XformPrim("/World/franka", positions=[0.0, 0.0, 0.0], orientations=[1.0, 0.0, 0.0, 0.0])
        franka = Articulation("/World/franka")

        # 立方体：少し高い位置から落とし、物理が働いているかを最終位置で確認する
        cube_shape = Cube(
            "/World/cube",
            sizes=1.0,
            scales=[CUBE_SIZE, CUBE_SIZE, CUBE_SIZE],
            positions=[0.5, 0.0, CUBE_DROP_HEIGHT],
            colors="blue",
        )
        GeomPrim(cube_shape.paths, apply_collision_apis=True)
        cube = RigidPrim(cube_shape.paths)

        # RGB カメラ（新 API: RtxCamera + CameraSensor。解像度は (height, width) の順）
        camera_sensor = None
        if use_camera:
            from isaacsim.core.rendering_manager import ViewportManager
            from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

            rtx_camera = RtxCamera("/World/smoke_camera")
            camera_sensor = CameraSensor(rtx_camera, resolution=(height, width), annotators=["rgb"])
            ViewportManager.set_camera_view("/World/smoke_camera", eye=[2.4, 1.8, 1.5], target=[0.25, 0.0, 0.35])

        # アセット（Franka の USD など）の読み込み完了を待つ
        while stage_utils.is_stage_loading():
            simulation_app.update()
        scene_s = time.perf_counter() - t_scene
        print(f"[smoke] シーン構築 {scene_s:.1f}s", flush=True)

        # ------------------------------------------------------------
        # 3. 物理 60Hz を設定して再生開始 → 慣らし運転
        # ------------------------------------------------------------
        SimulationManager.setup_simulation(dt=PHYSICS_DT)
        app_utils.play()
        for _ in range(WARMUP_STEPS):
            simulation_app.update()

        # Franka が物理エンティティとして有効になったことを確認し、関節の初期目標を取得
        num_dofs = franka.num_dofs
        dof_names = list(franka.dof_names)
        base_targets = franka.get_dof_position_targets().numpy()[0].copy()
        joint1 = dof_names.index("panda_joint1")

        # ------------------------------------------------------------
        # 4. 計測ループ：N ステップ（1 update = 物理1ステップ + レンダリング1回）
        #    Franka の第1関節をゆっくり振り、関節制御も動くことを確認する
        # ------------------------------------------------------------
        sim_time_start = SimulationManager.get_simulation_time()
        physics_steps_start = SimulationManager.get_num_physics_steps()
        t_loop = time.perf_counter()
        for i in range(args.steps):
            targets = base_targets.copy()
            targets[joint1] += 0.5 * math.sin(2.0 * math.pi * 0.25 * i * PHYSICS_DT)
            franka.set_dof_position_targets(targets)
            simulation_app.update()
        wall_s = time.perf_counter() - t_loop
        sim_s = SimulationManager.get_simulation_time() - sim_time_start
        physics_steps = SimulationManager.get_num_physics_steps() - physics_steps_start

        # 物理の妥当性チェック：落とした立方体が地面の上で止まっているか
        cube_z = float(cube.get_world_poses()[0].numpy()[0][2])
        cube_on_ground = abs(cube_z - CUBE_SIZE / 2) < 0.01
        joint1_pos = float(franka.get_dof_positions().numpy()[0][joint1])

        result["timing"] = {
            "launch_s": round(launch_s, 2),
            "scene_setup_s": round(scene_s, 2),
            "warmup_steps": WARMUP_STEPS,
            "steps": args.steps,
            "physics_dt": PHYSICS_DT,
            "physics_steps": physics_steps,
            "wall_time_s": round(wall_s, 3),
            "sim_time_s": round(sim_s, 3),
            "realtime_factor": round(sim_s / wall_s, 3),
            "avg_fps": round(args.steps / wall_s, 2),
        }
        result["physics_checks"] = {
            "franka_num_dofs": num_dofs,
            "franka_joint1_pos_rad": round(joint1_pos, 4),
            "cube_final_z_m": round(cube_z, 4),
            "cube_expected_z_m": CUBE_SIZE / 2,
            "cube_on_ground": cube_on_ground,
        }
        print(f"[smoke] {args.steps} steps: {wall_s:.2f}s, RTF {sim_s / wall_s:.2f}", flush=True)

        # ------------------------------------------------------------
        # 5. 最後のカメラ画像を PNG で保存
        # ------------------------------------------------------------
        if camera_sensor is not None:
            from PIL import Image

            rgb = None
            for _ in range(CAMERA_MAX_WAIT_STEPS):
                rgb, _info = camera_sensor.get_data("rgb")
                if rgb is not None:
                    break
                simulation_app.update()  # まだ画像が無ければ1フレーム進めて待つ
            if rgb is None:
                raise RuntimeError("カメラから RGB 画像を取得できませんでした")
            image = np.ascontiguousarray(rgb.numpy()[:, :, :3], dtype=np.uint8)
            image_path = out_dir / "camera_rgb.png"
            Image.fromarray(image).save(image_path)
            result["camera"] = {
                "image": image_path.name,
                "shape": list(image.shape),
                "mean_pixel": round(float(image.mean()), 2),  # 0 付近なら真っ黒＝描画失敗の疑い
            }
            if image.max() == 0:
                raise RuntimeError("カメラ画像が真っ黒です（レンダリング失敗の疑い）")

        if not cube_on_ground:
            raise RuntimeError(f"立方体が地面上にありません（z={cube_z:.4f} m）。物理が正しく動いていない可能性")

        app_utils.stop()
        result["status"] = "ok"
        exit_code = EXIT_OK
    except Exception as exc:
        # ------------------------------------------------------------
        # 6. 失敗時：エラー内容を result.json に残し、非ゼロで終了
        # ------------------------------------------------------------
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        exit_code = EXIT_FAILED
    finally:
        monitor.stop()
        result["resources"] = monitor.to_dict()
        result["finished_at"] = datetime.now().isoformat(timespec="seconds")
        path = write_result(out_dir, result)
        print(f"[smoke] status={result['status']} -> {path}", flush=True)

    # ------------------------------------------------------------
    # 終了処理：fast_shutdown では close() 内でプロセスが終了するため、
    # 結果の書き出しは必ずこの前に済ませ、終了コードは exit_code で渡す
    # ------------------------------------------------------------
    if simulation_app is not None:
        simulation_app.close(exit_code=exit_code)
    return exit_code


def _safe_version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    sys.exit(main())
