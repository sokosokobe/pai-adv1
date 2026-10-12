"""Issue #8：Isaac Sim の起動時間を、対策ごとに測る。

対策（--mode）:
    baseline       何もしない（標準の experience）
    stub_torch     SimulationApp の前に、中身のない torch のモジュールを sys.modules に登録する
                   （torch の DLL がスマート アプリ コントロールに阻止され、読み込みのたびに待たされるのを避ける）
    no_deprecated  非推奨の拡張機能（extsDeprecated/ 配下）を除いた experience で起動する
    both           stub_torch と no_deprecated の両方

起動の後、これまでの検証で使った API（シーンの作成、物理の開始、CameraSensor の画像の取得）が動くかも確認する。

使い方::

    .venv\\Scripts\\python.exe scripts\\experiments\\startup_time.py --mode baseline

出力：outputs/m0_8/startup.jsonl に 1 行追記（時間の内訳、有効になった非推奨の拡張機能、確認の結果）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path

T0 = time.perf_counter()
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))  # experiments.* を import するため

from experiments.startup_common import deprecated_extensions, write_experience_without_deprecated  # noqa: E402

MODES = ("baseline", "stub_torch", "no_deprecated", "both")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--trace-torch", action="store_true", help="最初に torch を import した呼び出し元を記録する")
    args = parser.parse_args()
    out_dir = REPO_ROOT / "outputs" / "m0_8"
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    row: dict = {"mode": args.mode, "python_start_s": time.perf_counter() - T0}

    if args.trace_torch:
        import traceback as tb

        class _TorchTracer:
            """最初に torch が探されたときの呼び出し元を記録する（sys.meta_path に入れる。読み込みは通常どおり）。"""

            def find_spec(self, name, path=None, target=None):
                if name == "torch" and "torch_import_stack" not in row:
                    frames = [f for f in tb.extract_stack()[:-1] if "importlib" not in f.filename]
                    row["torch_import_stack"] = [f"{f.filename}:{f.lineno}" for f in frames][-8:]
                return None

        sys.meta_path.insert(0, _TorchTracer())
    if args.mode in ("stub_torch", "both"):
        sys.modules["torch"] = types.ModuleType("torch")  # import torch がすぐに（中身のないモジュールで）成功する

    t = time.perf_counter()
    import isaacsim  # noqa: F401  （パッケージの初期化：環境変数 EXP_PATH などの設定）
    from isaacsim import SimulationApp

    row["import_isaacsim_s"] = time.perf_counter() - t
    isaac_root = Path(isaacsim.__file__).parent
    experience = ""
    if args.mode in ("no_deprecated", "both"):
        experience = str(write_experience_without_deprecated(isaac_root, REPO_ROOT / "build" / "kit_apps"))
    row["experience"] = experience or "(default)"

    t = time.perf_counter()
    simulation_app = SimulationApp({"headless": True}, experience=experience)
    row["simulation_app_s"] = time.perf_counter() - t
    exit_code = 1
    try:
        import omni.kit.app

        manager = omni.kit.app.get_app().get_extension_manager()
        row["deprecated_enabled"] = sorted(
            n for n in deprecated_extensions(isaac_root) if manager.is_extension_enabled(n)
        )
        # 起動の途中で本物の torch が読み込まれたか（スマート アプリ コントロールに阻止されたのは torch の DLL）
        row["torch_loaded_at_startup"] = getattr(sys.modules.get("torch"), "__file__", None) is not None

        # これまでの検証で使った API が動くかの確認（#2〜#6）
        t = time.perf_counter()
        import isaacsim.core.experimental.utils.app as app_utils
        import numpy as np
        from isaacsim.core.experimental.objects import Cube, DomeLight, GroundPlane
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera

        GroundPlane("/World/ground")
        DomeLight("/World/light").set_intensities(1000)
        Cube("/World/cube", sizes=0.2, positions=[[0.0, 0.0, 0.1]], colors="red")
        sensor = CameraSensor(
            RtxCamera("/World/cam", positions=[[0.0, 0.0, 2.0]], orientations=[[1.0, 0.0, 0.0, 0.0]]),
            resolution=(240, 320),
            annotators=["rgb"],
        )
        SimulationManager.setup_simulation(dt=1.0 / 60.0)
        app_utils.play()
        rgb = None
        for _ in range(60):
            simulation_app.update()
            rgb, _ = sensor.get_data("rgb")
            if rgb is not None:
                break
        row["first_scene_s"] = time.perf_counter() - t
        img = None if rgb is None else rgb.numpy()
        # 画像の中央（赤い立方体）が赤いか
        row["camera_ok"] = bool(img is not None and img[120, 160, 0] > 100 > img[120, 160, 1])
        app_utils.enable_extension("isaacsim.robot_motion.cumotion")  # #3・#4 で使った拡張機能
        app_utils.enable_extension("isaacsim.robot_motion.examples")
        simulation_app.update()
        from isaacsim.robot_motion.cumotion import RmpFlowController  # noqa: F401
        from isaacsim.robot_motion.examples.manipulation.controllers import PickPlaceController  # noqa: F401

        row["cumotion_import_ok"] = True
        row["np_version"] = np.__version__
        row["status"] = "ok"
        exit_code = 0
    except Exception as exc:
        traceback.print_exc()
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["total_s"] = time.perf_counter() - T0
        with (out_dir / "startup.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"[startup] {args.mode}: total={row['total_s']:.1f}s app={row.get('simulation_app_s', 0):.1f}s "
            f"deprecated_enabled={len(row.get('deprecated_enabled', []))} status={row.get('status')}",
            flush=True,
        )
    simulation_app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
