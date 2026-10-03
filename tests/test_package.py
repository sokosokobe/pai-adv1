"""restorebot パッケージの骨格と、Isaac Sim 依存の分離ルールのテスト（Isaac Sim 不要）。"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# core から読み込まれてはいけない Isaac Sim / Omniverse 系のトップレベルモジュール
ISAAC_MODULES = ("isaacsim", "omni", "pxr", "carb")


def test_core_imports_without_isaac_sim():
    # 別プロセスで import し、sys.modules に Isaac Sim 系が入っていないことを確認する
    code = (
        "import sys, restorebot.core; "
        f"bad = sorted(m for m in sys.modules if m.split('.')[0] in {ISAAC_MODULES!r}); "
        "print(','.join(bad))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8", cwd=REPO_ROOT)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", f"core が Isaac Sim 系を import している: {proc.stdout}"


def test_sim_package_exists():
    import restorebot.sim  # noqa: F401  （空の骨格。Isaac Sim はまだ import しない）


def _ruff_check(source: str, virtual_path: str) -> subprocess.CompletedProcess:
    """ruff に、指定したパスにあるファイルとして source を lint させる。"""
    return subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--stdin-filename", virtual_path, "-"],
        input=source,
        capture_output=True,
        text=True,
        encoding="utf-8",  # ruff のメッセージは UTF-8（Windows の既定 cp932 では読めない）
        cwd=REPO_ROOT,
    )


@pytest.mark.parametrize("module", ISAAC_MODULES)
@pytest.mark.parametrize(
    "virtual_path",
    ["src/restorebot/core/example.py", "src/restorebot/example.py", "tests/test_example.py"],
)
def test_lint_bans_isaac_imports_outside_sim(module, virtual_path):
    sources = (
        f"import {module}\n\nprint({module})\n",
        f"from {module} import x\n\nprint(x)\n",
        # 実際のコードではサブモジュールから import することが多い
        f"import {module}.sub.mod\n\nprint({module})\n",
        f"from {module}.sub import x\n\nprint(x)\n",
    )
    for source in sources:
        proc = _ruff_check(source, virtual_path)
        assert proc.returncode != 0, f"{virtual_path} で {source!r} が許可されてしまった"
        assert "TID251" in proc.stdout, proc.stdout


@pytest.mark.parametrize("module", ISAAC_MODULES)
@pytest.mark.parametrize(
    "virtual_path",
    ["src/restorebot/sim/example.py", "scripts/example.py", "scripts/experiments/example.py", "tests/sim/test_x.py"],
)
def test_lint_allows_isaac_imports_in_sim_layers(module, virtual_path):
    proc = _ruff_check(f"import {module}\n\nprint({module})\n", virtual_path)
    assert proc.returncode == 0, proc.stdout
