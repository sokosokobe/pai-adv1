"""scripts/experiments/startup_common.py（Isaac Sim の起動時間の検証用の処理）の単体テスト。Isaac Sim 不要。"""

from experiments.startup_common import extension_name, rewrite_kit

BASE_KIT = """[package]
title = "Isaac Sim Base"

[dependencies]
# Isaac Sim extensions
"isaacsim.common" = { order = -100 }
"isaacsim.core.api" = {}
"isaacsim.core.experimental.prims" = {}
"isaacsim.sensors.camera" = {}

[settings]
versionFile = "${app}/../VERSION"

[settings.app.exts.folders]
'++' = [
    "${app}",
    "${app}/../exts",
    "${app}/../extsDeprecated",
]
"""


def test_extension_name_strips_version():
    assert extension_name("isaacsim.core.api-4.12.3") == "isaacsim.core.api"
    assert extension_name("omni.videoencoding-0.2.1+00c488ae.wx64.r.cp312") == "omni.videoencoding"
    assert extension_name("isaacsim.core.api") == "isaacsim.core.api"


def test_rewrite_kit_drops_excluded_dependencies_only():
    out = rewrite_kit(BASE_KIT, isaac_root="C:/isaac", exclude={"isaacsim.core.api", "isaacsim.sensors.camera"})
    assert '"isaacsim.core.api"' not in out
    assert '"isaacsim.sensors.camera"' not in out
    assert '"isaacsim.common" = { order = -100 }' in out
    assert '"isaacsim.core.experimental.prims" = {}' in out


def test_rewrite_kit_makes_parent_paths_absolute_and_keeps_app():
    out = rewrite_kit(BASE_KIT, isaac_root="C:/isaac", exclude=set())
    assert 'versionFile = "C:/isaac/VERSION"' in out
    assert '"C:/isaac/exts",' in out
    assert '"${app}",' in out  # 生成したファイルのフォルダ（依存する experience を探す場所）はそのまま


def test_rewrite_kit_renames_dependency():
    kit = '[dependencies]\n"isaacsim.exp.base" = {}\n\n[settings]\nx = 1\n'
    out = rewrite_kit(kit, isaac_root="C:/isaac", exclude=set(), rename={"isaacsim.exp.base": "restorebot.exp.base"})
    assert '"restorebot.exp.base" = {}' in out
    assert "isaacsim.exp.base" not in out


def test_rewrite_kit_ignores_keys_outside_dependencies():
    kit = '[dependencies]\n"a.b" = {}\n\n[settings.exts."a.b"]\nenabled = true\n'
    out = rewrite_kit(kit, isaac_root="C:/isaac", exclude={"a.b"})
    assert '[settings.exts."a.b"]' in out and "enabled = true" in out
    assert '"a.b" = {}' not in out
