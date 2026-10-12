"""Issue #8 Isaac Sim の起動時間の検証用の処理（Isaac Sim を import しない）。

Isaac Sim の標準の experience（`apps/isaacsim.exp.base.kit`）は、非推奨の拡張機能（`extsDeprecated/` 配下）も
読み込む。ここでは、それらを除いた experience のファイルを、インストール先の標準のファイルから作る。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path

_DEP_LINE = re.compile(r'^\s*"([^"]+)"\s*=')


def extension_name(folder_name: str) -> str:
    """拡張機能のフォルダ名（例：isaacsim.core.api-4.12.3）から、バージョンを除いた名前を返す。"""
    return re.sub(r"-\d.*$", "", folder_name)


def rewrite_kit(text: str, *, isaac_root: str, exclude: Iterable[str], rename: Mapping[str, str] | None = None) -> str:
    """experience（.kit）の文字列を書き換える。

    - [dependencies] の中の、exclude に含まれる拡張機能の行を消す。rename に含まれる名前は置き換える
    - `${app}/../` を Isaac Sim のインストール先（isaac_root）の絶対パスにする。単独の `${app}`（このファイルの
      置き場所）はそのまま残す
    """
    exclude = set(exclude)
    rename = dict(rename or {})
    root = isaac_root.replace("\\", "/").rstrip("/")
    out, in_deps = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_deps = stripped == "[dependencies]"
        elif in_deps:
            m = _DEP_LINE.match(line)
            if m:
                name = m.group(1)
                if name in exclude:
                    continue
                if name in rename:
                    line = line.replace(f'"{name}"', f'"{rename[name]}"', 1)
        out.append(line.replace("${app}/../", f"{root}/"))
    return "\n".join(out) + "\n"


def deprecated_extensions(isaac_root: Path) -> set[str]:
    """`extsDeprecated/` 配下の拡張機能の名前。"""
    return {extension_name(d.name) for d in (isaac_root / "extsDeprecated").iterdir() if d.is_dir()}


def write_experience_without_deprecated(isaac_root: Path, out_dir: Path) -> Path:
    """非推奨の拡張機能を除いた experience を out_dir に作り、SimulationApp に渡すファイルのパスを返す。

    - restorebot.exp.base.kit：isaacsim.exp.base.kit から非推奨の拡張機能を除いたもの
    - restorebot.exp.python.kit：isaacsim.exp.base.python.kit の依存先を restorebot.exp.base に替えたもの
    """
    apps = isaac_root / "apps"
    out_dir.mkdir(parents=True, exist_ok=True)
    root = str(isaac_root)
    exclude = deprecated_extensions(isaac_root)
    base = rewrite_kit((apps / "isaacsim.exp.base.kit").read_text(encoding="utf-8"), isaac_root=root, exclude=exclude)
    python = rewrite_kit(
        (apps / "isaacsim.exp.base.python.kit").read_text(encoding="utf-8"),
        isaac_root=root,
        exclude=(),
        rename={"isaacsim.exp.base": "restorebot.exp.base"},
    )
    (out_dir / "restorebot.exp.base.kit").write_text(base, encoding="utf-8")
    path = out_dir / "restorebot.exp.python.kit"
    path.write_text(python, encoding="utf-8")
    return path
