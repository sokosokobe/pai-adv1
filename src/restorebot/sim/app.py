"""SimulationApp の起動（headless が既定）。

SimulationApp を最初に生成し、ほかの Isaac Sim のモジュール（restorebot.sim の他のファイルを含む）は
その後で import する。
"""

from __future__ import annotations

import os


def start_app(*, headless: bool = True):
    """SimulationApp を起動して返す。EULA は環境変数 OMNI_KIT_ACCEPT_EULA=YES で受け入れる。"""
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp

    return SimulationApp({"headless": headless})
