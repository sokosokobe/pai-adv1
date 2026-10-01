"""pytest 共通設定。"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# scripts/ 配下のモジュール（smoke_test など）を import できるようにする
sys.path.insert(0, str(REPO_ROOT / "scripts"))
