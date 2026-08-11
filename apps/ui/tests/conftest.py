"""Pytest config that adds apps/ui/src to sys.path so `import ui.*` works."""

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
SRC = APP_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
