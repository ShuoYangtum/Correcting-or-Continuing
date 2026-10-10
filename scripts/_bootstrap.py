"""Make `pssd` importable when the package is not pip-installed.

Every script imports this module first, so `python scripts/run_*.py` works
straight from a fresh checkout (no `pip install -e .` required).
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
