"""Pytest configuration.

We add the repository root to sys.path so tests can import top-level folders
like `examples/` without requiring them to be installed as packages.

This is especially helpful in environments where pytest is configured to use
importlib-based imports.
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
