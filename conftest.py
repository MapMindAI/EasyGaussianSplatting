"""Puts the repository root on sys.path so `gsplat_server` and `mapping` import."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
