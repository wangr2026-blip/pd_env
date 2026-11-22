"""Shared pipeline infrastructure: config, caching, folds and aggregation.

Named ``lib`` rather than ``core`` because the repository .gitignore excludes
any path called ``core`` (core-dump rule).
"""
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
# Reuse the repository's existing helpers (atomic table/JSON writes, FFmpeg
# decoding) instead of duplicating them. Same convention as tests/conftest.py.
for _path in (REPO, REPO / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
