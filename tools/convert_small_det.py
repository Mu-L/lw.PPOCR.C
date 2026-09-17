#!/usr/bin/env python3
"""Convert the pinned PP-OCRv6 Small DET graph for a production pack."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.convert_small_det_experimental import main


if __name__ == "__main__":
    # Keep the stable entry point production-only even if a caller copied the
    # experimental --runtime-status switch into its command line.
    raise SystemExit(main([*sys.argv[1:], "--runtime-status", "production"]))
