#!/usr/bin/env python3
"""Source-checkout wrapper for ``naturallab calibrate``.

The implementation lives in the installed package so this script and the
console command cannot diverge.
"""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from naturallab.spatial_tracking.calibration.commands import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
