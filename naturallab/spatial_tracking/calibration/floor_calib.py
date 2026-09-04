"""Compatibility entry point for automatic floor-plane calibration.

The former GUI manually accepted positions and reconstructed corners by
scaling rays with ``t_z``.  The supported path detects stationary placements
automatically and uses the complete PnP transform ``R @ X + t``.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from .automatic import (
    BoardSpec,
    FloorCalibrationRun,
    calibrate_floor_from_video,
)
from .commands import main as calibration_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    return calibration_main(["floor", *arguments])


__all__ = [
    "BoardSpec",
    "FloorCalibrationRun",
    "calibrate_floor_from_video",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
