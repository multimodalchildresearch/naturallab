"""Compatibility entry point for the supported automatic floor calibration."""

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
