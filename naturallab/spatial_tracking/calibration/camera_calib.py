"""Compatibility entry point for automatic intrinsic calibration.

The former GUI required researchers to choose frames with the space bar.  That
workflow was removed because it was subjective and its reported error was not
the OpenCV RMS.  Use ``naturallab calibrate intrinsic`` for new work.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from .automatic import (
    BoardSpec,
    IntrinsicCalibrationRun,
    calibrate_intrinsics_from_video,
)
from .commands import main as calibration_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    return calibration_main(["intrinsic", *arguments])


__all__ = [
    "BoardSpec",
    "IntrinsicCalibrationRun",
    "calibrate_intrinsics_from_video",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
