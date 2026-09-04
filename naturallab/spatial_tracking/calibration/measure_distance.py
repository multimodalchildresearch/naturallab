"""Compatibility entry point for automatic calibration verification.

Manual clicks and the former hidden 1.1 distance multiplier have been removed.
Verification now measures automatically detected chessboard spans through a
fixed calibration and never modifies or rescales that calibration.
"""

from __future__ import annotations

import sys
from typing import Optional, Sequence

from .automatic import (
    BoardSpec,
    VerificationRun,
    verify_floor_from_video,
)
from .commands import main as calibration_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    return calibration_main(["verify", *arguments])


__all__ = [
    "BoardSpec",
    "VerificationRun",
    "verify_floor_from_video",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
