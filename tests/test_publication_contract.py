"""Public-release contracts for documentation and reusable examples."""

from __future__ import annotations

from pathlib import Path
import re
from urllib.parse import urlparse


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PUBLIC_FILES = (
    "README.md",
    "LICENSE",
    "docs/quickstart.md",
    "docs/lab_setup_guide.md",
    "docs/object_detection_guide.md",
    "docs/calibration_workflow.md",
    "docs/researcher_workflow.md",
    "examples/study_manifest.yaml",
    "examples/shared_board_extrinsics.yaml",
)

PROHIBITED_DOCUMENTATION_PHRASES = (
    "millisecond-precision synchronization",
    "comparable to commercial motion capture systems",
    "100% cross-view tracking accuracy",
    "four-camera volumetric validation passed",
    "provisional multi-view triangulation",
)

PROHIBITED_LEGACY_MODULES = (
    "naturallab/utils/granularity.py",
    "naturallab/gaze_analysis/object_detection/two_stage.py",
    "naturallab/spatial_tracking/tracking/track_identity_matching.py",
)

CLONE_COMMAND_PATTERN = re.compile(
    r"(?im)^\s*git\s+clone\s+(https://github\.com/[^\s`]+)"
)
WINDOWS_ABSOLUTE_PATH_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")
PRIVATE_PATH_MARKERS = (
    "/Users/",
    "/home/",
    "/pfss/",
    "\\Users\\",
    ".codex/attachments/",
    "file://",
)


def _documentation_files() -> list[Path]:
    files = [REPOSITORY_ROOT / "README.md"]
    files.extend(sorted((REPOSITORY_ROOT / "docs").rglob("*.md")))
    files.extend(sorted((REPOSITORY_ROOT / "examples").rglob("*.md")))
    return files


def _public_text_files() -> list[Path]:
    files = _documentation_files()
    for suffix in ("*.yaml", "*.yml", "*.json"):
        files.extend(sorted((REPOSITORY_ROOT / "examples").rglob(suffix)))
    return files


def test_required_public_docs_and_examples_exist() -> None:
    missing = [
        relative_path
        for relative_path in REQUIRED_PUBLIC_FILES
        if not (REPOSITORY_ROOT / relative_path).is_file()
    ]

    assert not missing, "Missing public files: " + ", ".join(missing)


def test_study_specific_legacy_modules_are_not_shipped() -> None:
    present = [
        relative_path
        for relative_path in PROHIBITED_LEGACY_MODULES
        if (REPOSITORY_ROOT / relative_path).exists()
    ]

    assert not present, "Study-specific legacy modules remain: " + ", ".join(
        present
    )


def test_extrinsics_report_source_does_not_claim_triangulation_support() -> None:
    source = (
        REPOSITORY_ROOT
        / "naturallab/spatial_tracking/calibration/extrinsics.py"
    ).read_text(encoding="utf-8")

    assert "provisional multi-view triangulation" not in source
    assert "general-purpose multi-view point or skeleton triangulation" in source
    assert '"volumetric_validated": False' in source


def test_documentation_contains_a_non_anonymous_public_clone_url() -> None:
    clone_urls: list[str] = []
    for path in _documentation_files():
        clone_urls.extend(CLONE_COMMAND_PATTERN.findall(path.read_text("utf-8")))

    assert clone_urls, "Documentation must include an HTTPS GitHub clone command"
    invalid_urls = []
    for url in clone_urls:
        parsed = urlparse(url.removesuffix(".git"))
        path_parts = [part for part in parsed.path.split("/") if part]
        normalized = url.casefold()
        if (
            parsed.scheme != "https"
            or parsed.netloc.casefold() != "github.com"
            or len(path_parts) != 2
            or any(
                marker in normalized
                for marker in (
                    "anonymous",
                    "example",
                    "placeholder",
                    "username",
                    "your-org",
                    "your-repo",
                    "<",
                    ">",
                )
            )
        ):
            invalid_urls.append(url)

    assert not invalid_urls, "Anonymous or placeholder clone URLs: " + ", ".join(
        invalid_urls
    )


def test_documentation_avoids_unsupported_affirmative_overclaims() -> None:
    failures = []
    for path in _documentation_files():
        text = path.read_text("utf-8").casefold()
        for phrase in PROHIBITED_DOCUMENTATION_PHRASES:
            if phrase.casefold() in text:
                failures.append(
                    f"{path.relative_to(REPOSITORY_ROOT)}: {phrase!r}"
                )

    assert not failures, "Unsupported documentation claims:\n" + "\n".join(failures)


def test_public_docs_and_examples_do_not_embed_private_paths() -> None:
    failures = []
    for path in _public_text_files():
        for line_number, line in enumerate(
            path.read_text("utf-8").splitlines(),
            start=1,
        ):
            if WINDOWS_ABSOLUTE_PATH_PATTERN.match(line.strip()) or any(
                marker.casefold() in line.casefold()
                for marker in PRIVATE_PATH_MARKERS
            ):
                failures.append(
                    f"{path.relative_to(REPOSITORY_ROOT)}:{line_number}"
                )

    assert not failures, "Private paths in public files: " + ", ".join(failures)
