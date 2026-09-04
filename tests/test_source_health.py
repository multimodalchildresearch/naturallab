"""Fast checks for Python source files that do not import optional dependencies."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SKIPPED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "venv",
}
STALE_IMPORT_PREFIXES = (
    "motion_tracking",
    "naturallab.object_detection",
)


def python_sources() -> List[Path]:
    """Return repository Python sources while ignoring generated environments."""
    return [
        path
        for path in sorted(REPOSITORY_ROOT.rglob("*.py"))
        if not any(part in SKIPPED_DIRECTORIES for part in path.parts)
    ]


def test_all_python_sources_compile() -> None:
    failures = []

    for path in python_sources():
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except (SyntaxError, UnicodeDecodeError) as error:
            failures.append(f"{path.relative_to(REPOSITORY_ROOT)}: {error}")

    assert not failures, "Python source compilation failed:\n" + "\n".join(failures)


def test_no_stale_internal_imports() -> None:
    failures = []

    for path in python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported_modules = []
            if isinstance(node, ast.Import):
                imported_modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules = [node.module]

            for module in imported_modules:
                if any(
                    module == prefix or module.startswith(f"{prefix}.")
                    for prefix in STALE_IMPORT_PREFIXES
                ):
                    failures.append(
                        f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}: {module}"
                    )

    assert not failures, "Stale internal imports found:\n" + "\n".join(failures)
