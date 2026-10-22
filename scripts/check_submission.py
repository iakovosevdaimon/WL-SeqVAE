#!/usr/bin/env python3
"""Fail when a source package contains common anonymity or artifact leaks."""

from __future__ import annotations

import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
FORBIDDEN_TOP_LEVEL = {
    "benchmark_data",
    "benchmark_runs",
    "checkpoints",
    "inference_runs",
    "trace_cache",
    "visualization_runs",
}
TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
LEAK_PATTERNS = {
    "local Unix home path": re.compile(r"/home/[^/\s]+/"),
    "local macOS home path": re.compile(r"/Users/[^/\s]+/"),
    "local data mount": re.compile(
        r"(?<![A-Za-z0-9.:])/(?:data|datadisk\d*)/[^/\s]+/"
    ),
    "local Windows profile": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
    "email address": re.compile(
        r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    ),
}
MAX_FILE_BYTES = 5 * 1024 * 1024


def iter_text_files():
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or SKIP_PARTS.intersection(path.parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {".gitignore"}:
            yield path


def main() -> int:
    failures: list[str] = []

    for name in sorted(FORBIDDEN_TOP_LEVEL):
        if (REPO_ROOT / name).exists():
            failures.append(f"generated directory is present: {name}")

    for path in REPO_ROOT.rglob("*"):
        if path.is_file() and path.stat().st_size > MAX_FILE_BYTES:
            relative = path.relative_to(REPO_ROOT)
            failures.append(f"file exceeds 5 MiB: {relative}")

    for path in iter_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(REPO_ROOT)
        for label, pattern in LEAK_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                if line_end < 0:
                    line_end = len(text)
                if label != "email address" and "://" in text[line_start:line_end]:
                    continue
                failures.append(f"{relative}:{line}: {label}")

    if failures:
        print("Submission check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("Submission check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
