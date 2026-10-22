"""Repository-relative paths shared by command-line tools.

Keeping path resolution here prevents launchers from depending on a submitter's
working directory or local filesystem layout.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_ROOT = PROJECT_ROOT / "datasets"


def portable_path(value: str | Path | None) -> str | None:
    """Return repository-relative metadata without exposing a local directory."""
    if value is None:
        return None
    resolved = Path(value).expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return f"<external>/{resolved.name}"
