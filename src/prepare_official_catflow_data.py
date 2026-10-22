#!/usr/bin/env python3
"""Download and verify the ordered GDSS graph collections used by CatFlow."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from gpt_datasets import (
    CATFLOW_GDSS_GENERIC_ASSETS,
    CATFLOW_GDSS_REVISION,
    load_graph_dataset_bundle,
)
from project_paths import PROJECT_ROOT


DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark_data" / "official_catflow"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url: str, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        actual = sha256_file(destination)
        if actual != expected_sha256:
            raise RuntimeError(
                f"Refusing to overwrite checksum-mismatched asset {destination}: "
                f"received {actual}, expected {expected_sha256}"
            )
        print(f"verified {destination}")
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        urllib.request.urlretrieve(url, temporary)
        actual = sha256_file(temporary)
        if actual != expected_sha256:
            raise RuntimeError(
                f"Checksum mismatch for {url}: received {actual}, "
                f"expected {expected_sha256}"
            )
        os.replace(temporary, destination)
        print(f"downloaded {destination}")
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    print(f"GDSS revision: {CATFLOW_GDSS_REVISION}")
    for spec in CATFLOW_GDSS_GENERIC_ASSETS.values():
        download_verified(
            spec["url"],
            output_dir / spec["filename"],
            spec["sha256"],
        )

    for dataset in ("EGO_SMALL", "COMMUNITY_SMALL", "ENZYMES", "GRID"):
        bundle = load_graph_dataset_bundle(
            dataset,
            data_root=output_dir,
            paper_split_protocol="catflow_gdss",
        )
        print(
            f"{dataset}: train={len(bundle.train_graphs)} "
            f"val-view={len(bundle.val_graphs)} test={len(bundle.test_graphs)}"
        )


if __name__ == "__main__":
    main()
