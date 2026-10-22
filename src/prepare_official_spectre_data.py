#!/usr/bin/env python3
"""Download and verify the official SPECTRE graph assets used by DiGress."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from datetime import datetime
from pathlib import Path

from gpt_datasets import (
    DIGRESS_MAIN_SPLIT_REVISION,
    DIGRESS_PAPER_SPLIT_REVISION,
    SPECTRE_DD_RAW_ASSETS,
    SPECTRE_OFFICIAL_ASSETS,
    SPECTRE_SOURCE_REVISION,
    load_graph_dataset_bundle,
)
from project_paths import PROJECT_ROOT, portable_path
from ring_traces import graph_collection_fingerprint


DEFAULT_OUTPUT = PROJECT_ROOT / "benchmark_data" / "official_spectre"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    out.add_argument(
        "--datasets",
        nargs="*",
        default=sorted(set(SPECTRE_OFFICIAL_ASSETS) | {"PROTEINS"}),
    )
    out.add_argument("--check-only", action="store_true")
    return out


def ensure_asset(output_dir: Path, dataset: str, *, check_only: bool) -> dict:
    if dataset not in SPECTRE_OFFICIAL_ASSETS:
        raise ValueError(f"No official serialized SPECTRE asset is registered for {dataset}")
    spec = SPECTRE_OFFICIAL_ASSETS[dataset]
    target = output_dir / spec["filename"]
    actual = sha256_file(target) if target.is_file() else None
    if actual != spec["sha256"]:
        if check_only:
            state = "missing" if actual is None else f"checksum_mismatch:{actual}"
            raise RuntimeError(f"{dataset}: {state}: {target}")
        output_dir.mkdir(parents=True, exist_ok=True)
        partial = output_dir / f".{spec['filename']}.partial"
        partial.unlink(missing_ok=True)
        print(f"Downloading {dataset}: {spec['url']}", flush=True)
        try:
            urllib.request.urlretrieve(spec["url"], partial)
            received = sha256_file(partial)
            if received != spec["sha256"]:
                raise RuntimeError(
                    f"Checksum mismatch for {dataset}: received {received}, expected {spec['sha256']}"
                )
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)
        actual = sha256_file(target)
    print(f"Verified {dataset}: {target} sha256={actual}", flush=True)
    return {
        "dataset": dataset,
        "path": portable_path(target),
        "url": spec["url"],
        "sha256": actual,
        "bytes": target.stat().st_size,
    }


def ensure_dd_assets(output_dir: Path, *, check_only: bool) -> dict:
    dd_dir = output_dir / "DD"
    records = []
    for filename, spec in SPECTRE_DD_RAW_ASSETS.items():
        target = dd_dir / filename
        actual = sha256_file(target) if target.is_file() else None
        if actual != spec["sha256"]:
            if check_only:
                state = "missing" if actual is None else f"checksum_mismatch:{actual}"
                raise RuntimeError(f"PROTEINS/{filename}: {state}: {target}")
            dd_dir.mkdir(parents=True, exist_ok=True)
            partial = dd_dir / f".{filename}.partial"
            partial.unlink(missing_ok=True)
            print(f"Downloading PROTEINS/{filename}: {spec['url']}", flush=True)
            try:
                urllib.request.urlretrieve(spec["url"], partial)
                received = sha256_file(partial)
                if received != spec["sha256"]:
                    raise RuntimeError(
                        f"Checksum mismatch for PROTEINS/{filename}: "
                        f"received {received}, expected {spec['sha256']}"
                    )
                os.replace(partial, target)
            finally:
                partial.unlink(missing_ok=True)
            actual = sha256_file(target)
        print(f"Verified PROTEINS/{filename}: {target} sha256={actual}", flush=True)
        records.append({
            "path": portable_path(target),
            "url": spec["url"],
            "sha256": actual,
            "bytes": target.stat().st_size,
        })
    return {"dataset": "PROTEINS", "files": records}


def main() -> None:
    args = parser().parse_args()
    datasets = [str(value).upper() for value in args.datasets]
    unsupported = sorted(set(datasets) - set(SPECTRE_OFFICIAL_ASSETS) - {"PROTEINS"})
    if unsupported:
        raise SystemExit(f"Unsupported official SPECTRE datasets: {unsupported}")
    records = [
        ensure_dd_assets(args.output_dir, check_only=args.check_only)
        if dataset == "PROTEINS"
        else ensure_asset(args.output_dir, dataset, check_only=args.check_only)
        for dataset in datasets
    ]

    split_audit = {}
    for dataset in datasets:
        split_audit[dataset] = {}
        protocols = ("spectre_seed1234",) if dataset == "PROTEINS" else (
            "digress_seed0", "spectre_seed1234"
        )
        for protocol in protocols:
            bundle = load_graph_dataset_bundle(
                dataset,
                data_root=args.output_dir,
                paper_split_protocol=protocol,
            )
            split_audit[dataset][protocol] = {
                "train": len(bundle.train_graphs),
                "val": len(bundle.val_graphs),
                "test": len(bundle.test_graphs),
                "train_fingerprint": graph_collection_fingerprint(bundle.train_graphs),
                "val_fingerprint": graph_collection_fingerprint(bundle.val_graphs),
                "test_fingerprint": graph_collection_fingerprint(bundle.test_graphs),
                "provenance": bundle.provenance,
            }

    manifest = {
        "schema_version": 2,
        "created": datetime.now().astimezone().isoformat(),
        "runtime": {"torch_version": str(__import__("torch").__version__)},
        "upstream_revisions": {
            "spectre_assets": SPECTRE_SOURCE_REVISION,
            "digress_current_main_split": DIGRESS_MAIN_SPLIT_REVISION,
            "digress_paper_era_split": DIGRESS_PAPER_SPLIT_REVISION,
        },
        "assets": records,
        "split_audit": split_audit,
    }
    manifest_path = args.output_dir / "official_spectre_manifest.json"
    if not args.check_only:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(split_audit, indent=2, sort_keys=True))
    if args.check_only:
        print(f"Check-only passed; prospective manifest path: {manifest_path}")
    else:
        print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
