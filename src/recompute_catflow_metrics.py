#!/usr/bin/env python3
"""Recompute CatFlow main-table graph metrics and audit comparability.

The CatFlow paper cites the GDSS ordered train/test split for Ego-small,
Community-small, Enzymes, and Grid.  Its main table reports raw biased MMD^2
for degree, clustering, and exact four-node ORCA orbit features.  Degree and
clustering use Gaussian-EMD kernels; orbit features use a Gaussian kernel.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from gpt_datasets import load_graph_dataset_bundle
from project_paths import PROJECT_ROOT, portable_path
from recompute_digress_spectre_metrics import (
    OrbitCache,
    clustering_features,
    compute_mmd2,
    degree_features,
    extract_generated_graphs,
    json_number,
    read_json,
    simple_graph,
)


DEFAULT_RUN_ROOT = PROJECT_ROOT / "benchmark_runs"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "benchmark_data" / "official_catflow"

DATASETS = ("EGO_SMALL", "COMMUNITY_SMALL", "ENZYMES", "GRID")
EXPECTED_SAMPLES = {
    "EGO_SMALL": 40,
    "COMMUNITY_SMALL": 20,
    "ENZYMES": 117,
    "GRID": 20,
}
PAPER_RESULTS = {
    "EGO_SMALL": {
        "degree": (0.013, 0.007),
        "clustering": (0.024, 0.009),
        "orbit": (0.008, 0.005),
    },
    "COMMUNITY_SMALL": {
        "degree": (0.018, 0.012),
        "clustering": (0.086, 0.021),
        "orbit": (0.007, 0.005),
    },
    "ENZYMES": {
        "degree": (0.013, 0.012),
        "clustering": (0.062, 0.011),
        "orbit": (0.008, 0.007),
    },
    "GRID": {
        "degree": (0.115, 0.010),
        "clustering": (0.004, 0.002),
        "orbit": (0.075, 0.071),
    },
}
ROLE_NAMES = {
    "structural_validation_best": "structural-validation best",
    "fixed_beta_best": "fixed-beta best",
    "legacy_val_loss_best": "legacy-val best",
    "last_epoch": "last epoch",
}
ROLE_ORDER = {
    "structural_validation_best": 0,
    "fixed_beta_best": 1,
    "legacy_val_loss_best": 2,
    "last_epoch": 3,
}


@dataclass(frozen=True)
class Run:
    dataset: str
    wl: int
    role: str
    metric_file: Path
    run_config_file: Path
    sample_file: Path


@dataclass
class Result:
    dataset: str
    wl: int
    checkpoint_role: str
    checkpoint_epoch: int | None
    samples: int
    target_samples: int
    complete_pct: float
    degree_mmd2: float
    clustering_mmd2: float
    orbit_mmd2: float
    directly_catflow_comparable: bool
    comparability_failures: list[str]
    reference_split: str
    paper_split_protocol: str
    sample_file: str
    original_metric_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument(
        "--orbit-cache",
        type=Path,
        default=None,
        help="Optional shared ORCA feature cache; defaults to OUTPUT_DIR/orbit_cache.sqlite3.",
    )
    parser.add_argument(
        "--orca",
        type=Path,
        default=PROJECT_ROOT / "third_party" / "orca" / "orca",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--datasets", nargs="*", choices=DATASETS, default=None)
    return parser.parse_args()


def role_from_metrics(metric_file: Path, metrics: dict) -> str | None:
    embedded = str(metrics.get("config/checkpoint_role") or "")
    normalized = {
        "legacy_warmup_weighted_val_loss_best": "legacy_val_loss_best",
        "legacy_val_loss_best": "legacy_val_loss_best",
        "fixed_beta_best": "fixed_beta_best",
        "structural_validation_best": "structural_validation_best",
        "last_epoch": "last_epoch",
    }.get(embedded)
    if normalized:
        return normalized
    variant = metric_file.parents[1].name
    if "legacy_val_loss_best" in variant:
        return "legacy_val_loss_best"
    if "last_epoch" in variant:
        return "last_epoch"
    if variant.endswith("_allow_incomplete"):
        return "fixed_beta_best"
    return None


def scan_runs(run_root: Path, selected: set[str] | None) -> list[Run]:
    runs = []
    for metric_file in sorted(run_root.rglob("samples/metrics.json")):
        if not metric_file.parents[1].name.endswith("_allow_incomplete"):
            continue
        metrics = read_json(metric_file)
        dataset = str(metrics.get("config/dataset") or "")
        if dataset not in DATASETS or (selected and dataset not in selected):
            continue
        role = role_from_metrics(metric_file, metrics)
        if role is None:
            continue
        sample_file = metric_file.parent / "ring_vae_samples.pt"
        run_config_file = metric_file.parent / "run_config.json"
        if not sample_file.is_file() or not run_config_file.is_file():
            continue
        runs.append(
            Run(
                dataset=dataset,
                wl=int(
                    metrics.get(
                        "config/requested_n_wl_iters",
                        metrics.get("config/final_stage", 0),
                    )
                ),
                role=role,
                metric_file=metric_file,
                run_config_file=run_config_file,
                sample_file=sample_file,
            )
        )
    return sorted(
        runs,
        key=lambda run: (
            DATASETS.index(run.dataset),
            run.wl,
            ROLE_ORDER[run.role],
            str(run.metric_file),
        ),
    )


def feature_bundle(graphs, cache: OrbitCache, label: str) -> dict:
    return {
        "degree": degree_features(graphs),
        "clustering": clustering_features(graphs),
        "orbit": cache.features(graphs, label),
    }


def compute_metrics(reference: dict, generated: dict) -> dict[str, float]:
    return {
        "degree": compute_mmd2(
            reference["degree"],
            generated["degree"],
            kernel="gaussian_emd",
            sigma=1.0,
            is_hist=True,
            distance_scaling=1.0,
        ),
        "clustering": compute_mmd2(
            reference["clustering"],
            generated["clustering"],
            kernel="gaussian_emd",
            sigma=0.1,
            is_hist=True,
            distance_scaling=100.0,
        ),
        "orbit": compute_mmd2(
            reference["orbit"],
            generated["orbit"],
            kernel="gaussian",
            sigma=30.0,
            is_hist=False,
        ),
    }


def relative(path: Path) -> str:
    return str(portable_path(path))


def comparability_failures(
    run: Run,
    metrics: dict,
    config: dict,
    expected_provenance: dict,
    generated_samples: int,
) -> list[str]:
    failures = []
    if config.get("paper_split_protocol") != "catflow_gdss":
        failures.append(
            f"paper_split_protocol_{config.get('paper_split_protocol') or 'missing'}"
        )
    recorded_provenance = metrics.get("config/dataset_provenance")
    if recorded_provenance != expected_provenance:
        failures.append("checkpoint_dataset_or_split_not_official_catflow_gdss")
    if not config.get("verify_data_provenance", False):
        failures.append("checkpoint_data_provenance_unverified")
    if metrics.get("config/checkpoint_mismatch_warnings"):
        failures.append("checkpoint_mismatch_warnings_present")
    reference_split = str(
        metrics.get("config/reference_split")
        or config.get("reference_split")
        or ""
    )
    if reference_split != "test":
        failures.append(f"reference_split_{reference_split or 'missing'}")
    expected = EXPECTED_SAMPLES[run.dataset]
    target = int(metrics.get("sampling/target_n_samples", generated_samples))
    if target != expected:
        failures.append(f"target_samples_{target}_expected_{expected}")
    if generated_samples != expected:
        failures.append(
            f"generated_samples_{generated_samples}_expected_{expected}"
        )
    return failures


def recompute(
    runs: list[Run],
    data_root: Path,
    cache: OrbitCache,
) -> tuple[list[Result], dict[str, dict]]:
    contexts = {}
    results = []
    for dataset in sorted({run.dataset for run in runs}, key=DATASETS.index):
        bundle = load_graph_dataset_bundle(
            dataset,
            data_root=data_root,
            paper_split_protocol="catflow_gdss",
        )
        reference_graphs = [simple_graph(graph) for graph in bundle.test_graphs]
        contexts[dataset] = {
            "provenance": bundle.provenance,
            "train_graphs": len(bundle.train_graphs),
            "validation_view_graphs": len(bundle.val_graphs),
            "test_graphs": len(bundle.test_graphs),
            "reference_features": feature_bundle(
                reference_graphs,
                cache,
                f"{dataset}/official-test",
            ),
        }

    for index, run in enumerate(runs, start=1):
        print(
            f"[{index}/{len(runs)}] {run.dataset} WL{run.wl} {run.role}",
            flush=True,
        )
        original = read_json(run.metric_file)
        config = read_json(run.run_config_file)
        generated_graphs = extract_generated_graphs(run.sample_file)
        generated_features = feature_bundle(
            generated_graphs,
            cache,
            f"{run.dataset}/WL{run.wl}/{run.role}",
        )
        values = compute_metrics(
            contexts[run.dataset]["reference_features"],
            generated_features,
        )
        failures = comparability_failures(
            run,
            original,
            config,
            contexts[run.dataset]["provenance"],
            len(generated_graphs),
        )
        results.append(
            Result(
                dataset=run.dataset,
                wl=run.wl,
                checkpoint_role=run.role,
                checkpoint_epoch=original.get("config/checkpoint_epoch"),
                samples=len(generated_graphs),
                target_samples=int(
                    original.get("sampling/target_n_samples", len(generated_graphs))
                ),
                complete_pct=100.0
                * float(original.get("sampling/complete_fraction") or 0.0),
                degree_mmd2=values["degree"],
                clustering_mmd2=values["clustering"],
                orbit_mmd2=values["orbit"],
                directly_catflow_comparable=not failures,
                comparability_failures=failures,
                reference_split=str(
                    original.get("config/reference_split")
                    or config.get("reference_split")
                    or ""
                ),
                paper_split_protocol=str(
                    config.get("paper_split_protocol") or ""
                ),
                sample_file=relative(run.sample_file),
                original_metric_file=relative(run.metric_file),
            )
        )
    return results, contexts


def fmt(value: float) -> str:
    return f"{value:.6f}"


def markdown(results: list[Result], contexts: dict, run_root: Path, orca_sha: str) -> str:
    comparable = [result for result in results if result.directly_catflow_comparable]
    lines = [
        "# CatFlow main-table graph metric audit",
        "",
        f"Generated: **{datetime.now().astimezone().isoformat(timespec='minutes')}**",
        "",
        f"Run root: `{relative(run_root)}`",
        "",
        "This audit uses the CatFlow/GDSS ordered 80/20 outer split and reports raw "
        "biased MMD². Degree and clustering use Gaussian-EMD; orbit uses a Gaussian "
        "kernel over exact ORCA four-node orbit features. Values are not normalized "
        "by train/test denominators.",
        "",
        f"Directly CatFlow-comparable rows: **{len(comparable)}/{len(results)}**.",
        "",
        "## Published CatFlow references",
        "",
        "| Dataset | Test graphs | Degree MMD² ↓ | Clustering MMD² ↓ | Orbit MMD² ↓ |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        paper = PAPER_RESULTS[dataset]
        lines.append(
            f"| {dataset} | {EXPECTED_SAMPLES[dataset]} | "
            f"{paper['degree'][0]:.3f} ± {paper['degree'][1]:.3f} | "
            f"{paper['clustering'][0]:.3f} ± {paper['clustering'][1]:.3f} | "
            f"{paper['orbit'][0]:.3f} ± {paper['orbit'][1]:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Recomputed rows",
            "",
            "| Dataset | WL | Checkpoint | Epoch | Samples | Complete | Degree MMD² ↓ | "
            "Clustering MMD² ↓ | Orbit MMD² ↓ | Direct? | Failure(s) |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for result in results:
        failures = ", ".join(result.comparability_failures) or "—"
        lines.append(
            f"| {result.dataset} | {result.wl} | "
            f"{ROLE_NAMES[result.checkpoint_role]} | "
            f"{result.checkpoint_epoch if result.checkpoint_epoch is not None else '—'} | "
            f"{result.samples}/{EXPECTED_SAMPLES[result.dataset]} | "
            f"{result.complete_pct:.2f}% | {fmt(result.degree_mmd2)} | "
            f"{fmt(result.clustering_mmd2)} | {fmt(result.orbit_mmd2)} | "
            f"{'yes' if result.directly_catflow_comparable else 'no'} | {failures} |"
        )

    lines.extend(
        [
            "",
            "## Dataset provenance",
            "",
            "| Dataset | Official train | Validation view | Official test | Asset SHA-256 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for dataset in DATASETS:
        if dataset not in contexts:
            continue
        context = contexts[dataset]
        provenance = context["provenance"]
        lines.append(
            f"| {dataset} | {context['train_graphs']} | "
            f"{context['validation_view_graphs']} | {context['test_graphs']} | "
            f"`{provenance.get('asset_sha256')}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A row is publishable against CatFlow only when its checkpoint was trained "
            "with the same pinned asset and ordered outer split, evaluation uses the "
            "official test set, and the full paper-scale sample count is present.",
            "- `Complete` is a decoder diagnostic; CatFlow's published metric evaluates "
            "the generated graph collection and does not define this project-specific flag.",
            "- The paper reports mean ± standard deviation over multiple seeds but does "
            "not disclose the exact seed list or number of runs. Report the run seeds and "
            "aggregation count explicitly for new comparisons.",
            "",
            f"ORCA executable SHA-256: `{orca_sha}`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    results: list[Result],
    contexts: dict,
    output_dir: Path,
    markdown_path: Path,
    run_root: Path,
    orca_sha: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serializable_contexts = {
        dataset: {
            key: value
            for key, value in context.items()
            if key != "reference_features"
        }
        for dataset, context in contexts.items()
    }
    payload = {
        "protocol": {
            "name": "CatFlow main abstract-graph table",
            "split": "GDSS ordered first 20% test, remaining 80% train",
            "split_seed": None,
            "mmd_estimator": "biased_mmd_squared",
            "degree_kernel": "gaussian_emd_sigma_1",
            "clustering_kernel": "gaussian_emd_sigma_0.1_distance_scaling_100",
            "orbit_kernel": "gaussian_sigma_30",
            "orbit_backend": "ORCA node orbits up to size four, exact",
            "orca_sha256": orca_sha,
        },
        "contexts": serializable_contexts,
        "paper_results": PAPER_RESULTS,
        "results": [asdict(result) for result in results],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_number) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(asdict(results[0]).keys()) if results else [
            field.name for field in Result.__dataclass_fields__.values()
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["comparability_failures"] = ";".join(result.comparability_failures)
            writer.writerow(row)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        markdown(results, contexts, run_root, orca_sha),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    selected = set(args.datasets) if args.datasets else None
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else args.run_root.resolve() / "catflow_compatible"
    )
    markdown_path = (
        args.markdown.resolve()
        if args.markdown
        else output_dir / "catflow_metrics.md"
    )
    runs = scan_runs(args.run_root.resolve(), selected)
    if not runs:
        raise SystemExit(f"No permissive CatFlow graph artifacts found under {args.run_root}")
    orbit_cache = (
        args.orbit_cache.resolve()
        if args.orbit_cache
        else output_dir / "orbit_cache.sqlite3"
    )
    cache = OrbitCache(orbit_cache, args.orca, args.workers)
    try:
        results, contexts = recompute(runs, args.data_root.resolve(), cache)
        write_outputs(
            results,
            contexts,
            output_dir,
            markdown_path,
            args.run_root.resolve(),
            cache.orca_sha256,
        )
    finally:
        cache.close()
    print(f"Markdown: {markdown_path}")
    print(f"JSON/CSV: {output_dir}")


if __name__ == "__main__":
    main()
