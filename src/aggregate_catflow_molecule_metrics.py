#!/usr/bin/env python3
"""Aggregate CatFlow molecule metrics across declared seeds without test cherry-picking."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


CHECKPOINT_ROLES = (
    "structural_validation_best",
    "legacy_warmup_weighted_val_loss_best",
    "last_epoch",
)
CHECKPOINT_ROLE_ALIASES = {
    "structural_validation_best": "structural_validation_best",
    "legacy_val_loss_best": "legacy_warmup_weighted_val_loss_best",
    "legacy_warmup_weighted_val_loss_best": "legacy_warmup_weighted_val_loss_best",
    "last_epoch": "last_epoch",
}
METRICS = ("mol/validity", "mol/uniqueness", "mol/fcd")


@dataclass
class RunMetric:
    dataset: str
    wl: int
    seed: int
    checkpoint_role: str
    checkpoint_epoch: int | None
    validity: float | None
    uniqueness: float | None
    fcd: float | None
    directly_catflow_comparable: bool
    comparability_failures: list[str]
    metrics_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", required=True, choices=("QM9", "ZINC250K"))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--wl-iters", type=int, nargs="+", required=True)
    parser.add_argument(
        "--primary-checkpoint-role",
        choices=CHECKPOINT_ROLES,
        default="structural_validation_best",
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def path_index(parts: tuple[str, ...], prefix: str) -> int | None:
    pattern = re.compile(rf"{re.escape(prefix)}([0-9]+)")
    for part in parts:
        match = pattern.fullmatch(part)
        if match:
            return int(match.group(1))
    return None


def load_run(path: Path, root: Path, expected_dataset: str) -> RunMetric | None:
    if path.parent.name != "samples":
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if "sampling/generated_graphs" not in payload:
        return None

    relative_parts = path.relative_to(root).parts
    wl = path_index(relative_parts, "wl")
    seed = path_index(relative_parts, "seed")
    if wl is None or seed is None:
        return None
    dataset = str(payload.get("config/dataset") or "").upper()
    if dataset != expected_dataset:
        return None
    raw_role = str(payload.get("config/checkpoint_role") or "")
    role = CHECKPOINT_ROLE_ALIASES.get(raw_role)
    if role is None:
        return None

    failures = list(payload.get("mol/catflow_comparability_failures") or [])
    requirements = {
        "catflow_gate": payload.get("mol/catflow_table_comparable") is True,
        "no_checkpoint_mismatch_warnings": not payload.get(
            "config/checkpoint_mismatch_warnings"
        ),
        "checkpoint_data_provenance_verified": payload.get(
            "config/verify_data_provenance"
        )
        is True,
        "requested_wl_matches_directory": int(
            payload.get("config/requested_n_wl_iters") or -1
        )
        == wl,
        "ten_thousand_attempts": int(payload.get("mol/fcd_attempts") or -1)
        == 10000,
        "test_reference": payload.get("config/reference_split") == "test",
        "test_fcd_reference": payload.get("mol/fcd_reference_split") == "test",
        "permissive_paper_collection": payload.get("config/require_complete")
        is False,
        "validity_available": finite_float(payload.get("mol/validity"))
        is not None,
        "uniqueness_available": finite_float(payload.get("mol/uniqueness"))
        is not None,
        "fcd_available": finite_float(payload.get("mol/fcd")) is not None,
    }
    for name, satisfied in requirements.items():
        if not satisfied and name not in failures:
            failures.append(name)

    return RunMetric(
        dataset=dataset,
        wl=wl,
        seed=seed,
        checkpoint_role=role,
        checkpoint_epoch=payload.get("config/checkpoint_epoch"),
        validity=finite_float(payload.get("mol/validity")),
        uniqueness=finite_float(payload.get("mol/uniqueness")),
        fcd=finite_float(payload.get("mol/fcd")),
        directly_catflow_comparable=not failures,
        comparability_failures=failures,
        metrics_path=str(path),
    )


def aggregate(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "population_std": statistics.pstdev(values),
    }


def fmt_metric(summary: dict, *, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    digits = 2 if percent else 4
    return (
        f"{scale * summary['mean']:.{digits}f} ± "
        f"{scale * summary['population_std']:.{digits}f}"
    )


def write_outputs(
    *,
    dataset: str,
    root: Path,
    output_dir: Path,
    declared_seeds: list[int],
    declared_wls: list[int],
    primary_role: str,
    runs: list[RunMetric],
    failures: list[str],
) -> None:
    grouped = {}
    for run in runs:
        key = (run.wl, run.checkpoint_role)
        grouped.setdefault(key, []).append(run)

    aggregates = []
    for wl in declared_wls:
        for role in CHECKPOINT_ROLES:
            group = sorted(grouped.get((wl, role), []), key=lambda run: run.seed)
            if len(group) != len(declared_seeds):
                continue
            if not all(run.directly_catflow_comparable for run in group):
                continue
            aggregates.append(
                {
                    "dataset": dataset,
                    "wl": wl,
                    "checkpoint_role": role,
                    "seeds": declared_seeds,
                    "validity": aggregate([run.validity for run in group]),
                    "uniqueness": aggregate([run.uniqueness for run in group]),
                    "fcd": aggregate([run.fcd for run in group]),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "dataset": dataset,
            "root": str(root),
            "declared_seeds": declared_seeds,
            "declared_wl_iters": declared_wls,
            "aggregation": "arithmetic_mean_and_population_standard_deviation",
            "primary_checkpoint_role": primary_role,
            "selection_policy": (
                "Report every declared WL at the fixed primary checkpoint role; "
                "do not select WL or checkpoint using test FCD."
            ),
        },
        "directly_catflow_comparable": not failures,
        "failures": failures,
        "aggregates": aggregates,
        "runs": [asdict(run) for run in runs],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "dataset",
                "wl",
                "checkpoint_role",
                "seeds",
                "n",
                "validity_mean",
                "validity_population_std",
                "uniqueness_mean",
                "uniqueness_population_std",
                "fcd_mean",
                "fcd_population_std",
            ]
        )
        for row in aggregates:
            writer.writerow(
                [
                    dataset,
                    row["wl"],
                    row["checkpoint_role"],
                    " ".join(map(str, declared_seeds)),
                    row["validity"]["n"],
                    row["validity"]["mean"],
                    row["validity"]["population_std"],
                    row["uniqueness"]["mean"],
                    row["uniqueness"]["population_std"],
                    row["fcd"]["mean"],
                    row["fcd"]["population_std"],
                ]
            )

    aggregate_lookup = {
        (row["wl"], row["checkpoint_role"]): row for row in aggregates
    }
    lines = [
        f"# CatFlow {dataset} multi-seed results",
        "",
        f"Declared seeds: **{', '.join(map(str, declared_seeds))}**.",
        "",
        f"Declared WL iterations: **{', '.join(map(str, declared_wls))}**.",
        "",
        "Aggregation: arithmetic mean ± population standard deviation.",
        "",
        f"Directly CatFlow-comparable collection: **{'yes' if not failures else 'no'}**.",
        "",
        "## Primary publication rows",
        "",
        (
            f"The checkpoint role is fixed to `{primary_role}` for every seed. "
            "No checkpoint or WL is selected using test FCD."
        ),
        "",
        "| WL | Seeds (n) | Validity % ↑ | Uniqueness % ↑ | FCD ↓ |",
        "|---:|---|---:|---:|---:|",
    ]
    for wl in declared_wls:
        row = aggregate_lookup.get((wl, primary_role))
        if row is None:
            lines.append(f"| {wl} | -- | -- | -- | -- |")
        else:
            lines.append(
                f"| {wl} | {', '.join(map(str, declared_seeds))} "
                f"({row['validity']['n']}) | "
                f"{fmt_metric(row['validity'], percent=True)} | "
                f"{fmt_metric(row['uniqueness'], percent=True)} | "
                f"{fmt_metric(row['fcd'])} |"
            )

    lines.extend(
        [
            "",
            "## All checkpoint roles",
            "",
            "| WL | Checkpoint role | n | Validity % ↑ | Uniqueness % ↑ | FCD ↓ |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in aggregates:
        lines.append(
            f"| {row['wl']} | {row['checkpoint_role']} | "
            f"{row['validity']['n']} | "
            f"{fmt_metric(row['validity'], percent=True)} | "
            f"{fmt_metric(row['uniqueness'], percent=True)} | "
            f"{fmt_metric(row['fcd'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-run comparability audit",
            "",
            "| WL | Seed | Checkpoint role | Direct? | Failure(s) | Metrics |",
            "|---:|---:|---|---|---|---|",
        ]
    )
    for run in sorted(
        runs, key=lambda item: (item.wl, item.seed, CHECKPOINT_ROLES.index(item.checkpoint_role))
    ):
        lines.append(
            f"| {run.wl} | {run.seed} | {run.checkpoint_role} | "
            f"{'yes' if run.directly_catflow_comparable else 'no'} | "
            f"{', '.join(run.comparability_failures) or '--'} | "
            f"`{run.metrics_path}` |"
        )
    if failures:
        lines.extend(["", "## Blocking failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    lines.append("")
    (output_dir / "catflow_molecule_results.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else root / "catflow_aggregate"
    )
    declared_seeds = list(dict.fromkeys(args.seeds))
    declared_wls = list(dict.fromkeys(args.wl_iters))
    failures = []
    if len(declared_seeds) != len(args.seeds):
        failures.append("declared_seed_list_contains_duplicates")
    if len(declared_wls) != len(args.wl_iters):
        failures.append("declared_wl_list_contains_duplicates")
    if any(seed < 0 for seed in declared_seeds):
        failures.append("declared_seed_must_be_non_negative")
    if any(wl < 1 for wl in declared_wls):
        failures.append("declared_wl_must_be_positive")

    runs = []
    for path in sorted(root.rglob("metrics.json")):
        run = load_run(path, root, args.dataset)
        if run is not None:
            runs.append(run)

    by_key = {}
    for run in runs:
        by_key.setdefault((run.wl, run.seed, run.checkpoint_role), []).append(run)
    expected_keys = {
        (wl, seed, role)
        for wl in declared_wls
        for seed in declared_seeds
        for role in CHECKPOINT_ROLES
    }
    for key in sorted(expected_keys):
        matches = by_key.get(key, [])
        if not matches:
            failures.append(
                f"missing_WL{key[0]}_seed{key[1]}_{key[2]}"
            )
        elif len(matches) > 1:
            failures.append(
                f"duplicate_WL{key[0]}_seed{key[1]}_{key[2]}_{len(matches)}rows"
            )
        elif not matches[0].directly_catflow_comparable:
            failures.append(
                f"noncomparable_WL{key[0]}_seed{key[1]}_{key[2]}:"
                + ",".join(matches[0].comparability_failures)
            )
    unexpected_keys = sorted(set(by_key) - expected_keys)
    for wl, seed, role in unexpected_keys:
        failures.append(f"unexpected_WL{wl}_seed{seed}_{role}")

    write_outputs(
        dataset=args.dataset,
        root=root,
        output_dir=output_dir,
        declared_seeds=declared_seeds,
        declared_wls=declared_wls,
        primary_role=args.primary_checkpoint_role,
        runs=runs,
        failures=failures,
    )
    print(f"Markdown: {output_dir / 'catflow_molecule_results.md'}")
    print(f"JSON/CSV: {output_dir}")
    if failures:
        raise SystemExit(
            f"CatFlow aggregation failed closed with {len(failures)} issue(s); "
            f"see {output_dir / 'catflow_molecule_results.md'}"
        )


if __name__ == "__main__":
    main()
