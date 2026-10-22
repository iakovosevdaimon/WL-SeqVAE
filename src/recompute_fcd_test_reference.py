"""Recompute FCD against checksum-pinned official molecule test references."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from compute_fcd import read_smiles
from gpt_datasets import (
    QM9_PROTOCOLS,
    ZINC250K_PROTOCOLS,
    load_graph_dataset_bundle,
)
from zinc_utils import build_smiles


MOLECULE_DATASETS = ("QM9", "QM9_H", "ZINC250K")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute FCD for existing molecule runs using only the dataset test split as reference."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--preprocess", default="catflow", choices=["repo", "catflow"])
    parser.add_argument("--qm9-protocol", default="gdss_catflow", choices=QM9_PROTOCOLS)
    parser.add_argument(
        "--zinc250k-protocol",
        default="gdss_catflow",
        choices=ZINC250K_PROTOCOLS,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pct(value):
    value = as_float(value)
    return "--" if value is None else f"{100.0 * value:.2f}"


def num(value, digits=6):
    value = as_float(value)
    return "--" if value is None else f"{value:.{digits}f}"


def text(value):
    return "--" if value is None else str(value)


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def collect_molecule_runs(root: Path):
    rows = []
    for metrics_path in sorted(root.glob("*/samples/metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            continue
        dataset = metrics.get("config/dataset")
        if dataset not in MOLECULE_DATASETS:
            continue
        generated_path = metrics_path.parent / "fcd_generated_smiles.txt"
        if not generated_path.exists():
            continue
        rows.append(
            {
                "variant": metrics_path.parent.parent.name,
                "dataset": dataset,
                "metrics_path": str(metrics_path),
                "generated_path": generated_path,
                "metrics": metrics,
            }
        )
    return rows


def build_test_references(
    root: Path,
    out_dir: Path,
    datasets,
    *,
    seed: int,
    preprocess: str,
    qm9_protocol: str,
    zinc250k_protocol: str,
):
    refs = {}
    for dataset in datasets:
        print(f"Building {dataset} test-reference SMILES ({preprocess})...", flush=True)
        bundle = load_graph_dataset_bundle(
            dataset,
            zinc_subset=("large" if dataset == "ZINC250K" else None),
            seed=seed,
            qm9_protocol=(qm9_protocol if dataset == "QM9" else "legacy_local"),
            zinc250k_protocol=(
                zinc250k_protocol if dataset == "ZINC250K" else "legacy_pyg"
            ),
        )
        cache_path = out_dir / f"{dataset.lower()}_test_{preprocess}_retained_smiles_cache.json"
        smiles = build_smiles(
            bundle.test_graphs,
            bundle.atom_decoder,
            cache_path=cache_path,
            preprocess=preprocess,
            deduplicate=False,
        )
        txt_path = out_dir / f"{dataset.lower()}_test_reference_smiles.txt"
        txt_path.write_text("\n".join(smiles) + ("\n" if smiles else ""))
        refs[dataset] = {"smiles": smiles, "path": txt_path, "n_graphs": len(bundle.test_graphs)}
        print(f"  {dataset}: graphs={len(bundle.test_graphs)} smiles={len(smiles)}", flush=True)
    return refs


def make_fcd_backend(device: str, n_jobs: int, batch_size: int):
    try:
        from fcd_torch import FCD

        fcd = FCD(device=device, n_jobs=n_jobs, batch_size=batch_size)

        def compute(ref_smiles, gen_smiles):
            return float(fcd(ref=ref_smiles, gen=gen_smiles)), "fcd_torch"

        return compute
    except ImportError:
        from fcd import get_fcd

        def compute(ref_smiles, gen_smiles):
            return float(get_fcd(ref_smiles, gen_smiles)), "fcd"

        return compute


def write_csv(path: Path, rows):
    fieldnames = [
        "dataset",
        "variant",
        "fcd_test_reference",
        "fcd_all_reference",
        "delta_test_minus_all",
        "test_reference_graphs",
        "test_reference_smiles",
        "generated_smiles",
        "validity_percent",
        "uniqueness_percent",
        "novelty_percent",
        "complete_percent",
        "node_tv",
        "edge_tv",
        "degree_tv",
        "wl_tv",
        "backend",
        "seconds",
        "reference_path",
        "generated_path",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown(path: Path, rows):
    ranked = sorted(rows, key=lambda row: (row["dataset"], as_float(row["fcd_test_reference"]) or float("inf")))
    table = markdown_table(
        [
            "Dataset",
            "Variant",
            "FCD test ref",
            "FCD all ref",
            "Delta",
            "Test ref smiles",
            "Gen smiles",
            "Valid %",
            "Unique %",
            "Novel %",
            "Complete %",
            "WL TV",
        ],
        [
            [
                row["dataset"],
                row["variant"],
                num(row["fcd_test_reference"]),
                num(row["fcd_all_reference"]),
                num(row["delta_test_minus_all"]),
                text(row["test_reference_smiles"]),
                text(row["generated_smiles"]),
                num(row["validity_percent"], 2),
                num(row["uniqueness_percent"], 2),
                num(row["novelty_percent"], 2),
                num(row["complete_percent"], 2),
                num(row["wl_tv"], 4),
            ]
            for row in ranked
        ],
    )

    best_rows = []
    for dataset in sorted({row["dataset"] for row in rows}):
        group = [row for row in rows if row["dataset"] == dataset]
        best = min(group, key=lambda row: as_float(row["fcd_test_reference"]) or float("inf"))
        best_rows.append(
            [
                dataset,
                best["variant"],
                num(best["fcd_test_reference"]),
                num(best["fcd_all_reference"]),
                (
                    "Original QM9 with explicit H atoms"
                    if dataset == "QM9_H"
                    else "QM9 no-H/heavy-atom setting"
                    if dataset == "QM9"
                    else "ZINC250K"
                ),
            ]
        )

    best_table = markdown_table(
        ["Dataset", "Best variant", "FCD test ref", "FCD all ref", "Description"],
        best_rows,
    )

    path.write_text(
        "\n".join(
            [
                "# Test-Reference FCD Recompute",
                "",
                "FCD here uses only the dataset test split as reference. Generated SMILES are reused from the existing tuning run, with generated duplicates retained.",
                "",
                table,
                "",
                "## Best By Test-Reference FCD",
                "",
                best_table,
                "",
            ]
        )
    )


def main():
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "summary" / "fcd_test_reference"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = collect_molecule_runs(root)
    if not runs:
        raise SystemExit(f"No molecule runs with fcd_generated_smiles.txt found under {root}")
    if args.preprocess == "catflow":
        skipped_qm9_h = sum(run["dataset"] == "QM9_H" for run in runs)
        if skipped_qm9_h:
            print(
                f"Skipping {skipped_qm9_h} QM9_H run(s): CatFlow H-stripped FCD is not a "
                "published-baseline comparison for the explicit-H benchmark.",
                flush=True,
            )
            runs = [run for run in runs if run["dataset"] != "QM9_H"]
    runs = [
        run
        for run in runs
        if (run["metrics"].get("mol/fcd_preprocess") or run["metrics"].get("config/mol_preprocess"))
        in {None, args.preprocess}
    ]
    if not runs:
        raise SystemExit("No molecule runs match the requested representation/preprocessing policy")
    datasets = sorted({run["dataset"] for run in runs})
    refs = build_test_references(
        root,
        out_dir,
        datasets,
        seed=args.seed,
        preprocess=args.preprocess,
        qm9_protocol=args.qm9_protocol,
        zinc250k_protocol=args.zinc250k_protocol,
    )
    compute_fcd = make_fcd_backend(args.device, args.n_jobs, args.batch_size)

    summary_rows = []
    for idx, run in enumerate(runs, start=1):
        dataset = run["dataset"]
        variant = run["variant"]
        metrics = run["metrics"]
        result_path = out_dir / f"{variant}_fcd_test_reference.json"
        if result_path.exists() and not args.force:
            result = json.loads(result_path.read_text())
            print(f"[{idx}/{len(runs)}] Reusing {variant}: {result['fcd']:.6f}", flush=True)
        else:
            gen_smiles = read_smiles(run["generated_path"])
            start = time.perf_counter()
            value, backend = compute_fcd(refs[dataset]["smiles"], gen_smiles)
            seconds = time.perf_counter() - start
            result = {
                "dataset": dataset,
                "variant": variant,
                "fcd": value,
                "backend": backend,
                "seconds": seconds,
                "reference_split": "test",
                "reference_preprocess": args.preprocess,
                "test_reference_graphs": refs[dataset]["n_graphs"],
                "test_reference_smiles": len(refs[dataset]["smiles"]),
                "generated_smiles": len(gen_smiles),
                "reference_path": str(refs[dataset]["path"]),
                "generated_path": str(run["generated_path"]),
            }
            result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(f"[{idx}/{len(runs)}] {variant}: {value:.6f} ({seconds:.1f}s)", flush=True)

        all_fcd = as_float(metrics.get("mol/fcd"))
        test_fcd = as_float(result["fcd"])
        summary_rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "fcd_test_reference": test_fcd,
                "fcd_all_reference": all_fcd,
                "delta_test_minus_all": None if test_fcd is None or all_fcd is None else test_fcd - all_fcd,
                "test_reference_graphs": result["test_reference_graphs"],
                "test_reference_smiles": result["test_reference_smiles"],
                "generated_smiles": result["generated_smiles"],
                "validity_percent": None if metrics.get("mol/validity") is None else 100.0 * float(metrics["mol/validity"]),
                "uniqueness_percent": None if metrics.get("mol/uniqueness") is None else 100.0 * float(metrics["mol/uniqueness"]),
                "novelty_percent": None if metrics.get("mol/novelty") is None else 100.0 * float(metrics["mol/novelty"]),
                "complete_percent": None if metrics.get("sampling/complete_fraction") is None else 100.0 * float(metrics["sampling/complete_fraction"]),
                "node_tv": metrics.get("dist/node_count/tv"),
                "edge_tv": metrics.get("dist/edge_count/tv"),
                "degree_tv": metrics.get("dist/degree/tv"),
                "wl_tv": metrics.get("dist/wl_color/tv"),
                "backend": result.get("backend"),
                "seconds": result.get("seconds"),
                "reference_path": result.get("reference_path"),
                "generated_path": result.get("generated_path"),
            }
        )

    summary_rows.sort(key=lambda row: (row["dataset"], as_float(row["fcd_test_reference"]) or float("inf")))
    csv_path = out_dir / "fcd_test_reference_summary.csv"
    md_path = out_dir / "fcd_test_reference_summary.md"
    json_path = out_dir / "fcd_test_reference_summary.json"
    write_csv(csv_path, summary_rows)
    write_markdown(md_path, summary_rows)
    json_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
