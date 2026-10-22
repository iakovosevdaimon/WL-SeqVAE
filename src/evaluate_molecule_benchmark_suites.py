"""Evaluate generated molecules with MOSES- and GuacaMol-style metrics."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path


INVALID_SMILES = "INVALID"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate generated SMILES with optional MOSES and GuacaMol benchmark suites."
    )
    parser.add_argument("--runs-root", default="benchmark_runs/tuning_suggestions_seed0",
                        help="Root containing */samples/metrics.json outputs.")
    parser.add_argument("--run-dir", action="append", default=[],
                        help="Specific run directory to evaluate. Can be a run folder or its samples folder.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--suite", action="append", choices=["moses", "guacamol"],
                        help="Suite(s) to run. Defaults to both when importable.")
    parser.add_argument("--reference-mode", default="run", choices=["run", "official", "external"],
                        help="run: use each run's own train/reference SMILES; official: MOSES/GuacaMol defaults; external: provided files.")
    parser.add_argument("--train-smiles", default=None,
                        help="External train/reference SMILES for novelty and GuacaMol distribution learning.")
    parser.add_argument("--guacamol-train-smiles", default=None,
                        help="External GuacaMol/ChEMBL train SMILES. Overrides --train-smiles for GuacaMol metrics.")
    parser.add_argument("--test-smiles", default=None,
                        help="External test/reference SMILES for MOSES Test metrics.")
    parser.add_argument("--test-scaffolds-smiles", default=None,
                        help="External scaffold test SMILES for MOSES TestSF metrics.")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--include-invalid-placeholders", action=argparse.BooleanOptionalAction, default=True,
                        help="Append INVALID placeholders up to the recorded number of generation attempts.")
    parser.add_argument("--moses-backend", default="auto", choices=["auto", "official", "local"],
                        help="Use official moses.metrics when available, or a local RDKit-based fallback.")
    parser.add_argument("--local-moses-max-pairwise", type=int, default=1000,
                        help="Cap for pairwise local MOSES-style IntDiv/SNN computations.")
    parser.add_argument("--local-moses-max-reference", type=int, default=2000,
                        help="Cap for local MOSES-style reference fingerprints.")
    parser.add_argument("--guacamol-version", default="v1")
    return parser.parse_args()


def read_smiles(path: str | Path | None):
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing SMILES file: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.split()[0])
    return out


def write_smiles(path: Path, smiles):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(smiles) + ("\n" if smiles else ""))
    return path


def as_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def collect_run_dirs(args):
    dirs = []
    for item in args.run_dir:
        path = Path(item)
        samples = path if path.name == "samples" else path / "samples"
        if (samples / "metrics.json").exists():
            dirs.append(samples)
    if not dirs:
        root = Path(args.runs_root)
        dirs.extend(sorted(path.parent for path in root.glob("*/samples/metrics.json")))
    return dirs


def interleave_invalid_placeholders(valid_smiles, attempts: int):
    valid_smiles = list(valid_smiles)
    attempts = max(int(attempts), len(valid_smiles))
    invalid_count = max(0, attempts - len(valid_smiles))
    if invalid_count == 0:
        return valid_smiles[:attempts]

    out = []
    valid_idx = 0
    invalid_used = 0
    for pos in range(attempts):
        target_invalid = ((pos + 1) * invalid_count) // attempts
        if invalid_used < target_invalid:
            out.append(INVALID_SMILES)
            invalid_used += 1
        elif valid_idx < len(valid_smiles):
            out.append(valid_smiles[valid_idx])
            valid_idx += 1
        else:
            out.append(INVALID_SMILES)
            invalid_used += 1
    return out


def load_run(samples_dir: Path, *, include_invalid: bool):
    metrics_path = samples_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    dataset = str(metrics.get("config/dataset") or "")
    if metrics.get("config/mode") != "molecule" and dataset not in {"QM9", "QM9_H", "ZINC", "ZINC250K", "MOSES", "GUACAMOL"}:
        return None
    valid_path = samples_dir / "valid_smiles.txt"
    gen_path = valid_path if valid_path.exists() else samples_dir / "fcd_generated_smiles.txt"
    if not gen_path.exists():
        return None
    generated = read_smiles(gen_path) or []
    attempts = int(metrics.get("mol/fcd_attempts") or metrics.get("sampling/target_n_samples") or len(generated))
    if include_invalid:
        generated = interleave_invalid_placeholders(generated, attempts)
    variant = samples_dir.parent.name
    train_path = next(iter(sorted(samples_dir.glob("train_smiles_*_cache.json"))), None)
    return {
        "dataset": dataset,
        "variant": variant,
        "samples_dir": samples_dir,
        "metrics": metrics,
        "generated_smiles": generated,
        "valid_generated_smiles": read_smiles(gen_path) or [],
        "train_smiles": read_smiles(train_path) if train_path else None,
        "test_smiles": read_smiles(samples_dir / "reference_smiles.txt") if (samples_dir / "reference_smiles.txt").exists() else None,
    }


def moses_available():
    try:
        patch_pandas_for_legacy_moses()
        from moses.metrics import get_all_metrics  # noqa: F401

        return True
    except Exception:
        return False


def moses_suite_available(args):
    return args.moses_backend in {"auto", "local"} or moses_available()


def patch_scipy_for_legacy_guacamol():
    try:
        import numpy as np
        import scipy

        if not hasattr(scipy, "histogram"):
            scipy.histogram = np.histogram
    except Exception:
        pass


def patch_pandas_for_legacy_moses():
    try:
        import pandas as pd

        if not hasattr(pd.DataFrame, "append"):
            def _append(self, other, ignore_index=False, verify_integrity=False, sort=False):
                return pd.concat(
                    [self, other],
                    ignore_index=ignore_index,
                    verify_integrity=verify_integrity,
                    sort=sort,
                )

            pd.DataFrame.append = _append
    except Exception:
        pass


def guacamol_available():
    try:
        patch_scipy_for_legacy_guacamol()
        import guacamol  # noqa: F401

        return True
    except Exception:
        return False


def selected_suites(args):
    if args.suite:
        return list(dict.fromkeys(args.suite))
    suites = []
    if moses_suite_available(args):
        suites.append("moses")
    if guacamol_available():
        suites.append("guacamol")
    return suites


def resolve_references(run, args, out_dir: Path):
    if args.reference_mode == "official":
        return None, None, None, None
    if args.reference_mode == "external":
        train = read_smiles(args.train_smiles)
        test = read_smiles(args.test_smiles)
        test_scaffolds = read_smiles(args.test_scaffolds_smiles)
        train_file = Path(args.train_smiles) if args.train_smiles else None
        return train, test, test_scaffolds, train_file

    train = run.get("train_smiles")
    test = run.get("test_smiles")
    test_scaffolds = None
    train_file = None
    if train:
        train_file = write_smiles(out_dir / "references" / f"{run['variant']}_train_smiles.txt", train)
    return train, test, test_scaffolds, train_file


def run_moses(run, args, out_dir: Path):
    if args.moses_backend in {"auto", "official"}:
        try:
            patch_pandas_for_legacy_moses()
            from moses.metrics import get_all_metrics
        except Exception:
            if args.moses_backend == "official":
                raise
        else:
            train, test, test_scaffolds, _train_file = resolve_references(run, args, out_dir)
            metrics = get_all_metrics(
                gen=run["generated_smiles"][: int(args.n_samples)],
                k=[1000, int(args.n_samples)],
                n_jobs=int(args.n_jobs),
                device=args.device,
                batch_size=int(args.batch_size),
                test=test,
                test_scaffolds=test_scaffolds,
                train=train,
            )
            rows = []
            for name, value in sorted(metrics.items()):
                rows.append({
                    "suite": "moses",
                    "dataset": run["dataset"],
                    "variant": run["variant"],
                    "reference_mode": args.reference_mode,
                    "metric": name,
                    "value": as_float(value),
                    "raw_value": jsonable(value),
                    "metadata": {"backend": "official"},
                })
            return rows

    return run_moses_local(run, args, out_dir)


def canonicalize_smiles(smiles):
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    out = []
    mols = []
    for smi in smiles:
        if not smi or str(smi).upper() == INVALID_SMILES:
            out.append(None)
            mols.append(None)
            continue
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            out.append(None)
            mols.append(None)
            continue
        out.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
        mols.append(mol)
    return out, mols


def make_fingerprints(mols):
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return [generator.GetFingerprint(mol) for mol in mols if mol is not None]


def pairwise_internal_diversity(fps, *, max_items):
    from rdkit import DataStructs

    fps = list(fps[: max(0, int(max_items))])
    if len(fps) < 2:
        return None, None
    total = 0.0
    total_sq = 0.0
    count = 0
    for idx in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[idx], fps[:idx])
        total += sum(sims)
        total_sq += sum(sim * sim for sim in sims)
        count += len(sims)
    if count <= 0:
        return None, None
    mean_sim = total / count
    mean_sim_sq = total_sq / count
    return 1.0 - mean_sim, 1.0 - math.sqrt(max(mean_sim_sq, 0.0))


def nearest_neighbor_similarity(gen_fps, ref_fps, *, max_items, max_reference):
    from rdkit import DataStructs

    gen_fps = list(gen_fps[: max(0, int(max_items))])
    ref_fps = list(ref_fps[: max(0, int(max_reference))])
    if not gen_fps or not ref_fps:
        return None
    best = []
    for fp in gen_fps:
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        best.append(max(sims) if sims else 0.0)
    return sum(best) / len(best) if best else None


def cosine_counter_similarity(left: Counter, right: Counter):
    if not left or not right:
        return None
    dot = sum(float(left[key]) * float(right.get(key, 0)) for key in left)
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left.values()))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right.values()))
    denom = left_norm * right_norm
    return dot / denom if denom > 0.0 else None


def fragment_counter(mols):
    from rdkit.Chem import BRICS

    counts = Counter()
    for mol in mols:
        if mol is None:
            continue
        try:
            fragments = BRICS.BRICSDecompose(mol)
        except Exception:
            fragments = set()
        for fragment in fragments:
            counts[str(fragment)] += 1
    return counts


def scaffold_counter(mols):
    from rdkit.Chem.Scaffolds import MurckoScaffold

    counts = Counter()
    for mol in mols:
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
        except Exception:
            scaffold = ""
        if scaffold:
            counts[str(scaffold)] += 1
    return counts


def local_moses_metrics(run, args, out_dir: Path):
    train, test, test_scaffolds, _train_file = resolve_references(run, args, out_dir)
    generated = list(run["generated_smiles"][: int(args.n_samples)])
    canonical, mols = canonicalize_smiles(generated)
    valid = [smi for smi in canonical if smi is not None]
    valid_mols = [mol for mol in mols if mol is not None]
    unique_valid = sorted(set(valid))
    unique_fraction = len(unique_valid) / len(valid) if valid else 0.0
    train_set = set(canonicalize_smiles(train or [])[0]) - {None}
    novelty = (sum(1 for smi in unique_valid if smi not in train_set) / len(unique_valid)) if unique_valid and train_set else None

    metrics = {
        "valid": len(valid) / len(generated) if generated else 0.0,
        "unique": unique_fraction,
        f"unique@{min(1000, len(generated))}": None,
        f"unique@{int(args.n_samples)}": unique_fraction,
        "Novelty": novelty,
        "Filters": None,
    }
    for k in [1000, int(args.n_samples)]:
        subset = generated[: min(k, len(generated))]
        subset_canonical, _subset_mols = canonicalize_smiles(subset)
        subset_valid = [smi for smi in subset_canonical if smi is not None]
        metrics[f"unique@{k}"] = len(set(subset_valid)) / len(subset_valid) if subset_valid else 0.0

    gen_fps = make_fingerprints(valid_mols)
    intdiv, intdiv2 = pairwise_internal_diversity(gen_fps, max_items=args.local_moses_max_pairwise)
    metrics["IntDiv"] = intdiv
    metrics["IntDiv2"] = intdiv2

    reference = test or test_scaffolds
    if reference:
        ref_canonical, ref_mols = canonicalize_smiles(reference)
        ref_mols = [mol for mol in ref_mols if mol is not None]
        ref_fps = make_fingerprints(ref_mols)
        metrics["SNN/Test"] = nearest_neighbor_similarity(
            gen_fps,
            ref_fps,
            max_items=args.local_moses_max_pairwise,
            max_reference=args.local_moses_max_reference,
        )
        metrics["Frag/Test"] = cosine_counter_similarity(fragment_counter(valid_mols), fragment_counter(ref_mols))
        metrics["Scaf/Test"] = cosine_counter_similarity(scaffold_counter(valid_mols), scaffold_counter(ref_mols))
    else:
        metrics["SNN/Test"] = None
        metrics["Frag/Test"] = None
        metrics["Scaf/Test"] = None

    if run["metrics"].get("mol/fcd") is not None:
        fcd_split = str(run["metrics"].get("mol/fcd_reference_split") or run["metrics"].get("config/fcd_reference_split") or "")
        metrics["FCD/Test" if fcd_split == "test" else "FCD/RunRef"] = run["metrics"].get("mol/fcd")

    return metrics


def run_moses_local(run, args, out_dir: Path):
    metrics = local_moses_metrics(run, args, out_dir)
    rows = []
    for name, value in sorted(metrics.items()):
        rows.append({
            "suite": "moses_local",
            "dataset": run["dataset"],
            "variant": run["variant"],
            "reference_mode": args.reference_mode,
            "metric": name,
            "value": as_float(value),
            "raw_value": jsonable(value),
            "metadata": {
                "backend": "local_rdkit",
                "note": "MOSES-style fallback; not an exact official MOSES implementation.",
            },
        })
    return rows


class FixedSmilesGenerator:
    def __init__(self, smiles):
        patch_scipy_for_legacy_guacamol()
        from guacamol.distribution_matching_generator import DistributionMatchingGenerator

        class _Generator(DistributionMatchingGenerator):
            def __init__(self, values):
                self.values = list(values)

            def generate(self, number_samples: int):
                values = list(self.values[: int(number_samples)])
                if len(values) < int(number_samples):
                    values.extend([INVALID_SMILES] * (int(number_samples) - len(values)))
                return values

        self.impl = _Generator(smiles)

    def generate(self, number_samples: int):
        return self.impl.generate(number_samples)


def run_guacamol(run, args, out_dir: Path):
    patch_scipy_for_legacy_guacamol()
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    from guacamol.assess_distribution_learning import (
        _assess_distribution_learning,
        _evaluate_distribution_learning_benchmarks,
    )
    from guacamol.benchmark_suites import distribution_learning_benchmark_suite

    train, _test, _test_scaffolds, train_file = resolve_references(run, args, out_dir)
    guacamol_train_smiles = args.guacamol_train_smiles or args.train_smiles
    if args.reference_mode == "official":
        if not guacamol_train_smiles:
            raise RuntimeError("GuacaMol official/external evaluation needs --train-smiles with the ChEMBL training file.")
        train_file = Path(guacamol_train_smiles)
    elif args.reference_mode == "external":
        train_file = Path(guacamol_train_smiles) if guacamol_train_smiles else None

    if train_file is None or not train_file.exists():
        raise RuntimeError(f"No training SMILES file available for GuacaMol variant {run['variant']}.")

    generator = FixedSmilesGenerator(run["generated_smiles"])
    result_path = out_dir / "guacamol_raw" / f"{run['variant']}_{args.reference_mode}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_fcd = False
    try:
        _assess_distribution_learning(
            model=generator.impl,
            chembl_training_file=str(train_file),
            json_output_file=str(result_path),
            benchmark_version=args.guacamol_version,
            number_samples=int(args.n_samples),
        )
        payload = json.loads(result_path.read_text())
        raw_results = payload.get("results", [])
    except FileNotFoundError as exc:
        if "ChemNet" not in str(exc) and "fcd" not in str(exc).lower():
            raise
        skipped_fcd = True
        benchmarks = distribution_learning_benchmark_suite(
            chembl_file_path=str(train_file),
            version_name=args.guacamol_version,
            number_samples=int(args.n_samples),
        )
        benchmarks = [
            benchmark for benchmark in benchmarks
            if getattr(benchmark, "name", "") != "Frechet ChemNet Distance"
        ]
        raw_results = [
            vars(result)
            for result in _evaluate_distribution_learning_benchmarks(
                model=generator.impl,
                benchmarks=benchmarks,
            )
        ]
        result_path.write_text(json.dumps({
            "results": raw_results,
            "skipped": ["Frechet ChemNet Distance: missing FCD ChemNet weights"],
        }, indent=2, sort_keys=True) + "\n")

    rows = []
    for result in raw_results:
        rows.append({
            "suite": "guacamol",
            "dataset": run["dataset"],
            "variant": run["variant"],
            "reference_mode": args.reference_mode,
            "metric": result.get("benchmark_name"),
            "value": as_float(result.get("score")),
            "raw_value": as_float(result.get("score")),
            "metadata": jsonable(result.get("metadata", {})),
        })
    if skipped_fcd:
        rows.append({
            "suite": "guacamol",
            "dataset": run["dataset"],
            "variant": run["variant"],
            "reference_mode": args.reference_mode,
            "metric": "Frechet ChemNet Distance",
            "value": None,
            "raw_value": None,
            "metadata": {"status": "missing_fcd_chemnet_weights"},
        })
    return rows


def write_csv(path: Path, rows):
    fieldnames = ["suite", "dataset", "variant", "reference_mode", "metric", "value", "raw_value", "metadata"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["raw_value"] = json.dumps(jsonable(out["raw_value"]), sort_keys=True)
            out["metadata"] = json.dumps(jsonable(out["metadata"]), sort_keys=True)
            writer.writerow(out)


def write_markdown(path: Path, rows, skipped):
    key_metrics = {
        "valid",
        "unique@1000",
        "unique@10000",
        "FCD/Test",
        "FCD/TestSF",
        "SNN/Test",
        "Frag/Test",
        "Scaf/Test",
        "IntDiv",
        "IntDiv2",
        "Filters",
        "Novelty",
        "Validity",
        "Uniqueness",
        "Novelty",
        "KL divergence",
        "Frechet ChemNet Distance",
    }
    lines = [
        "# MOSES / GuacaMol Metrics",
        "",
        "Generated SMILES are taken from existing run folders. When enabled, invalid placeholders are appended up to the recorded number of generation attempts, so validity is not inflated by valid-only files.",
        "",
        "| Suite | Dataset | Variant | Metric | Value |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        if row["metric"] not in key_metrics and not row["metric"].lower().startswith(("valid", "unique", "novel", "kl", "frechet")):
            continue
        value = row["value"]
        value_text = "--" if value is None else f"{float(value):.6f}"
        lines.append(f"| {row['suite']} | {row['dataset']} | {row['variant']} | {row['metric']} | {value_text} |")
    if skipped:
        lines.extend(["", "## Skipped", ""])
        for item in skipped:
            lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.runs_root) / "summary" / "molecule_benchmark_suites"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for samples_dir in collect_run_dirs(args):
        run = load_run(samples_dir, include_invalid=bool(args.include_invalid_placeholders))
        if run is not None:
            runs.append(run)
    if not runs:
        raise SystemExit("No molecule runs with generated SMILES were found.")

    suites = selected_suites(args)
    skipped = []
    if not suites:
        skipped.append("No benchmark suites are importable. Install moses and/or guacamol.")

    rows = []
    for run in runs:
        for suite in suites:
            try:
                if suite == "moses":
                    rows.extend(run_moses(run, args, out_dir))
                elif suite == "guacamol":
                    rows.extend(run_guacamol(run, args, out_dir))
            except Exception as exc:
                skipped.append(f"{suite} {run['variant']}: {type(exc).__name__}: {exc}")

    csv_path = out_dir / "molecule_benchmark_metrics.csv"
    json_path = out_dir / "molecule_benchmark_metrics.json"
    md_path = out_dir / "molecule_benchmark_metrics.md"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps({"rows": jsonable(rows), "skipped": skipped}, indent=2, sort_keys=True) + "\n")
    write_markdown(md_path, rows, skipped)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if skipped:
        print("Skipped:")
        for item in skipped:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
