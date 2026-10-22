#!/usr/bin/env python3
"""Generate the LaTeX summary for corrected DiGress/SPECTRE-compatible metrics."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from project_paths import PROJECT_ROOT, portable_path

RUN_ROOT = PROJECT_ROOT / "benchmark_runs" / "corrected_graphs_wl1_wl2_wl3_seed0"
METRICS = RUN_ROOT / "digress_spectre_compatible" / "metrics.json"
OUTPUT = PROJECT_ROOT / "current_best_benchmark_results.tex"

DATASET_ORDER = (
    "COMMUNITY_SMALL",
    "EGO_SMALL",
    "ENZYMES",
    "GRID",
    "MUTAG",
    "NCI1",
    "PLANAR",
    "PROTEINS",
    "SBM",
)

ROLE_ORDER = {
    "structural_validation_best": 0,
    "fixed_beta_best": 1,
    "legacy_val_loss_best": 2,
    "last_epoch": 3,
}
ROLE_SHORT = {
    "structural_validation_best": "SV",
    "fixed_beta_best": "FB",
    "legacy_val_loss_best": "LV",
    "last_epoch": "Last",
}

OTHER_BENCHMARKS = (
    ("GRID", "Graph", "No finished compatible result in this sweep"),
    ("QM9", "Molecule", "No finished corrected result in this sweep"),
    ("QM9_H", "Molecule", "No finished corrected result in this sweep"),
    ("ZINC250K", "Molecule", "No finished corrected result in this sweep"),
    ("MOSES", "Molecule", "No finished corrected result in this sweep"),
    ("GuacaMol", "Molecule", "No finished corrected result in this sweep"),
)


@dataclass(frozen=True)
class Result:
    dataset: str
    wl: int
    checkpoint_role: str
    epoch: int | None
    samples: int
    target_samples: int
    strict_generated: int
    complete_pct: float
    vun_pct: float
    validity_metric: str | None
    sbm_accuracy_refinement_steps: int | None
    sbm_vun_refinement_steps: int | None
    directly_paper_comparable: bool
    comparison_protocol: str | None
    comparability_failures: list[str]
    degree_mmd2: float
    clustering_mmd2: float
    orbit_mmd2: float
    degree_ratio: float | None
    clustering_ratio: float | None
    orbit_ratio: float | None
    mean_ratio: float | None
    ratio_metrics_used: int
    denominator_source: str


def tex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    return "".join(replacements.get(char, char) for char in text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=PROJECT_ROOT / "current_corrected_digress_spectre_metrics.md",
    )
    return parser.parse_args()


def load_results(metrics: Path) -> tuple[dict, list[Result]]:
    aggregate = json.loads(metrics.read_text(encoding="utf-8"))
    rows = [
        Result(
            dataset=row["dataset"],
            wl=int(row["wl"]),
            checkpoint_role=row["checkpoint_role"],
            epoch=row["checkpoint_epoch"],
            samples=int(row["samples"]),
            target_samples=int(row["target_samples"]),
            strict_generated=int(row["strict_generated"]),
            complete_pct=float(row["complete_pct"]),
            vun_pct=float(row["vun_pct"]),
            validity_metric=row.get("validity_metric"),
            sbm_accuracy_refinement_steps=row.get("sbm_accuracy_refinement_steps"),
            sbm_vun_refinement_steps=row.get("sbm_vun_refinement_steps"),
            directly_paper_comparable=bool(row.get("directly_paper_comparable")),
            comparison_protocol=row.get("comparison_protocol"),
            comparability_failures=list(row.get("comparability_failures") or []),
            degree_mmd2=float(row["degree_mmd2"]),
            clustering_mmd2=float(row["clustering_mmd2"]),
            orbit_mmd2=float(row["orbit_mmd2"]),
            degree_ratio=row["degree_ratio"],
            clustering_ratio=row["clustering_ratio"],
            orbit_ratio=row["orbit_ratio"],
            mean_ratio=row["mean_ratio"],
            ratio_metrics_used=int(row["ratio_metrics_used"]),
            denominator_source=row["denominator_source"],
        )
        for row in aggregate["rows"]
    ]
    return aggregate, rows


def fmt_ratio(value: float | None) -> str:
    if value is None:
        return "--"
    if abs(value) >= 1000:
        return f"{value:.1f}"
    if abs(value) >= 100:
        return f"{value:.2f}"
    return f"{value:.3f}"


def fmt_denominator(value: float | None) -> str:
    return "--" if value is None else f"{value:.6f}"


def coherent_choice(rows: list[Result]) -> Result:
    candidates = [row for row in rows if row.mean_ratio is not None]
    if not candidates:
        raise ValueError(f"No defined compatible ratio for {rows[0].dataset}")
    return min(
        candidates,
        key=lambda row: (row.mean_ratio, row.wl, ROLE_ORDER[row.checkpoint_role]),
    )


def build_tex(
    aggregate: dict,
    results: list[Result],
    *,
    run_root: Path,
    ledger: Path,
) -> str:
    by_dataset = {
        dataset: sorted(
            (row for row in results if row.dataset == dataset),
            key=lambda row: (row.wl, ROLE_ORDER[row.checkpoint_role]),
        )
        for dataset in DATASET_ORDER
    }
    stage_count = len({(row.dataset, row.wl) for row in results})
    timestamp = tex_escape(str(aggregate["snapshot"]))

    metric_rows: list[str] = []
    coherent_rows: list[str] = []
    denominator_rows: list[str] = []
    status_rows: list[str] = []

    active_datasets = [dataset for dataset in DATASET_ORDER if by_dataset[dataset]]
    for dataset in active_datasets:
        rows = by_dataset[dataset]
        dataset_tex = tex_escape(dataset)
        selected = coherent_choice(rows)
        mean_marker = r"$^{\dagger}$" if selected.ratio_metrics_used < 3 else ""
        metric_values = (
            str(selected.wl),
            ROLE_SHORT[selected.checkpoint_role],
            fmt_ratio(selected.degree_ratio),
            fmt_ratio(selected.clustering_ratio),
            fmt_ratio(selected.orbit_ratio),
            fmt_ratio(selected.mean_ratio) + mean_marker,
            f"{selected.complete_pct:.2f}\\%",
            f"{selected.vun_pct:.2f}\\%",
        )
        metric_rows.append(dataset_tex + " & " + " & ".join(metric_values) + r" \\")

        if selected.directly_paper_comparable:
            scope = "Direct; " + tex_escape(selected.comparison_protocol or "official protocol")
        elif selected.denominator_source == "spectre_published_table":
            scope = "Not direct; " + tex_escape(", ".join(selected.comparability_failures))
        else:
            scope = "Local train--test denom."
        coherent_rows.append(
            " & ".join(
                (
                    dataset_tex,
                    str(selected.wl),
                    ROLE_SHORT[selected.checkpoint_role],
                    "--" if selected.epoch is None else str(selected.epoch),
                    f"{selected.strict_generated}/{selected.target_samples}",
                    f"{selected.complete_pct:.2f}\\%",
                    f"{selected.degree_mmd2:.6f}",
                    f"{selected.clustering_mmd2:.6f}",
                    f"{selected.orbit_mmd2:.6f}",
                    f"{selected.vun_pct:.2f}\\%",
                    scope,
                )
            )
            + r" \\"
        )

        context = aggregate["dataset_context"][dataset]
        published = aggregate["protocol"]["published_denominators"].get(dataset)
        local = context["local_denominators"]
        published_text = (
            "--"
            if published is None
            else " / ".join(fmt_denominator(published[name]) for name in ("degree", "clustering", "orbit"))
        )
        local_text = " / ".join(fmt_denominator(local[name]) for name in ("degree", "clustering", "orbit"))
        denominator_rows.append(
            f"{dataset_tex} & {context['train_graphs']}/{context['test_graphs']} & "
            f"{tex_escape(context['denominator_source'])} & {published_text} & {local_text} & "
            f"{'Yes' if context.get('paper_dataset_match') else 'No'} \\\\"
        )

        stages = sorted({row.wl for row in rows})
        stage_text = ", ".join(f"WL{stage}" for stage in stages)
        if stages != [1, 2, 3]:
            missing = ", ".join(f"WL{stage}" for stage in sorted({1, 2, 3} - set(stages)))
            stage_text += f"; {missing} unfinished"
        status_rows.append(f"{dataset_tex} & Graph & {stage_text}; three checkpoint roles each \\\\"
        )

    for dataset, family, status in OTHER_BENCHMARKS:
        if dataset in active_datasets:
            continue
        status_rows.append(
            f"{tex_escape(dataset)} & {tex_escape(family)} & {tex_escape(status)} \\\\"
        )

    relative_root = portable_path(run_root)
    relative_ledger = portable_path(ledger)
    direct_datasets = [
        dataset for dataset in active_datasets
        if all(row.directly_paper_comparable for row in results if row.dataset == dataset)
    ]
    non_direct_datasets = [dataset for dataset in active_datasets if dataset not in direct_datasets]
    sbm_rows = [row for row in results if row.dataset == "SBM"]
    exact_sbm = bool(sbm_rows) and all(
        row.validity_metric == "sbm_graph_tool_exact"
        and row.sbm_accuracy_refinement_steps == 100
        and row.sbm_vun_refinement_steps == 1000
        for row in sbm_rows
    )
    missing_stages = [
        f"{tex_escape(dataset)} WL{wl}"
        for dataset in active_datasets
        for wl in (1, 2, 3)
        if not any(row.dataset == dataset and row.wl == wl for row in results)
    ]
    coverage_sentence = (
        "All expected WL1--WL3 stages are complete."
        if not missing_stages
        else "Unfinished stages: " + ", ".join(missing_stages) + "."
    )
    comparison_sentence = (
        "Official asset/split provenance is verified for "
        + (", ".join(tex_escape(dataset) for dataset in direct_datasets) if direct_datasets else "no datasets")
        + "."
    )
    if non_direct_datasets:
        comparison_sentence += (
            " The following remain metric-compatible rather than end-to-end paper reproductions: "
            + ", ".join(tex_escape(dataset) for dataset in non_direct_datasets)
            + "."
        )
    sbm_sentence = (
        "All SBM rows use exact graph-tool validity with 100 accuracy and 1,000 V.U.N. refinements."
        if exact_sbm
        else "SBM V.U.N. is not exact in every SBM row; inspect the recorded validity backend."
    )
    return rf"""\documentclass[10pt]{{article}}
\usepackage[a4paper,margin=1.45cm]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{pdflscape}}
\usepackage{{xcolor}}
\usepackage[hidelinks]{{hyperref}}
\usepackage{{url}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.55em}}

\title{{Current Corrected DiGress/SPECTRE-Compatible Benchmark Results}}
\author{{Anonymous submission}}
\date{{{timestamp}}}

\begin{{document}}
\maketitle

\textbf{{Outcome.}} Degree, clustering, and orbit metrics were recomputed for all
{len(results)} finished permissive evaluations: {stage_count} dataset/WL stages,
each with fixed-$\beta$ best (FB), legacy validation-loss best (LV), and last-epoch
(Last) checkpoints. {coverage_sentence} Exact ORCA node-orbit
counts replace the former orbit proxy. One whole run is selected per benchmark by
the lowest custom three-metric selection mean; all reported metrics for that benchmark use the
same WL iteration and checkpoint. Ties prefer the lower WL and then FB, LV, Last.

\textbf{{Paper convention.}} Every value called MMD$^2$ uses the biased squared-MMD
implementation in the public DiGress/SPECTRE evaluator. Ratio metrics are
$\mathrm{{MMD}}^2(\mathrm{{generated}},\mathrm{{test}})/
\mathrm{{MMD}}^2(\mathrm{{train}},\mathrm{{test}})$. COMMUNITY\_SMALL, PLANAR,
PROTEINS, and SBM use the rounded denominators printed by SPECTRE, as specified by
DiGress Appendix F.1. The remaining datasets use locally recomputed denominators.

\textbf{{Comparability boundary.}} {comparison_sentence} {sbm_sentence}

\begin{{landscape}}
\begin{{table}}[htbp]
\centering
\caption{{Selected best coherent run per benchmark: all metrics use the displayed WL/checkpoint.}}
\label{{tab:metricwise}}
\scriptsize
\setlength{{\tabcolsep}}{{6pt}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrcrrrrrr}}
\toprule
Dataset & WL & Ckpt. & Degree ratio $\downarrow$ & Clustering ratio $\downarrow$ &
Orbit ratio $\downarrow$ & 3-metric selection mean $\downarrow$ & Complete $\uparrow$ & V.U.N. $\uparrow$ \\
\midrule
{chr(10).join(metric_rows)}
\bottomrule
\end{{tabular}}%
}}
\vspace{{0.35em}}

\begin{{minipage}}{{0.98\linewidth}}
\footnotesize
Each row is selected by its lowest three-metric mean and is never assembled from separate
checkpoints. $^\dagger$MUTAG's local train--test clustering MMD$^2$ is exactly zero,
so its clustering ratio is undefined and its displayed mean covers degree and orbit.
This three-metric selection score is not SPECTRE's five-metric Ratio for Planar,
SBM, or Proteins.
\end{{minipage}}
\end{{table}}

\begin{{table}}[htbp]
\centering
\caption{{Raw MMD$^2$ and decoder diagnostics from the same selected runs as Table~\ref{{tab:metricwise}}.}}
\label{{tab:coherent}}
\scriptsize
\setlength{{\tabcolsep}}{{3.3pt}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lrrrcrrrrrl}}
\toprule
Dataset & WL & Ckpt. & Epoch & Strict & Complete $\uparrow$ & Degree MMD$^2$ $\downarrow$ &
Clust. MMD$^2$ $\downarrow$ & Orbit MMD$^2$ $\downarrow$ & V.U.N. $\uparrow$ & Ratio scope \\
\midrule
{chr(10).join(coherent_rows)}
\bottomrule
\end{{tabular}}%
}}
\vspace{{0.35em}}

\begin{{minipage}}{{0.98\linewidth}}
\footnotesize
``Strict'' is complete decoder outputs / target samples. MMD uses every permissive
output, matching the public evaluator's generated-graph treatment. These are the
raw MMD$^2$ numerators corresponding to exactly the selected rows in Table~\ref{{tab:metricwise}}.
\end{{minipage}}
\end{{table}}
\end{{landscape}}

\begin{{table}}[htbp]
\centering
\caption{{Ratio denominator and dataset-provenance audit. Values are Degree / Clustering / Orbit.}}
\label{{tab:denominators}}
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{lcllll}}
\toprule
Dataset & Train/Test & Selected source & Published denominator & Local denominator &
Exact paper dataset/split? \\
\midrule
{chr(10).join(denominator_rows)}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}

\begin{{table}}[htbp]
\centering
\caption{{Coverage of the configured benchmark suite.}}
\label{{tab:coverage}}
\small
\begin{{tabular}}{{llp{{9.3cm}}}}
\toprule
Benchmark & Family & Status \\
\midrule
{chr(10).join(status_rows)}
\bottomrule
\end{{tabular}}
\end{{table}}

\textbf{{Detailed ledger.}} Every row, including raw MMD$^2$, ratios, completion,
V.U.N., exact per-run JSON links, and caveats, is in
\path{{{tex_escape(str(relative_ledger))}}}. Machine-readable aggregates
are in \path{{{relative_root}/digress_spectre_compatible/metrics.json}} and
\path{{{relative_root}/digress_spectre_compatible/metrics.csv}}.

\textbf{{Sources.}} \url{{https://arxiv.org/abs/2209.14734}},
\url{{https://arxiv.org/abs/2204.01613}}, and
\url{{https://github.com/cvignac/DiGress/tree/main/src/analysis}}.

\textbf{{Caution.}} These are single-seed point estimates. ``Current best'' means
best among already-finished artifacts in the scoped corrected sweep.

\end{{document}}
"""


def main() -> None:
    args = parse_args()
    if not args.metrics.exists():
        raise SystemExit(f"Compatible aggregate does not exist: {args.metrics}")
    aggregate, results = load_results(args.metrics)
    unknown = sorted({row.dataset for row in results} - set(DATASET_ORDER))
    if unknown:
        raise SystemExit(f"Unexpected datasets: {unknown}")
    run_root = args.metrics.parent.parent
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_tex(aggregate, results, run_root=run_root, ledger=args.ledger),
        encoding="utf-8",
    )
    print(f"Wrote {args.output} from {len(results)} evaluations.")


if __name__ == "__main__":
    main()
