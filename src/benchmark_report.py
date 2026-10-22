"""Collect benchmark metadata and render CSV, JSON, and Markdown summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SUMMARY_KEYS = [
    "config/model",
    "config/dataset",
    "config/mode",
    "config/reference_split",
    "config/reference_graphs",
    "config/mol_preprocess",
    "config/requested_mol_preprocess",
    "config/paper_split_protocol",
    "config/qm9_protocol",
    "config/zinc250k_protocol",
    "config/fcd_reference_split",
    "config/fcd_reference_graphs",
    "config/checkpoint",
    "config/checkpoint_role",
    "config/checkpoint_epoch",
    "config/checkpoint_selection_metric",
    "config/checkpoint_selection_score",
    "config/checkpoint_legacy_selection_metric",
    "config/checkpoint_legacy_selection_score",
    "config/verify_data_provenance",
    "config/use_budget_context",
    "config/use_edge_budget_context",
    "config/constraint_kind",
    "config/use_action_head",
    "config/use_completion_head",
    "config/use_latent_plan",
    "config/use_plan_context",
    "config/connect_existing_first",
    "config/node_budget_factor",
    "config/node_budget_margin",
    "config/max_restarts",
    "config/max_rings",
    "config/action_temperature",
    "config/plan_temperature",
    "config/plan_prior_weight",
    "config/adaptive_prior_weight",
    "config/adaptive_prior_pseudocount",
    "config/empirical_prior_weight",
    "config/require_complete",
    "config/use_stub_closure",
    "config/use_learned_stop",
    "config/use_learned_actions",
    "config/use_learned_closure",
    "config/final_stage",
    "config/requested_n_wl_iters",
    "sampling/target_n_samples",
    "sampling/generated_graphs",
    "sampling/generation_fraction",
    "sampling/complete_fraction",
    "sampling/ring_complete_fraction",
    "sampling/mean_closure_edges",
    "sampling/mean_frontier_closure_edges",
    "sampling/mean_final_closure_edges",
    "sampling/mean_node_budget_repair_edges",
    "sampling/mean_plan_prefix_match_fraction",
    "sampling/mean_abs_target_node_error",
    "sampling/mean_abs_target_edge_error",
    "sampling/mean_unresolved_stubs",
    "dist/node_count/tv",
    "dist/edge_count/tv",
    "dist/degree/tv",
    "dist/wl_color/tv",
    "mmd/degree",
    "mmd/clustering",
    "mmd/orbit",
    "mmd/reference/degree",
    "mmd/reference/clustering",
    "mmd/reference/orbit",
    "mmd/ratio/degree",
    "mmd/ratio/clustering",
    "mmd/ratio/orbit",
    "mmd/ratio/mean_degree_clustering_orbit",
    "mmd/ratio/status",
    "mmd/spectral",
    "mmd/wavelet",
    "mmd/orbit_4node_proxy",
    "mmd/node_count",
    "mmd/edge_count",
    "mmd/wl_color",
    "graph/connected_fraction",
    "graph/planar_fraction",
    "graph/validity_metric",
    "graph/valid_fraction",
    "graph/valid_clean_fraction",
    "graph/unique_fraction",
    "graph/novel_fraction",
    "graph/valid_unique_novel_fraction",
    "graph/n_unique",
    "graph/n_novel",
    "graph/n_valid_unique_novel",
    "mol/validity",
    "mol/uniqueness",
    "mol/novelty",
    "mol/primary_uniqueness_definition",
    "mol/benchmark_policy",
    "mol/catflow_table_comparable",
    "mol/catflow_comparability_failures",
    "mol/comparison_note",
    "mol/explicit_h/validity",
    "mol/explicit_h/uniqueness",
    "mol/explicit_h/novelty",
    "mol/h_stripped/validity",
    "mol/h_stripped/uniqueness",
    "mol/h_stripped/novelty",
    "mol/hydrogen_removal_uniqueness_loss",
    "mol/fcd",
    "mol/fcd_available",
    "mol/fcd_status",
    "mol/fcd_preprocess",
    "mol/fcd_reference_split",
    "mol/fcd_reference_graphs",
    "mol/fcd_reference_smiles",
    "mol/fcd_generated_smiles",
    "mol/fcd_attempts",
    "mol/fcd_invalid_filtering",
    "mol/fcd_generation_protocol",
    "mol/fcd_duplicate_handling",
    "mol/fcd_reference_duplicate_handling",
    "timing/sample_seconds",
]

VIEW_COMPARISON_KEYS = [
    "config/primary_graph_view",
    "config/secondary_graph_view",
    "config/largest_component_matches_raw",
    "sampling/largest_component_graphs",
    "sampling/largest_component_survival_fraction",
    "graph/raw_mean_component_count",
    "graph/raw_max_component_count",
    "graph/raw_multiple_components_fraction",
    "graph/raw_mean_isolated_nodes",
    "graph/raw_isolate_fraction",
    "graph/raw_mean_largest_component_fraction",
    "graph/cleanup_mean_removed_nodes",
    "graph/cleanup_mean_removed_edges",
]

VIEW_METRIC_BASE_KEYS = [
    "graph/connected_fraction",
    "graph/valid_fraction",
    "graph/unique_fraction",
    "graph/novel_fraction",
    "graph/valid_unique_novel_fraction",
    "dist/node_count/tv",
    "dist/edge_count/tv",
    "dist/degree/tv",
    "dist/wl_color/tv",
    "mmd/degree",
    "mmd/clustering",
    "mmd/orbit",
    "mmd/spectral",
    "mmd/ratio/mean_degree_clustering_orbit",
    "mol/validity",
    "mol/uniqueness",
    "mol/novelty",
    "mol/fcd",
    "mol/fcd_available",
    "mol/fcd_status",
]

SUMMARY_KEYS.extend(VIEW_COMPARISON_KEYS)
SUMMARY_KEYS.extend(f"largest_component/{key}" for key in VIEW_METRIC_BASE_KEYS)


PAPER_BENCHMARKS = {
    "spectre_2204_01613": {
        "experiments": [
            "general graph generation",
            "QM9 molecule generation",
        ],
        "datasets": ["Community-small", "QM9", "Planar", "SBM", "Proteins"],
        "metrics": [
            "MMD degree",
            "MMD clustering",
            "MMD orbit",
            "MMD spectral",
            "MMD wavelet",
            "valid",
            "unique",
            "novel",
        ],
    },
    "catflow_2406_04843": {
        "experiments": [
            "abstract graph generation",
            "QM9/ZINC250K molecular generation",
            "CatFlow-vs-flow-matching ablation with 100%, 20%, and 5% training data",
            "extra SBM/Planar graph results",
        ],
        "datasets": [
            "Ego-small",
            "Community-small",
            "Enzymes",
            "Grid",
            "QM9",
            "ZINC250K",
            "SBM",
            "Planar",
        ],
        "metrics": ["MMD degree", "MMD clustering", "MMD orbit", "valid", "unique", "FCD"],
    },
    "digress_style_extended": {
        "experiments": [
            "DiGress metric/seed-style abstract graph generation extension",
        ],
        "datasets": ["Ego-small", "Enzymes", "Grid"],
        "metrics": [
            "MMD degree",
            "MMD clustering",
            "MMD orbit",
            "valid",
            "unique",
            "novel",
        ],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Collect Ring WL-VAE benchmark metrics into CSV/JSON/Markdown tables.")
    parser.add_argument("--root", default="benchmark_runs", help="Directory containing benchmark output subdirectories.")
    parser.add_argument("--out", default="benchmark_runs/summary", help="Output prefix or directory for summary files.")
    return parser.parse_args()


def infer_benchmark_context(root: Path, path: Path):
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    context = {
        "benchmark/dataset_slug": None,
        "benchmark/variant": None,
        "benchmark/is_main": False,
    }
    if len(parts) >= 4 and parts[-2] == "samples":
        context["benchmark/dataset_slug"] = parts[-4]
        context["benchmark/variant"] = parts[-3]
    elif len(parts) >= 3 and parts[-2] == "samples":
        context["benchmark/dataset_slug"] = parts[-3]
        context["benchmark/variant"] = parts[-3]
    elif len(parts) >= 3:
        context["benchmark/dataset_slug"] = parts[-3]
        context["benchmark/variant"] = parts[-2]
    variant = str(context["benchmark/variant"] or "")
    context["benchmark/is_main"] = variant == "budget" or variant.endswith("_base")
    return context


def read_metric_files(root: Path):
    rows = []
    for path in sorted(root.rglob("metrics.json")):
        try:
            metrics = json.loads(path.read_text())
        except Exception:
            continue
        if "sampling/generated_graphs" not in metrics:
            continue
        row = {
            "metrics_path": str(path),
            **infer_benchmark_context(root, path),
        }
        for key in SUMMARY_KEYS:
            row[key] = metrics.get(key)
        fcd_available = bool(metrics.get("mol/fcd_available") and metrics.get("mol/fcd") is not None)
        row["paper/fcd"] = metrics.get("mol/fcd") if fcd_available else None
        row["paper/fcd_available"] = fcd_available
        rows.append(row)
    rows.sort(key=lambda row: (
        str(row.get("config/dataset") or row.get("benchmark/dataset_slug") or ""),
        0 if row.get("benchmark/is_main") else 1,
        str(row.get("benchmark/variant") or ""),
        str(row.get("metrics_path") or ""),
    ))
    return rows


def write_csv(path: Path, rows):
    fieldnames = [
        "metrics_path",
        "benchmark/dataset_slug",
        "benchmark/variant",
        "benchmark/is_main",
        *SUMMARY_KEYS,
        "paper/fcd",
        "paper/fcd_available",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def pct(value):
    if value is None:
        return "--"
    try:
        return f"{100.0 * float(value):.2f}"
    except (TypeError, ValueError):
        return "--"


def num(value, digits=4):
    if value is None:
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def text(value):
    return "--" if value is None else str(value)


def variant_text(row):
    variant = text(row.get("benchmark/variant"))
    requested_stage = row.get("config/requested_n_wl_iters", row.get("config/final_stage"))
    if requested_stage is None:
        return variant
    return f"{variant} / WL{requested_stage}"


def first_present(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def validity_label(row):
    if row.get("mol/validity") is not None:
        return "rdkit"
    return text(row.get("graph/validity_metric"))


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def base_preference(row):
    variant = str(row.get("benchmark/variant") or "")
    return 0 if variant.endswith("_base") or variant == "base" or variant == "budget" else 1


def finite_or(value, default):
    value = as_float(value)
    return default if value is None else value


def choose_best_row(dataset, rows):
    if dataset == "QM9_H":
        return max(
            rows,
            key=lambda row: (
                finite_or(row.get("mol/explicit_h/uniqueness"), -float("inf")),
                finite_or(row.get("mol/explicit_h/validity"), -float("inf")),
                finite_or(row.get("sampling/complete_fraction"), -float("inf")),
                -base_preference(row),
            ),
        )
    molecule_rows = [row for row in rows if as_float(row.get("mol/fcd")) is not None]
    if molecule_rows:
        return min(
            molecule_rows,
            key=lambda row: (
                finite_or(row.get("mol/fcd"), float("inf")),
                -finite_or(row.get("mol/validity"), -float("inf")),
                -finite_or(row.get("mol/uniqueness"), -float("inf")),
                base_preference(row),
            ),
        )

    ratio_rows = [row for row in rows if as_float(row.get("mmd/ratio/mean_degree_clustering_orbit")) is not None]
    if dataset in {"GRID", "PLANAR"}:
        best_vun = max((finite_or(row.get("graph/valid_unique_novel_fraction"), 0.0) for row in rows), default=0.0)
        if best_vun > 0.0:
            return max(
                rows,
                key=lambda row: (
                    finite_or(row.get("graph/valid_unique_novel_fraction"), 0.0),
                    -finite_or(row.get("mmd/ratio/mean_degree_clustering_orbit"), float("inf")),
                    -finite_or(row.get("dist/wl_color/tv"), float("inf")),
                    -base_preference(row),
                ),
            )
        candidates = ratio_rows or rows
        return min(
            candidates,
            key=lambda row: (
                finite_or(row.get("mmd/ratio/mean_degree_clustering_orbit"), float("inf")),
                finite_or(row.get("dist/wl_color/tv"), float("inf")),
                -finite_or(row.get("sampling/complete_fraction"), -float("inf")),
                base_preference(row),
            ),
        )

    if ratio_rows:
        return min(
            ratio_rows,
            key=lambda row: (
                finite_or(row.get("mmd/ratio/mean_degree_clustering_orbit"), float("inf")),
                -finite_or(row.get("graph/valid_unique_novel_fraction"), -float("inf")),
                finite_or(row.get("dist/wl_color/tv"), float("inf")),
                base_preference(row),
            ),
        )

    return min(
        rows,
        key=lambda row: (
            finite_or(row.get("dist/wl_color/tv"), float("inf")),
            -finite_or(row.get("sampling/complete_fraction"), -float("inf")),
            base_preference(row),
        ),
    )


def best_rows_by_dataset(rows):
    grouped = {}
    for row in rows:
        dataset = str(row.get("config/dataset") or row.get("benchmark/dataset_slug") or "").upper()
        if dataset:
            grouped.setdefault(dataset, []).append(row)
    return {dataset: choose_best_row(dataset, group) for dataset, group in grouped.items()}


def best_selection_basis(dataset, row):
    if dataset == "QM9_H":
        return "highest explicit-H uniqueness"
    if as_float(row.get("mol/fcd")) is not None:
        return "lowest FCD"
    if dataset in {"GRID", "PLANAR"}:
        if finite_or(row.get("graph/valid_unique_novel_fraction"), 0.0) > 0.0:
            return "highest VUN"
        return "best MMD/WL after zero exact validity"
    if as_float(row.get("mmd/ratio/mean_degree_clustering_orbit")) is not None:
        return "lowest ratio MMD"
    return "lowest WL TV"


def best_key_metrics(row):
    if str(row.get("config/dataset") or "").upper() == "QM9_H":
        return (
            f"explicit-H unique {pct(row.get('mol/explicit_h/uniqueness'))}%, "
            f"H-stripped unique {pct(row.get('mol/h_stripped/uniqueness'))}%, "
            f"valid {pct(row.get('mol/explicit_h/validity'))}%"
        )
    if as_float(row.get("mol/fcd")) is not None:
        return (
            f"FCD {num(row.get('mol/fcd'), 4)}, valid {pct(row.get('mol/validity'))}%, "
            f"unique {pct(row.get('mol/uniqueness'))}%, novel {pct(row.get('mol/novelty'))}%"
        )
    return (
        f"ratio {num(row.get('mmd/ratio/mean_degree_clustering_orbit'), 3)}, "
        f"WL TV {num(row.get('dist/wl_color/tv'))}, complete {pct(row.get('sampling/complete_fraction'))}%, "
        f"VUN {pct(row.get('graph/valid_unique_novel_fraction'))}%"
    )


def best_description(dataset, row):
    if dataset == "QM9_H":
        return "Project-specific explicit-H row; compare only with models retrained on explicit-H QM9."
    if as_float(row.get("mol/fcd")) is not None:
        return "Best molecule distribution match in this sweep; keep protocol matched to CatFlow before comparing externally."
    if dataset in {"GRID", "PLANAR"} and finite_or(row.get("graph/valid_unique_novel_fraction"), 0.0) == 0.0:
        return "Best diagnostic row, but exact structural validity is still zero; this needs decoder or constraint changes."
    if dataset == "PROTEINS" and finite_or(row.get("sampling/complete_fraction"), 0.0) == 0.0:
        return "Best weak row by distribution metrics, but completion is zero; do not treat as a strong benchmark result."
    if dataset == "EGO_SMALL":
        return "Best MMD row; check uniqueness/VUN if you want a more balanced reporting variant."
    return "Best row under the primary metric used for this benchmark group."


def write_markdown(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    explicit_main_rows = [row for row in rows if row.get("benchmark/is_main")]
    best_rows = best_rows_by_dataset(rows)
    main_rows = explicit_main_rows or [best_rows[dataset] for dataset in sorted(best_rows)]
    main_row_ids = {id(row) for row in main_rows}
    sbm_validity_kinds = sorted({
        str(row.get("graph/validity_metric"))
        for row in rows
        if str(row.get("config/dataset") or "").upper() == "SBM"
        and row.get("graph/validity_metric")
    })
    if "sbm_graph_tool_exact" in sbm_validity_kinds:
        sbm_validity_note = "SBM validity uses the exact graph-tool Bayesian block-model test."
        sbm_external_note = ""
    elif sbm_validity_kinds:
        sbm_validity_note = (
            "SBM validity is not paper-exact in these rows; recorded backend(s): "
            + ", ".join(f"`{kind}`" for kind in sbm_validity_kinds)
            + "."
        )
        sbm_external_note = " Exact SPECTRE graph-tool SBM validity remains external for these rows."
    else:
        sbm_validity_note = "No SBM validity rows are present in this report."
        sbm_external_note = ""
    if explicit_main_rows:
        ablation_rows = [row for row in rows if not row.get("benchmark/is_main")]
    else:
        ablation_rows = [row for row in rows if id(row) not in main_row_ids]

    main_table = markdown_table(
        [
            "Dataset",
            "Variant",
            "Target",
            "Generated",
            "Generated %",
            "Complete %",
            "Ring complete %",
            "Closure",
            "Abs N-target",
            "Node TV",
            "Edge TV",
            "Degree TV",
            "WL TV",
            "Ref",
            "Validity %",
            "Unique %",
            "Novel %",
        ],
        [
            [
                text(row.get("config/dataset")),
                variant_text(row),
                text(row.get("sampling/target_n_samples")),
                text(row.get("sampling/generated_graphs")),
                pct(row.get("sampling/generation_fraction")),
                pct(row.get("sampling/complete_fraction")),
                pct(row.get("sampling/ring_complete_fraction")),
                num(row.get("sampling/mean_closure_edges"), 2),
                num(row.get("sampling/mean_abs_target_node_error"), 2),
                num(row.get("dist/node_count/tv")),
                num(row.get("dist/edge_count/tv")),
                num(row.get("dist/degree/tv")),
                num(row.get("dist/wl_color/tv")),
                text(row.get("config/reference_split")),
                pct(first_present(row, "mol/validity", "graph/valid_fraction")),
                pct(first_present(row, "mol/uniqueness", "graph/unique_fraction")),
                pct(first_present(row, "mol/novelty", "graph/novel_fraction")),
            ]
            for row in main_rows
        ] or [["--"] * 17],
    )

    ablation_table = markdown_table(
        [
            "Dataset",
            "Variant",
            "Complete %",
            "Plan match %",
            "Node TV",
            "Edge TV",
            "Degree TV",
            "WL TV",
            "Validity %",
        ],
        [
            [
                text(row.get("config/dataset")),
                variant_text(row),
                pct(row.get("sampling/complete_fraction")),
                pct(row.get("sampling/mean_plan_prefix_match_fraction")),
                num(row.get("dist/node_count/tv")),
                num(row.get("dist/edge_count/tv")),
                num(row.get("dist/degree/tv")),
                num(row.get("dist/wl_color/tv")),
                pct(row.get("mol/validity")),
            ]
            for row in ablation_rows
        ] or [["--"] * 9],
    )

    all_metrics_table = markdown_table(
        [
            "Dataset",
            "Variant",
            "Target",
            "Generated",
            "Generated %",
            "Complete %",
            "Ring complete %",
            "Closure",
            "Abs N-target",
            "Node TV",
            "Edge TV",
            "Degree TV",
            "WL TV",
            "MMD degree",
            "MMD clustering",
            "MMD orbit",
            "Ratio MMD",
            "Spectral",
            "Graph valid %",
            "Graph unique %",
            "Graph novel %",
            "Graph VUN %",
            "Mol valid %",
            "Mol unique %",
            "Mol novel %",
            "FCD",
            "Budget factor",
            "Max rings",
            "Max restarts",
            "Adaptive prior",
            "Empirical prior",
            "FCD ref",
            "Preprocess",
            "Seconds",
        ],
        [
            [
                text(row.get("config/dataset")),
                variant_text(row),
                text(row.get("sampling/target_n_samples")),
                text(row.get("sampling/generated_graphs")),
                pct(row.get("sampling/generation_fraction")),
                pct(row.get("sampling/complete_fraction")),
                pct(row.get("sampling/ring_complete_fraction")),
                num(row.get("sampling/mean_closure_edges"), 2),
                num(row.get("sampling/mean_abs_target_node_error"), 2),
                num(row.get("dist/node_count/tv")),
                num(row.get("dist/edge_count/tv")),
                num(row.get("dist/degree/tv")),
                num(row.get("dist/wl_color/tv")),
                num(row.get("mmd/degree"), 6),
                num(row.get("mmd/clustering"), 6),
                num(row.get("mmd/orbit"), 6),
                num(row.get("mmd/ratio/mean_degree_clustering_orbit"), 3),
                num(row.get("mmd/spectral"), 6),
                pct(row.get("graph/valid_fraction")),
                pct(row.get("graph/unique_fraction")),
                pct(row.get("graph/novel_fraction")),
                pct(row.get("graph/valid_unique_novel_fraction")),
                pct(row.get("mol/validity")),
                pct(row.get("mol/uniqueness")),
                pct(row.get("mol/novelty")),
                num(row.get("paper/fcd"), 6),
                num(row.get("config/node_budget_factor"), 2),
                text(row.get("config/max_rings")),
                text(row.get("config/max_restarts")),
                num(row.get("config/adaptive_prior_weight"), 2),
                num(row.get("config/empirical_prior_weight"), 2),
                text(first_present(row, "mol/fcd_reference_split", "config/fcd_reference_split")),
                text(first_present(row, "mol/fcd_preprocess", "config/mol_preprocess")),
                num(row.get("timing/sample_seconds"), 2),
            ]
            for row in rows
        ] or [["--"] * 34],
    )

    graph_mmd_table = markdown_table(
        [
            "Dataset",
            "Variant",
            "Ref",
            "Degree",
            "Clustering",
            "Orbit",
            "Ratio MMD",
            "Spectral",
            "Wavelet",
        ],
        [
            [
                text(row.get("config/dataset")),
                variant_text(row),
                text(row.get("config/reference_split")),
                num(row.get("mmd/degree"), 6),
                num(row.get("mmd/clustering"), 6),
                num(row.get("mmd/orbit"), 6),
                num(row.get("mmd/ratio/mean_degree_clustering_orbit"), 3),
                num(row.get("mmd/spectral"), 6),
                num(row.get("mmd/wavelet"), 6),
            ]
            for row in main_rows
            if row.get("mmd/degree") is not None
            or row.get("mmd/clustering") is not None
            or row.get("mmd/orbit") is not None
        ] or [["--"] * 9],
    )

    graph_validity_table = markdown_table(
        [
            "Dataset",
            "Variant",
            "Target",
            "Generated",
            "Validity metric",
            "Valid %",
            "Unique %",
            "Novel %",
            "VUN %",
        ],
        [
            [
                text(row.get("config/dataset")),
                variant_text(row),
                text(row.get("sampling/target_n_samples")),
                text(row.get("sampling/generated_graphs")),
                validity_label(row),
                pct(first_present(row, "mol/validity", "graph/valid_fraction")),
                pct(first_present(row, "mol/uniqueness", "graph/unique_fraction")),
                pct(first_present(row, "mol/novelty", "graph/novel_fraction")),
                pct(row.get("graph/valid_unique_novel_fraction")),
            ]
            for row in main_rows
            if row.get("graph/valid_fraction") is not None or row.get("mol/validity") is not None
        ] or [["--"] * 9],
    )

    molecule_table = markdown_table(
        [
            "Dataset",
            "Variant",
            "Target",
            "Generated",
            "Valid %",
            "Primary unique %",
            "Primary definition",
            "Explicit-H unique %",
            "H-stripped unique %",
            "H-removal loss (pp)",
            "Novel %",
            "FCD",
            "QM9 CatFlow-comparable?",
        ],
        [
            [
                text(row.get("config/dataset")),
                variant_text(row),
                text(row.get("sampling/target_n_samples")),
                text(row.get("sampling/generated_graphs")),
                pct(row.get("mol/validity")),
                pct(row.get("mol/uniqueness")),
                text(row.get("mol/primary_uniqueness_definition")),
                pct(row.get("mol/explicit_h/uniqueness")),
                pct(row.get("mol/h_stripped/uniqueness")),
                pct(row.get("mol/hydrogen_removal_uniqueness_loss")),
                pct(row.get("mol/novelty")),
                num(row.get("paper/fcd"), 6),
                (
                    "yes" if row.get("mol/catflow_table_comparable")
                    else "no" if str(row.get("config/dataset") or "").upper() in {"QM9", "QM9_H"}
                    else "--"
                ),
            ]
            for row in main_rows
            if row.get("mol/validity") is not None
        ] or [["--"] * 13],
    )

    molecule_protocol_table = markdown_table(
        [
            "Dataset",
            "Variant",
            "Preprocess",
            "QM9 protocol",
            "FCD ref",
            "Ref smiles",
            "Gen smiles",
            "Invalid filtering",
            "Duplicates",
            "Comparison gate",
        ],
        [
            [
                text(row.get("config/dataset")),
                variant_text(row),
                text(first_present(row, "mol/fcd_preprocess", "config/mol_preprocess")),
                (
                    text(row.get("config/qm9_protocol"))
                    if str(row.get("config/dataset") or "").upper() in {"QM9", "QM9_H"}
                    else "--"
                ),
                text(first_present(row, "mol/fcd_reference_split", "config/fcd_reference_split")),
                text(row.get("mol/fcd_reference_smiles")),
                text(row.get("mol/fcd_generated_smiles")),
                text(row.get("mol/fcd_invalid_filtering")),
                f"gen={text(row.get('mol/fcd_duplicate_handling'))}; ref={text(row.get('mol/fcd_reference_duplicate_handling'))}",
                (
                    "pass"
                    if row.get("mol/catflow_table_comparable")
                    else ",".join(row.get("mol/catflow_comparability_failures") or [])
                    if str(row.get("config/dataset") or "").upper() == "QM9"
                    else "project-specific explicit-H"
                    if str(row.get("config/dataset") or "").upper() == "QM9_H"
                    else "--"
                ),
            ]
            for row in main_rows
            if row.get("mol/validity") is not None
        ] or [["--"] * 10],
    )

    component_view_table = markdown_table(
        [
            "Dataset",
            "Variant / WL",
            "Mean components",
            "Disconnected %",
            "Isolate %",
            "Largest nodes %",
            "Raw valid %",
            "Largest valid %",
            "Raw WL TV",
            "Largest WL TV",
            "Raw ratio MMD",
            "Largest ratio MMD",
            "Raw mol valid %",
            "Largest mol valid %",
            "Raw FCD",
            "Largest FCD",
        ],
        [
            [
                text(row.get("config/dataset")),
                variant_text(row),
                num(row.get("graph/raw_mean_component_count"), 2),
                pct(row.get("graph/raw_multiple_components_fraction")),
                pct(row.get("graph/raw_isolate_fraction")),
                pct(row.get("graph/raw_mean_largest_component_fraction")),
                pct(row.get("graph/valid_fraction")),
                pct(row.get("largest_component/graph/valid_fraction")),
                num(row.get("dist/wl_color/tv")),
                num(row.get("largest_component/dist/wl_color/tv")),
                num(row.get("mmd/ratio/mean_degree_clustering_orbit"), 3),
                num(row.get("largest_component/mmd/ratio/mean_degree_clustering_orbit"), 3),
                pct(row.get("mol/validity")),
                pct(row.get("largest_component/mol/validity")),
                num(row.get("mol/fcd"), 6),
                num(row.get("largest_component/mol/fcd"), 6),
            ]
            for row in rows
            if row.get("graph/raw_mean_component_count") is not None
        ] or [["--"] * 16],
    )

    best_table = markdown_table(
        ["Dataset", "Best variant", "Selection basis", "Key metrics", "Description"],
        [
            [
                dataset,
                variant_text(row),
                best_selection_basis(dataset, row),
                best_key_metrics(row),
                best_description(dataset, row),
            ]
            for dataset, row in sorted(best_rows.items())
        ] or [["--"] * 5],
    )

    coverage_rows = []
    for paper, spec in PAPER_BENCHMARKS.items():
        coverage_rows.append([
            paper,
            ", ".join(spec["experiments"]),
            ", ".join(spec["datasets"]),
            ", ".join(spec["metrics"]),
        ])
    coverage_table = markdown_table(["Paper", "Experiments", "Referenced datasets", "Referenced metrics"], coverage_rows)

    path.write_text(
        "\n".join([
            "# Ring WL-VAE Benchmark Summary",
            "",
            "Main model: `budget_ring_wl_vae`, the budget-conditioned Ring WL-VAE with constrained connect-first decoding.",
            "",
            "## Main Results",
            "",
            main_table,
            "",
            "## Ablations",
            "",
            "Rows here are non-selected experiments. For tuning roots, selected best rows are promoted to Main Results and the rest remain here.",
            "",
            ablation_table,
            "",
            "## All Experiment Metrics",
            "",
            "This table includes every metrics row found under the report root. Blank entries mean the metric is not defined for that dataset type.",
            "",
            all_metrics_table,
            "",
            "## Paper Graph MMD",
            "",
            "For Ego-small and Community-small this is the paper table shape: Degree, Clustering, Orbit.",
            "",
            graph_mmd_table,
            "",
            "## Paper Validity/Novelty",
            "",
            "Graph uniqueness/novelty use exact topology isomorphism. Planar validity is exact and Grid validity is grid-isomorphism. " + sbm_validity_note,
            "",
            graph_validity_table,
            "",
            "## Paper Molecule Metrics",
            "",
            "QM9/ZINC250K rows use raw RDKit validity. A CatFlow-comparable row must use the pinned GDSS heavy-atom asset and test holdout, CatFlow preprocessing, exactly 10,000 attempts, released-evaluator FCD duplicate handling, and an available FCD backend. QM9_H is a separate explicit-hydrogen benchmark and must not be compared with the published CatFlow QM9 row.",
            "",
            molecule_table,
            "",
            "## Molecular Protocol",
            "",
            molecule_protocol_table,
            "",
            "## Raw vs Largest Component",
            "",
            "Raw graphs remain the primary benchmark outputs. Largest-component values are diagnostics after removing isolates and retaining only the largest connected component.",
            "",
            component_view_table,
            "",
            "## Benchmark Coverage Notes",
            "",
            coverage_table,
            "",
            "Implemented internally: degree/clustering/spectral MMD, 15-dimensional graphlet orbit MMD for induced graphlets up to 4 nodes, ratio MMD denominators when `--compute-mmd-ratio` is used, target-vs-generated sample counts, node/edge/degree/WL-color TV, connected/planar/grid/SBM-valid graph fractions, exact-isomorphism unique/novel/VUN graph fractions, and molecule validity/uniqueness/novelty.",
            "",
            "External dependencies or metrics: the ORCA binary itself, SPECTRE wavelet MMD, official FCD/ChemNet weights, and MOSES filters/SNN/scaffold metrics." + sbm_external_note + " The Python orbit fallback is exact for Ego-small/Community-small and sampled for larger graphs.",
            "",
            "## Best Variant Recommendations",
            "",
            "These selections are automatic and use the primary metric for each benchmark group: FCD for molecules, ratio MMD for most graph datasets, and exact validity/VUN first for structural-validity datasets when it is nonzero.",
            "",
            best_table,
            "",
        ])
    )


def row_by_dataset(rows):
    out = {}
    for row in rows:
        if not row.get("benchmark/is_main"):
            continue
        dataset = str(row.get("config/dataset") or "").upper()
        if dataset:
            out[dataset] = row
    if not out:
        out = best_rows_by_dataset(rows)
    return out


def write_paper_tables(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    by_dataset = row_by_dataset(rows)

    def graph_row(dataset):
        row = by_dataset.get(dataset, {})
        return [
            dataset,
            text(row.get("sampling/generated_graphs")),
            num(row.get("mmd/degree"), 6),
            num(row.get("mmd/clustering"), 6),
            num(row.get("mmd/orbit"), 6),
            num(row.get("mmd/ratio/mean_degree_clustering_orbit"), 3),
        ]

    def planar_sbm_row(dataset):
        row = by_dataset.get(dataset, {})
        return [
            dataset,
            text(row.get("sampling/generated_graphs")),
            num(row.get("mmd/degree"), 6),
            num(row.get("mmd/clustering"), 6),
            num(row.get("mmd/orbit"), 6),
            pct(row.get("graph/valid_unique_novel_fraction")),
        ]

    def molecule_row(dataset):
        row = by_dataset.get(dataset, {})
        return [
            dataset,
            text(row.get("sampling/generated_graphs")),
            pct(row.get("mol/validity")),
            pct(row.get("mol/uniqueness")),
            num(row.get("paper/fcd"), 6),
        ]

    def ref_row(dataset):
        row = by_dataset.get(dataset, {})
        return [
            dataset,
            num(row.get("mmd/reference/degree"), 6),
            num(row.get("mmd/reference/clustering"), 6),
            num(row.get("mmd/reference/orbit"), 6),
            text(row.get("mmd/ratio/status")),
        ]

    fcd_rows = [by_dataset.get(dataset, {}) for dataset in ("QM9", "ZINC250K")]
    any_fcd_missing = any(not row or not row.get("paper/fcd_available") for row in fcd_rows)
    fcd_note = (
        "FCD is not available for every molecule row in this run. Install or provide an external FCD/ChemNet command and rerun with `FCD_COMMAND`."
        if any_fcd_missing else
        "FCD was computed by the external command configured for this run. See the molecular protocol table for generated/reference duplicate handling."
    )

    path.write_text("\n".join([
        "# Paper-Style Benchmark Tables",
        "",
        f"Source run: `{path.parent.parent}`",
        "",
        "`Ratio MMD` is the mean of generated-vs-reference MMD divided by train-vs-reference MMD for Degree, Clustering, and Orbit. Lower is better; `--` means a reference denominator was missing or zero.",
        "",
        "## Internal Abstract-Graph Diagnostics",
        "",
        "These generic runtime MMDs are diagnostics, not the exact CatFlow table. For the CatFlow main suite use `catflow_compatible/catflow_metrics.md`, which pins the ordered GDSS split, Gaussian-EMD kernels, and exact ORCA features.",
        "",
        markdown_table(
            ["Dataset", "Samples", "Degree ↓", "Clustering ↓", "Orbit ↓", "Diagnostic ratio ↓"],
            [graph_row(dataset) for dataset in ("EGO_SMALL", "COMMUNITY_SMALL", "ENZYMES", "GRID")],
        ),
        "",
        "## Planar and SBM Metrics",
        "",
        markdown_table(
            ["Dataset", "Samples", "Deg ↓", "Clus ↓", "Orb ↓", "V.U.N. ↑"],
            [planar_sbm_row(dataset) for dataset in ("PLANAR", "SBM")],
        ),
        "",
        "## Molecular Metrics",
        "",
        fcd_note,
        "",
        markdown_table(
            ["Dataset", "Samples", "Valid ↑", "Unique ↑", "FCD ↓"],
            [molecule_row(dataset) for dataset in ("QM9", "ZINC250K")],
        ),
        "",
        "Strict CatFlow comparison checklist: match reference set, RDKit preprocessing, 10k generated attempts, invalid-output filtering, duplicate handling, ChemNet/FCD implementation, and EMA-vs-raw checkpoint selection.",
        "",
        "## Ratio Denominators",
        "",
        markdown_table(
            ["Dataset", "Ref Degree MMD", "Ref Clustering MMD", "Ref Orbit MMD", "Status"],
            [ref_row(dataset) for dataset in ("EGO_SMALL", "COMMUNITY_SMALL", "ENZYMES", "GRID", "PLANAR", "SBM")],
        ),
        "",
    ]))


def main():
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out)
    if out.suffix:
        out_prefix = out.with_suffix("")
    else:
        out.mkdir(parents=True, exist_ok=True)
        out_prefix = out / "benchmark_summary"

    rows = read_metric_files(root)
    write_csv(out_prefix.with_suffix(".csv"), rows)
    write_markdown(out_prefix.with_suffix(".md"), rows)
    paper_tables_path = out_prefix.parent / "paper_metric_tables.md"
    write_paper_tables(paper_tables_path, rows)
    out_prefix.with_suffix(".json").write_text(json.dumps({
        "main_model": "budget_ring_wl_vae",
        "paper_benchmarks": PAPER_BENCHMARKS,
        "notes": {
            "main_model": "Budget-conditioned Ring WL-VAE with constrained connect-first decoding.",
            "ablation_policy": "Internal alternatives should be reported only as ablations.",
            "mmd/orbit": "15-dimensional graphlet orbit MMD for induced graphlets up to 4 nodes; exact for Ego-small/Community-small and sampled for larger graphs.",
            "mmd/orbit_4node_proxy": "Backward-compatible alias for mmd/orbit.",
            "mmd/spectral": "Histogram MMD over normalized Laplacian eigenvalues.",
            "mmd/wavelet": "External-only placeholder for SPECTRE wavelet MMD.",
            "graph/valid_unique_novel_fraction": "VUN over raw generated graph slots, exact topology uniqueness/novelty, and dataset-aware graph validity.",
            "largest_component/...": "Diagnostic metrics after removing isolates and retaining only the largest connected component; raw metrics remain primary.",
            "paper/fcd_available": "False unless --fcd-command/FCD_COMMAND writes or prints an FCD value.",
            "mol/fcd_generation_protocol": "By default FCD uses all RDKit-valid outputs among the requested generation attempts; it does not resample until 10000 valid molecules.",
            "mol/fcd_duplicate_handling": "Generated/reference duplicate handling is recorded per row; CatFlow rows require deduplicated generated valid SMILES.",
        },
        "rows": rows,
    }, indent=2, sort_keys=True))
    print(f"Wrote {out_prefix.with_suffix('.csv')}")
    print(f"Wrote {out_prefix.with_suffix('.json')}")
    print(f"Wrote {out_prefix.with_suffix('.md')}")
    print(f"Wrote {paper_tables_path}")


if __name__ == "__main__":
    main()
