"""Generate graphs from a checkpoint and compute evaluation diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import networkx as nx
import torch

# Import this before gpt_datasets/zinc_utils load RDKit.  The digress
# environment's RDKit and graph-tool builds otherwise collide in
# Boost.Python's converter registry during exact SBM evaluation.
from run_wl_witness_ar import distribution_metrics, generic_mmd_metrics, graph_metrics, json_ready
from check_ring_vae_checkpoint import checkpoint_compatibility_errors
from gpt_datasets import (
    PAPER_SPLIT_PROTOCOLS,
    QM9_PROTOCOLS,
    SUPPORTED_DATASETS,
    ZINC250K_PROTOCOLS,
    load_graph_dataset_bundle,
    normalize_dataset_name,
)
from graph_helpers import postprocess_graph
from project_paths import portable_path
from ring_vae_generator import RingWLVAEGenerator, ring_vae_model_from_checkpoint
from ring_traces import graph_collection_fingerprint
from wl_witness_ar import sample_graph_budgets
from zinc_utils import build_smiles, build_training_smiles, compute_fcd_with_command, compute_mol_metrics


def make_parser():
    parser = argparse.ArgumentParser(description="Sample graphs from a ring-structured WL VAE.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--allow-checkpoint-mismatch", action="store_true",
                        help="Allow dataset/WL-provenance mismatches when evaluating legacy checkpoints.")
    parser.add_argument(
        "--verify-data-provenance",
        action="store_true",
        help="Require checkpoint train/validation fingerprints and dataset provenance to match this run.",
    )
    parser.add_argument("--dataset", default="ZINC250K",
                        help=f"Dataset name or paper alias. Supported canonical names: {', '.join(SUPPORTED_DATASETS)}")
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--paper-split-protocol",
        default="legacy_local",
        choices=PAPER_SPLIT_PROTOCOLS,
        help="Dataset/split provenance for paper graph benchmarks.",
    )
    parser.add_argument(
        "--qm9-protocol",
        default="legacy_local",
        choices=QM9_PROTOCOLS,
        help="QM9 representation/split protocol; gdss_catflow is heavy-atom QM9 only.",
    )
    parser.add_argument(
        "--zinc250k-protocol",
        default="legacy_pyg",
        choices=ZINC250K_PROTOCOLS,
        help="ZINC250K representation/split protocol.",
    )
    parser.add_argument("--zinc-subset", default=None, choices=["small", "large"])
    parser.add_argument("--smiles-file", default=None)
    parser.add_argument("--moses-split", default="train", choices=["train", "test", "test_scaffolds"])
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--output-dir", default="inference_runs/ring_vae/default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", default="auto", choices=["auto", "generic", "molecule"])
    parser.add_argument("--max-train-graphs", type=int, default=None)
    parser.add_argument("--max-val-graphs", type=int, default=None)
    parser.add_argument("--metric-max-train-graphs", type=int, default=None)
    parser.add_argument("--mmd-max-graphs", type=int, default=2000)
    parser.add_argument("--compute-mmd-ratio", action="store_true",
                        help="Also compute train-vs-reference MMD denominators and generated/reference ratio MMDs.")
    parser.add_argument(
        "--sbm-validity-backend",
        default="auto",
        choices=("auto", "exact", "heuristic"),
        help="Use graph-tool for the exact DiGress/SPECTRE SBM test, or the legacy heuristic.",
    )
    parser.add_argument(
        "--sbm-refinement-steps",
        type=int,
        default=100,
        help="Merge-split MCMC refinements for exact SBM validity; DiGress sampling metrics use 100.",
    )
    parser.add_argument(
        "--sbm-vun-refinement-steps",
        type=int,
        default=1000,
        help="Merge-split MCMC refinements for SBM V.U.N.; DiGress is_sbm_graph defaults to 1000.",
    )
    parser.add_argument("--reference-split", default="train", choices=["train", "val", "test"],
                        help="Dataset split used as the reference distribution for TV/MMD metrics.")
    parser.add_argument("--mol-preprocess", default="repo", choices=["repo", "catflow"],
                        help=(
                            "Molecule canonicalization for primary metrics and FCD SMILES. "
                            "QM9_H always uses representation-preserving repo mode as primary and "
                            "adds H-stripped CatFlow metrics as a secondary view."
                        ))
    parser.add_argument("--train-smiles-cache", default=None)
    parser.add_argument("--fcd-reference-split", default=None, choices=["train", "val", "test", "all"],
                        help="Reference split used for FCD. Defaults to --reference-split.")
    parser.add_argument("--fcd-deduplicate-reference", action="store_true",
                        help="Deduplicate reference SMILES before FCD.")
    parser.add_argument(
        "--fcd-deduplicate-generated",
        action="store_true",
        help="Deduplicate valid generated SMILES before FCD, matching the released CatFlow evaluator.",
    )
    parser.add_argument("--fcd-command", default=None,
                        help="External FCD command. May use {ref}, {gen}, and {out} placeholders.")
    parser.add_argument("--fcd-timeout", type=float, default=1200.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--edge-temperature", type=float, default=1.0)
    parser.add_argument("--action-temperature", type=float, default=1.0)
    parser.add_argument("--plan-temperature", type=float, default=1.0)
    parser.add_argument("--plan-prior-weight", type=float, default=3.0,
                        help="Logit bonus when the AR decoder's next color matches a latent/external plan.")
    parser.add_argument("--plan-greedy", action="store_true",
                        help="Greedily decode the latent structured plan before AR sampling.")
    parser.add_argument("--empirical-prior-weight", type=float, default=0.0,
                        help="Add this multiplier times log(train WL-color frequency) to color logits.")
    parser.add_argument("--adaptive-prior-weight", type=float, default=0.0,
                        help="Bias color logits toward WL colors underrepresented so far in this sampling run.")
    parser.add_argument("--adaptive-prior-pseudocount", type=float, default=1000.0,
                        help="Pseudo-count mass for the adaptive WL-color prior.")
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no-learned-actions", action="store_true",
                        help="Disable the learned connect/create action policy even when the checkpoint has one.")
    parser.add_argument("--no-learned-stop", action="store_true",
                        help="Disable the learned stop/termination decision when the checkpoint has one.")
    parser.add_argument("--no-learned-closure", action="store_true",
                        help="Disable learned actions inside the final stub-closure pass.")
    parser.add_argument("--allow-disconnected", action="store_true")
    parser.add_argument("--require-complete", dest="require_complete", action="store_true", default=True,
                        help="Require all WL stubs to be satisfied before accepting a generated graph (default).")
    parser.add_argument("--allow-incomplete", dest="require_complete", action="store_false",
                        help="Accept connected partial graphs and report strict completion separately.")
    parser.add_argument("--no-connect-existing-first", action="store_true",
                        help="Disable frontier-time closure against compatible existing nodes.")
    parser.add_argument("--no-stub-closure", action="store_true",
                        help="Disable the final pass that tries to satisfy remaining stubs using existing nodes.")
    parser.add_argument("--max-restarts", type=int, default=200)
    parser.add_argument("--max-rings", type=int, default=12)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--node-budget-factor", type=float, default=1.0,
                        help="When --max-nodes is omitted, cap each sample at ceil(target_nodes * factor) + margin.")
    parser.add_argument("--node-budget-margin", type=int, default=0)
    parser.add_argument("--max-closure-steps-per-node", type=int, default=20)
    parser.add_argument("--diagnostic-max-train-graphs", type=int, default=20000,
                        help="Training-graph cap for complete/incomplete subset diagnostics.")
    parser.add_argument("--no-diagnostic-subsets", action="store_true")
    return parser


def validate_checkpoint_for_run(ckpt, args, bundle=None):
    provenance_args = {}
    if args.verify_data_provenance and bundle is not None:
        provenance_args = {
            "expected_data_provenance": bundle.provenance,
            "expected_train_fingerprint": graph_collection_fingerprint(bundle.train_graphs),
            "expected_val_fingerprint": graph_collection_fingerprint(bundle.val_graphs),
        }
    errors = checkpoint_compatibility_errors(
        ckpt,
        dataset=args.dataset,
        **provenance_args,
    )

    if errors and not args.allow_checkpoint_mismatch:
        joined = "; ".join(errors)
        raise RuntimeError(
            f"Checkpoint is not compatible with this evaluation run: {joined}. "
            "Use --allow-checkpoint-mismatch only for intentional legacy diagnostics."
        )
    return errors


def resolve_device(name):
    if name and str(name).lower() != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prefixed(prefix, values):
    return {f"{prefix}/{key}": value for key, value in values.items()}


def _fmt_metric(value, fmt=".4f"):
    if value is None:
        return "n/a"
    return format(float(value), fmt)


def _add_mmd_ratio_metrics(metrics, reference_mmd):
    ratio_keys = ("degree", "clustering", "orbit")
    ratios = []
    missing = []
    for key in ratio_keys:
        gen_value = metrics.get(f"mmd/{key}")
        ref_value = reference_mmd.get(f"mmd/{key}")
        metrics[f"mmd/reference/{key}"] = ref_value
        if gen_value is None or ref_value is None or float(ref_value) <= 0.0:
            metrics[f"mmd/ratio/{key}"] = None
            missing.append(key)
            continue
        ratio = float(gen_value) / float(ref_value)
        metrics[f"mmd/ratio/{key}"] = ratio
        ratios.append(ratio)

    metrics["mmd/ratio/mean_degree_clustering_orbit"] = (
        sum(ratios) / len(ratios) if len(ratios) == len(ratio_keys) else None
    )
    metrics["mmd/ratio/status"] = "ok" if not missing else f"missing_or_zero_reference:{','.join(missing)}"


def _graphs_for_reference_split(bundle, split: str):
    split = str(split)
    if split == "train":
        return bundle.train_graphs
    if split == "val":
        return bundle.val_graphs
    if split == "test":
        return bundle.test_graphs
    if split == "all":
        return list(bundle.train_graphs) + list(bundle.val_graphs) + list(bundle.test_graphs)
    raise ValueError(f"Unsupported reference split: {split}")


def largest_component_views(samples):
    graphs = []
    source_attempt_indices = []
    for attempt_idx, sample in enumerate(samples):
        graph = sample.graph
        if graph is None:
            continue
        components = list(nx.connected_components(graph)) if graph.number_of_nodes() > 0 else []
        n_nodes = int(graph.number_of_nodes())
        n_edges = int(graph.number_of_edges())
        largest_nodes = max((len(component) for component in components), default=0)
        cleaned = postprocess_graph(graph, min_nodes=2)
        cleaned_nodes = int(cleaned.number_of_nodes()) if cleaned is not None else 0
        cleaned_edges = int(cleaned.number_of_edges()) if cleaned is not None else 0
        sample.stats.update({
            "raw_n_components": len(components),
            "raw_n_isolates": len(list(nx.isolates(graph))),
            "raw_largest_component_nodes": largest_nodes,
            "raw_largest_component_fraction": largest_nodes / max(n_nodes, 1),
            "largest_component_available": cleaned is not None,
            "largest_component_n_nodes": cleaned_nodes,
            "largest_component_n_edges": cleaned_edges,
            "cleanup_removed_nodes": n_nodes - cleaned_nodes,
            "cleanup_removed_edges": n_edges - cleaned_edges,
        })
        if cleaned is not None:
            graphs.append(cleaned)
            source_attempt_indices.append(attempt_idx)
    return graphs, source_attempt_indices


def main():
    args = make_parser().parse_args()
    args.dataset = normalize_dataset_name(args.dataset)
    requested_mol_preprocess = args.mol_preprocess
    if args.dataset == "QM9_H" and args.mol_preprocess != "repo":
        print(
            "QM9_H requires explicit-H primary metrics; overriding --mol-preprocess catflow "
            "with repo and reporting CatFlow H-stripped metrics under mol/h_stripped/*.",
            flush=True,
        )
        args.mol_preprocess = "repo"
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    device = resolve_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_for_run(ckpt, args)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")

    checkpoint_name = Path(args.checkpoint).name
    checkpoint_role = ckpt.get("checkpoint_role") or {
        "best_ring_vae.pt": "fixed_beta_best",
        "best_legacy_ring_vae.pt": "legacy_warmup_weighted_val_loss_best",
        "last_ring_vae.pt": "last_epoch",
    }.get(checkpoint_name, "unspecified")
    checkpoint_history = ckpt.get("history") or []
    checkpoint_epoch = checkpoint_history[-1].get("epoch") if checkpoint_history else None
    checkpoint_selection = ckpt.get("selection") or {}
    print(f"Checkpoint role: {checkpoint_role}; epoch: {checkpoint_epoch}")
    metadata = ckpt["metadata"]
    final_stage = int(ckpt["final_stage"])
    color_specs = ckpt["color_specs"]

    bundle = load_graph_dataset_bundle(
        args.dataset,
        data_root=args.data_root,
        zinc_subset=args.zinc_subset,
        max_train_graphs=args.max_train_graphs,
        max_val_graphs=args.max_val_graphs,
        smiles_file=args.smiles_file,
        moses_split=args.moses_split,
        seed=args.seed,
        paper_split_protocol=args.paper_split_protocol,
        qm9_protocol=args.qm9_protocol,
        zinc250k_protocol=args.zinc250k_protocol,
    )
    mismatch_warnings = validate_checkpoint_for_run(ckpt, args, bundle)
    model = ring_vae_model_from_checkpoint(ckpt, device)
    mode = args.mode
    if mode == "auto":
        mode = bundle.mode
    reference_graphs = {
        "train": bundle.train_graphs,
        "val": bundle.val_graphs,
        "test": bundle.test_graphs,
    }[args.reference_split]
    fcd_reference_split = args.fcd_reference_split or args.reference_split
    fcd_reference_graphs = _graphs_for_reference_split(bundle, fcd_reference_split)

    generator = RingWLVAEGenerator(
        model=model,
        metadata=metadata,
        final_stage=final_stage,
        color_specs=color_specs,
        temperature=args.temperature,
        edge_temperature=args.edge_temperature,
        action_temperature=args.action_temperature,
        plan_temperature=args.plan_temperature,
        plan_prior_weight=args.plan_prior_weight,
        plan_greedy=args.plan_greedy,
        empirical_prior_weight=args.empirical_prior_weight,
        adaptive_prior_weight=args.adaptive_prior_weight,
        adaptive_prior_pseudocount=args.adaptive_prior_pseudocount,
        greedy=args.greedy,
        use_learned_actions=not args.no_learned_actions,
        use_learned_stop=not args.no_learned_stop,
        use_learned_closure=not args.no_learned_closure,
        require_connected=not args.allow_disconnected,
        require_complete=args.require_complete,
        connect_existing_first=not args.no_connect_existing_first,
        use_stub_closure=not args.no_stub_closure,
        max_restarts=args.max_restarts,
        max_rings=args.max_rings,
        max_nodes=args.max_nodes,
        node_budget_factor=args.node_budget_factor,
        node_budget_margin=args.node_budget_margin,
        max_closure_steps_per_node=args.max_closure_steps_per_node,
        constraint_kind=str(metadata.get("constraint_kind", "generic")),
        molecule_constraint_profile=metadata.get("molecule_constraint_profile"),
        rng=rng,
    )

    sampled_budgets = sample_graph_budgets(metadata, args.n_samples, rng)
    node_counts = [int(item["nodes"]) for item in sampled_budgets]
    edge_counts = [item.get("edges") for item in sampled_budgets]
    constraint_profiles = [dict(item) for item in sampled_budgets]
    t0 = time.perf_counter()
    samples = generator.sample_many(
        node_counts,
        edge_counts=edge_counts,
        constraint_profiles=constraint_profiles,
    )
    sample_seconds = time.perf_counter() - t0
    generated_samples = [sample for sample in samples if sample.graph is not None]
    generated_graphs = [sample.graph for sample in generated_samples]
    largest_component_graphs, largest_component_source_attempt_indices = largest_component_views(samples)
    largest_component_matches_raw = (
        len(largest_component_graphs) == len(generated_graphs)
        and all(
            int(sample.stats.get("cleanup_removed_nodes", 0)) == 0
            and int(sample.stats.get("cleanup_removed_edges", 0)) == 0
            for sample in generated_samples
        )
    )
    complete_graphs = [
        sample.graph for sample in generated_samples
        if bool(sample.stats.get("complete", False))
    ]
    incomplete_graphs = [
        sample.graph for sample in generated_samples
        if not bool(sample.stats.get("complete", False))
    ]

    failure_counts = {}
    complete = 0
    complete_before_closure = 0
    closure_edges = []
    frontier_closure_edges = []
    final_closure_edges = []
    node_budget_repair_edges = []
    unresolved = []
    restarts = []
    latent_draws = []
    decode_attempts = []
    rings = []
    target_errors = []
    target_abs_errors = []
    target_edge_errors = []
    target_edge_abs_errors = []
    plan_match_fracs = []
    plan_lengths = []
    action_counts = {}
    for sample in samples:
        if sample.graph is None:
            failure = sample.stats.get("failure", "failed")
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
            for key, value in (sample.stats.get("failure_counts") or {}).items():
                failure_counts[f"inner/{key}"] = failure_counts.get(f"inner/{key}", 0) + int(value)
            latent_draws.append(int(sample.stats.get("latent_draws", 1)))
            decode_attempts.append(int(sample.stats.get("decode_attempts", args.max_restarts)))
            continue
        complete += int(bool(sample.stats.get("complete", False)))
        complete_before_closure += int(bool(sample.stats.get("complete_before_closure", False)))
        closure_edges.append(int(sample.stats.get("closure_edges", 0)))
        frontier_closure_edges.append(int(sample.stats.get("frontier_closure_edges", 0)))
        final_closure_edges.append(int(sample.stats.get("final_closure_edges", 0)))
        node_budget_repair_edges.append(int(sample.stats.get("node_budget_repair_edges", 0)))
        unresolved.append(int(sample.stats.get("unresolved_stubs", 0)))
        restarts.append(int(sample.stats.get("restarts_used", 0)))
        latent_draws.append(int(sample.stats.get("latent_draws", 1)))
        decode_attempts.append(int(sample.stats.get("decode_attempts", int(sample.stats.get("restarts_used", 0)) + 1)))
        rings.append(int(sample.stats.get("n_rings", 0)))
        for key, value in (sample.stats.get("action_counts") or {}).items():
            action_counts[str(key)] = action_counts.get(str(key), 0) + int(value)
        target_error = int(sample.stats.get("n_nodes", 0)) - int(sample.stats.get("target_nodes", 0))
        target_errors.append(target_error)
        target_abs_errors.append(abs(target_error))
        target_edges = sample.stats.get("target_edges")
        if target_edges is not None:
            target_edge_error = int(sample.stats.get("n_edges", 0)) - int(target_edges)
            target_edge_errors.append(target_edge_error)
            target_edge_abs_errors.append(abs(target_edge_error))
        plan_len = int(sample.stats.get("plan_length", 0))
        if plan_len > 0:
            denom = max(1, min(int(sample.stats.get("n_nodes", 0)), plan_len))
            plan_match_fracs.append(float(sample.stats.get("plan_prefix_matches", 0)) / denom)
            plan_lengths.append(plan_len)

    metrics = {
        "config/model": "ring_wl_vae",
        "config/model_schema_version": ckpt.get("model_schema_version"),
        "config/primary_graph_view": "raw",
        "config/secondary_graph_view": "isolates_removed_then_largest_connected_component",
        "config/largest_component_matches_raw": largest_component_matches_raw,
        "config/use_budget_context": bool(getattr(model, "use_budget_context", False)),
        "config/use_edge_budget_context": bool(getattr(model, "use_edge_budget_context", False)),
        "config/constraint_kind": str(metadata.get("constraint_kind", "generic")),
        "config/decoder_color_backoff": metadata.get("decoder_color_backoff"),
        "config/requested_n_wl_iters": metadata.get("requested_n_wl_iters", final_stage),
        "config/use_action_head": bool(getattr(model, "use_action_head", False)),
        "config/use_completion_head": bool(getattr(model, "use_completion_head", False)),
        "config/use_plan_context": bool(getattr(model, "use_plan_context", False)),
        "config/use_latent_plan": bool(getattr(model, "use_latent_plan", False)),
        "config/use_learned_actions": bool(generator.use_learned_actions),
        "config/use_learned_stop": bool(generator.use_learned_stop),
        "config/use_learned_closure": bool(generator.use_learned_closure),
        "config/dataset": args.dataset,
        "config/dataset_provenance": bundle.provenance,
        "config/paper_split_protocol": args.paper_split_protocol,
        "config/qm9_protocol": args.qm9_protocol,
        "config/zinc250k_protocol": args.zinc250k_protocol,
        "config/sbm_validity_backend": args.sbm_validity_backend,
        "config/sbm_refinement_steps": int(args.sbm_refinement_steps),
        "config/sbm_vun_refinement_steps": int(args.sbm_vun_refinement_steps),
        "config/mode": mode,
        "config/reference_split": args.reference_split,
        "config/reference_graphs": len(reference_graphs),
        "config/mol_preprocess": args.mol_preprocess,
        "config/requested_mol_preprocess": requested_mol_preprocess,
        "config/fcd_reference_split": fcd_reference_split,
        "config/fcd_reference_graphs": len(fcd_reference_graphs),
        "config/checkpoint": portable_path(args.checkpoint),
        "config/checkpoint_role": checkpoint_role,
        "config/checkpoint_epoch": checkpoint_epoch,
        "config/checkpoint_selection_metric": checkpoint_selection.get("metric"),
        "config/checkpoint_selection_score": checkpoint_selection.get("score"),
        "config/checkpoint_legacy_selection_metric": checkpoint_selection.get("legacy_metric"),
        "config/checkpoint_legacy_selection_score": checkpoint_selection.get("legacy_score"),
        "config/checkpoint_mismatch_warnings": mismatch_warnings,
        "config/verify_data_provenance": bool(args.verify_data_provenance),
        "config/final_stage": final_stage,
        "config/n_samples": int(args.n_samples),
        "config/max_restarts": int(args.max_restarts),
        "config/max_rings": int(args.max_rings),
        "config/max_nodes": args.max_nodes,
        "config/node_budget_factor": float(args.node_budget_factor),
        "config/node_budget_margin": int(args.node_budget_margin),
        "config/action_temperature": float(args.action_temperature),
        "config/plan_temperature": float(args.plan_temperature),
        "config/plan_prior_weight": float(args.plan_prior_weight),
        "config/plan_greedy": bool(args.plan_greedy),
        "config/empirical_prior_weight": float(args.empirical_prior_weight),
        "config/adaptive_prior_weight": float(args.adaptive_prior_weight),
        "config/adaptive_prior_pseudocount": float(args.adaptive_prior_pseudocount),
        "config/require_complete": bool(args.require_complete),
        "config/connect_existing_first": not args.no_connect_existing_first,
        "config/use_stub_closure": not args.no_stub_closure,
        "timing/sample_seconds": sample_seconds,
        "sampling/target_n_samples": int(args.n_samples),
        "sampling/generated_graphs": len(generated_graphs),
        "sampling/largest_component_graphs": len(largest_component_graphs),
        "sampling/generation_fraction": len(generated_graphs) / max(args.n_samples, 1),
        "sampling/largest_component_survival_fraction": len(largest_component_graphs) / max(args.n_samples, 1),
        "sampling/strict_generation_fraction": complete / max(args.n_samples, 1),
        "sampling/permissive_generation_fraction": len(generated_graphs) / max(args.n_samples, 1),
        "sampling/complete_fraction": complete / max(args.n_samples, 1),
        "sampling/complete_given_graph_fraction": complete / max(len(generated_graphs), 1),
        "sampling/ring_complete_fraction": complete_before_closure / max(args.n_samples, 1),
        "sampling/ring_complete_given_graph_fraction": complete_before_closure / max(len(generated_graphs), 1),
        "sampling/mean_closure_edges": sum(closure_edges) / max(len(closure_edges), 1),
        "sampling/mean_frontier_closure_edges": sum(frontier_closure_edges) / max(len(frontier_closure_edges), 1),
        "sampling/mean_final_closure_edges": sum(final_closure_edges) / max(len(final_closure_edges), 1),
        "sampling/mean_node_budget_repair_edges": (
            sum(node_budget_repair_edges) / max(len(node_budget_repair_edges), 1)
        ),
        "sampling/mean_unresolved_stubs": sum(unresolved) / max(len(unresolved), 1),
        "sampling/mean_restarts_success": sum(restarts) / max(len(restarts), 1),
        "sampling/latent_draws": sum(latent_draws),
        "sampling/mean_decode_attempts": sum(decode_attempts) / max(len(decode_attempts), 1),
        "sampling/mean_rings": sum(rings) / max(len(rings), 1),
        "sampling/mean_target_node_error": sum(target_errors) / max(len(target_errors), 1),
        "sampling/mean_abs_target_node_error": sum(target_abs_errors) / max(len(target_abs_errors), 1),
        "sampling/mean_target_edge_error": sum(target_edge_errors) / max(len(target_edge_errors), 1),
        "sampling/mean_abs_target_edge_error": sum(target_edge_abs_errors) / max(len(target_edge_abs_errors), 1),
        "sampling/mean_plan_prefix_match_fraction": (
            sum(plan_match_fracs) / len(plan_match_fracs) if plan_match_fracs else None
        ),
        "plan/mean_length": sum(plan_lengths) / len(plan_lengths) if plan_lengths else 0.0,
        "sampling/action_counts": action_counts,
        "sampling/failure_counts": failure_counts,
    }
    raw_graph_metrics = graph_metrics(
        generated_graphs,
        train_graphs=bundle.train_graphs,
        dataset=args.dataset,
        sbm_validity_backend=args.sbm_validity_backend,
        sbm_refinement_steps=args.sbm_refinement_steps,
        sbm_vun_refinement_steps=args.sbm_vun_refinement_steps,
    )
    raw_distribution_metrics = distribution_metrics(
        reference_graphs,
        generated_graphs,
        metadata,
        final_stage,
        args.metric_max_train_graphs,
    )
    raw_mmd_metrics = generic_mmd_metrics(
        reference_graphs,
        generated_graphs,
        metadata,
        final_stage,
        args.mmd_max_graphs,
    )

    if largest_component_matches_raw:
        largest_graph_metrics = dict(raw_graph_metrics)
        largest_distribution_metrics = dict(raw_distribution_metrics)
        largest_mmd_metrics = dict(raw_mmd_metrics)
    else:
        largest_graph_metrics = graph_metrics(
            largest_component_graphs,
            train_graphs=bundle.train_graphs,
            dataset=args.dataset,
            sbm_validity_backend=args.sbm_validity_backend,
            sbm_refinement_steps=args.sbm_refinement_steps,
            sbm_vun_refinement_steps=args.sbm_vun_refinement_steps,
        )
        largest_distribution_metrics = distribution_metrics(
            reference_graphs,
            largest_component_graphs,
            metadata,
            final_stage,
            args.metric_max_train_graphs,
        )
        largest_mmd_metrics = generic_mmd_metrics(
            reference_graphs,
            largest_component_graphs,
            metadata,
            final_stage,
            args.mmd_max_graphs,
        )

    if args.compute_mmd_ratio and mode == "generic" and args.reference_split != "train":
        reference_mmd = generic_mmd_metrics(
            bundle.train_graphs,
            reference_graphs,
            metadata,
            final_stage,
            args.mmd_max_graphs,
        )
        _add_mmd_ratio_metrics(raw_mmd_metrics, reference_mmd)
        _add_mmd_ratio_metrics(largest_mmd_metrics, reference_mmd)

    raw_metric_suite = {
        **raw_graph_metrics,
        **raw_distribution_metrics,
        **raw_mmd_metrics,
    }
    largest_metric_suite = {
        **largest_graph_metrics,
        **largest_distribution_metrics,
        **largest_mmd_metrics,
    }
    metrics.update(raw_metric_suite)
    metrics.update(_prefixed("raw", raw_metric_suite))
    metrics.update(_prefixed("largest_component", largest_metric_suite))
    if not args.no_diagnostic_subsets:
        metrics["diagnostics/subset_train_graph_cap"] = int(args.diagnostic_max_train_graphs)
        metrics["diagnostics/complete_graphs"] = len(complete_graphs)
        metrics["diagnostics/incomplete_graphs"] = len(incomplete_graphs)
        if complete_graphs:
            metrics.update(_prefixed("complete", distribution_metrics(
                reference_graphs,
                complete_graphs,
                metadata,
                final_stage,
                args.diagnostic_max_train_graphs,
            )))
        if incomplete_graphs:
            metrics.update(_prefixed("incomplete", distribution_metrics(
                reference_graphs,
                incomplete_graphs,
                metadata,
                final_stage,
                args.diagnostic_max_train_graphs,
            )))

    if mode == "molecule" and bundle.atom_decoder is not None:
        if bundle.train_smiles and args.mol_preprocess == "repo":
            train_smiles = bundle.train_smiles
        else:
            cache_path = args.train_smiles_cache
            if cache_path is None:
                cache_path = output_dir / f"train_smiles_{args.mol_preprocess}_dedup_cache.json"
            train_smiles = build_training_smiles(
                bundle.train_graphs,
                bundle.atom_decoder,
                cache_path=cache_path,
                preprocess=args.mol_preprocess,
            )
        ref_dup_label = "deduplicated" if args.fcd_deduplicate_reference else "retained"
        reference_smiles = build_smiles(
            fcd_reference_graphs,
            bundle.atom_decoder,
            cache_path=output_dir / f"fcd_{fcd_reference_split}_{args.mol_preprocess}_{ref_dup_label}_smiles_cache.json",
            preprocess=args.mol_preprocess,
            deduplicate=args.fcd_deduplicate_reference,
        )
        mol_metrics = compute_mol_metrics(
            generated_graphs,
            train_smiles,
            bundle.atom_decoder,
            preprocess=args.mol_preprocess,
            requested_total=args.n_samples,
        )
        largest_mol_metrics = mol_metrics if largest_component_matches_raw else compute_mol_metrics(
            largest_component_graphs,
            train_smiles,
            bundle.atom_decoder,
            preprocess=args.mol_preprocess,
            requested_total=args.n_samples,
        )
        molecule_metric_views = {}
        largest_molecule_metric_views = {}
        if args.dataset == "QM9_H":
            for view_name, view_preprocess in (("explicit_h", "repo"), ("h_stripped", "catflow")):
                if view_preprocess == args.mol_preprocess:
                    view_train_smiles = train_smiles
                    view_metrics = mol_metrics
                    view_largest_metrics = largest_mol_metrics
                else:
                    view_train_smiles = build_training_smiles(
                        bundle.train_graphs,
                        bundle.atom_decoder,
                        cache_path=output_dir / f"train_smiles_{view_preprocess}_dedup_cache.json",
                        preprocess=view_preprocess,
                    )
                    view_metrics = compute_mol_metrics(
                        generated_graphs,
                        view_train_smiles,
                        bundle.atom_decoder,
                        preprocess=view_preprocess,
                        requested_total=args.n_samples,
                    )
                    view_largest_metrics = view_metrics if largest_component_matches_raw else compute_mol_metrics(
                        largest_component_graphs,
                        view_train_smiles,
                        bundle.atom_decoder,
                        preprocess=view_preprocess,
                        requested_total=args.n_samples,
                    )
                molecule_metric_views[view_name] = view_metrics
                largest_molecule_metric_views[view_name] = view_largest_metrics
        for key, value in mol_metrics.items():
            if not isinstance(value, list):
                metrics[f"mol/{key}"] = value
        for key, value in largest_mol_metrics.items():
            if not isinstance(value, list):
                metrics[f"largest_component/mol/{key}"] = value
        for view_name, view_metrics in molecule_metric_views.items():
            for key, value in view_metrics.items():
                if not isinstance(value, list):
                    metrics[f"mol/{view_name}/{key}"] = value
        for view_name, view_metrics in largest_molecule_metric_views.items():
            for key, value in view_metrics.items():
                if not isinstance(value, list):
                    metrics[f"largest_component/mol/{view_name}/{key}"] = value
        metrics["mol/preprocess"] = args.mol_preprocess
        if args.dataset == "QM9_H":
            metrics["mol/primary_uniqueness_definition"] = (
                "explicit_h_canonical_smiles"
                if args.mol_preprocess == "repo"
                else "h_stripped_catflow_canonical_smiles"
            )
            metrics["mol/benchmark_policy"] = "project_specific_explicit_h"
            metrics["mol/catflow_table_comparable"] = False
            metrics["mol/comparison_note"] = (
                "Compare only with models retrained using explicit hydrogens; "
                "mol/explicit_h/uniqueness is the representation-preserving score and "
                "mol/h_stripped/uniqueness is secondary."
            )
            metrics["mol/hydrogen_removal_uniqueness_loss"] = (
                metrics["mol/explicit_h/uniqueness"] - metrics["mol/h_stripped/uniqueness"]
            )
            metrics["largest_component/mol/hydrogen_removal_uniqueness_loss"] = (
                metrics["largest_component/mol/explicit_h/uniqueness"]
                - metrics["largest_component/mol/h_stripped/uniqueness"]
            )
        elif args.dataset == "QM9":
            metrics["mol/primary_uniqueness_definition"] = "heavy_atom_catflow_canonical_smiles"
            metrics["mol/benchmark_policy"] = "catflow_comparable_heavy_atom_qm9"
            catflow_requirements = {
                "preprocess_catflow": args.mol_preprocess == "catflow",
                "gdss_split": bundle.provenance.get("qm9_protocol") == "gdss_catflow",
                "reference_split_test": args.reference_split == "test",
                "ten_thousand_attempts": int(args.n_samples) == 10000,
                "four_heavy_atom_types": list(bundle.atom_decoder) == ["C", "N", "O", "F"],
                "maximum_nine_nodes": int(bundle.provenance.get("node_count_max") or 0) <= 9,
            }
            metrics["mol/catflow_comparability_requirements"] = catflow_requirements
            metrics["mol/catflow_comparability_failures"] = [
                name for name, satisfied in catflow_requirements.items() if not satisfied
            ]
            metrics["mol/catflow_table_comparable"] = all(catflow_requirements.values())
        elif args.dataset == "ZINC250K":
            metrics["mol/primary_uniqueness_definition"] = (
                "heavy_atom_catflow_canonical_smiles"
            )
            metrics["mol/benchmark_policy"] = (
                "catflow_comparable_heavy_atom_zinc250k"
            )
            catflow_requirements = {
                "preprocess_catflow": args.mol_preprocess == "catflow",
                "gdss_split": (
                    bundle.provenance.get("zinc250k_protocol")
                    == "gdss_catflow"
                ),
                "reference_split_test": args.reference_split == "test",
                "ten_thousand_attempts": int(args.n_samples) == 10000,
                "nine_heavy_atom_types": list(bundle.atom_decoder)
                == ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"],
                "maximum_38_nodes": int(
                    bundle.provenance.get("node_count_max") or 0
                )
                <= 38,
            }
            metrics["mol/catflow_comparability_requirements"] = (
                catflow_requirements
            )
            metrics["mol/catflow_comparability_failures"] = [
                name
                for name, satisfied in catflow_requirements.items()
                if not satisfied
            ]
            metrics["mol/catflow_table_comparable"] = all(
                catflow_requirements.values()
            )
        metrics["mol/fcd_preprocess"] = args.mol_preprocess
        metrics["mol/fcd_reference_split"] = fcd_reference_split
        metrics["mol/fcd_reference_graphs"] = len(fcd_reference_graphs)
        metrics["mol/fcd_reference_smiles"] = len(reference_smiles)
        fcd_generated_smiles = (
            list(dict.fromkeys(mol_metrics["valid_smiles"]))
            if args.fcd_deduplicate_generated
            else mol_metrics["valid_smiles"]
        )
        largest_fcd_generated_smiles = (
            list(dict.fromkeys(largest_mol_metrics["valid_smiles"]))
            if args.fcd_deduplicate_generated
            else largest_mol_metrics["valid_smiles"]
        )
        metrics["mol/fcd_generated_smiles"] = len(fcd_generated_smiles)
        metrics["mol/fcd_attempts"] = int(args.n_samples)
        metrics["mol/fcd_invalid_filtering"] = "valid_outputs_from_generated_attempts"
        metrics["mol/fcd_generation_protocol"] = "all_valid_outputs_among_requested_attempts"
        metrics["mol/fcd_duplicate_handling"] = (
            "generated_valid_smiles_deduplicated"
            if args.fcd_deduplicate_generated
            else "generated_duplicates_retained"
        )
        metrics["mol/fcd_reference_duplicate_handling"] = ref_dup_label
        fcd_command = args.fcd_command or os.environ.get("FCD_COMMAND")
        metrics["mol/fcd_available"] = False
        metrics["mol/fcd_status"] = "missing_fcd_command"
        if fcd_command:
            fcd_value, fcd_info = compute_fcd_with_command(
                reference_smiles,
                fcd_generated_smiles,
                fcd_command,
                work_dir=output_dir,
                timeout=args.fcd_timeout,
            )
            for key, value in fcd_info.items():
                metrics[f"mol/fcd_{key}"] = value
            if fcd_value is not None and fcd_info.get("status") == "ok":
                metrics["mol/fcd"] = fcd_value
                metrics["mol/fcd_available"] = True
                metrics["mol/fcd_status"] = "ok"
            else:
                metrics["mol/fcd_status"] = fcd_info.get("status", "failed")
        if args.dataset in {"QM9", "ZINC250K"}:
            catflow_requirements = dict(
                metrics.get("mol/catflow_comparability_requirements") or {}
            )
            catflow_requirements["fcd_available"] = bool(
                metrics.get("mol/fcd_available")
            )
            catflow_requirements["generated_fcd_smiles_deduplicated"] = bool(
                args.fcd_deduplicate_generated
            )
            metrics["mol/catflow_comparability_requirements"] = (
                catflow_requirements
            )
            metrics["mol/catflow_comparability_failures"] = [
                name
                for name, satisfied in catflow_requirements.items()
                if not satisfied
            ]
            metrics["mol/catflow_table_comparable"] = all(
                catflow_requirements.values()
            )
        metrics["largest_component/mol/preprocess"] = args.mol_preprocess
        metrics["largest_component/mol/fcd_preprocess"] = args.mol_preprocess
        metrics["largest_component/mol/fcd_reference_split"] = fcd_reference_split
        metrics["largest_component/mol/fcd_reference_graphs"] = len(fcd_reference_graphs)
        metrics["largest_component/mol/fcd_reference_smiles"] = len(reference_smiles)
        metrics["largest_component/mol/fcd_generated_smiles"] = len(
            largest_fcd_generated_smiles
        )
        metrics["largest_component/mol/fcd_attempts"] = int(args.n_samples)
        metrics["largest_component/mol/fcd_invalid_filtering"] = "valid_outputs_from_requested_raw_attempts_after_largest_component_cleanup"
        metrics["largest_component/mol/fcd_generation_protocol"] = "diagnostic_largest_component_view"
        metrics["largest_component/mol/fcd_duplicate_handling"] = (
            "generated_valid_smiles_deduplicated"
            if args.fcd_deduplicate_generated
            else "generated_duplicates_retained"
        )
        metrics["largest_component/mol/fcd_reference_duplicate_handling"] = ref_dup_label
        metrics["largest_component/mol/fcd_available"] = False
        metrics["largest_component/mol/fcd_status"] = "missing_fcd_command"
        if fcd_command and largest_component_matches_raw:
            for suffix in (
                "available",
                "status",
                "returncode",
                "command",
                "reference_smiles",
                "generated_smiles",
                "backend",
                "stdout_tail",
                "stderr_tail",
                "error",
            ):
                raw_key = f"mol/fcd_{suffix}"
                if raw_key in metrics:
                    metrics[f"largest_component/mol/fcd_{suffix}"] = metrics[raw_key]
            if "mol/fcd" in metrics:
                metrics["largest_component/mol/fcd"] = metrics["mol/fcd"]
        elif fcd_command:
            largest_fcd_value, largest_fcd_info = compute_fcd_with_command(
                reference_smiles,
                largest_fcd_generated_smiles,
                fcd_command,
                work_dir=output_dir / "largest_component_fcd",
                timeout=args.fcd_timeout,
            )
            for key, value in largest_fcd_info.items():
                metrics[f"largest_component/mol/fcd_{key}"] = value
            if largest_fcd_value is not None and largest_fcd_info.get("status") == "ok":
                metrics["largest_component/mol/fcd"] = largest_fcd_value
                metrics["largest_component/mol/fcd_available"] = True
                metrics["largest_component/mol/fcd_status"] = "ok"
            else:
                metrics["largest_component/mol/fcd_status"] = largest_fcd_info.get("status", "failed")

        for key, value in list(metrics.items()):
            if key.startswith("mol/"):
                metrics[f"raw/{key}"] = value
        (output_dir / "reference_smiles.txt").write_text(
            "\n".join(reference_smiles) + ("\n" if reference_smiles else "")
        )
        (output_dir / "fcd_reference_smiles.txt").write_text(
            "\n".join(reference_smiles) + ("\n" if reference_smiles else "")
        )
        (output_dir / "valid_smiles.txt").write_text(
            "\n".join(mol_metrics["valid_smiles"]) + ("\n" if mol_metrics["valid_smiles"] else "")
        )
        (output_dir / "unique_smiles.txt").write_text(
            "\n".join(mol_metrics["unique_smiles"]) + ("\n" if mol_metrics["unique_smiles"] else "")
        )
        (output_dir / "novel_smiles.txt").write_text(
            "\n".join(mol_metrics["novel_smiles"]) + ("\n" if mol_metrics["novel_smiles"] else "")
        )
        for view_name, view_metrics in molecule_metric_views.items():
            for collection_name in ("valid_smiles", "unique_smiles", "novel_smiles"):
                values = view_metrics[collection_name]
                (output_dir / f"{view_name}_{collection_name}.txt").write_text(
                    "\n".join(values) + ("\n" if values else "")
                )
        (output_dir / "largest_component_valid_smiles.txt").write_text(
            "\n".join(largest_mol_metrics["valid_smiles"])
            + ("\n" if largest_mol_metrics["valid_smiles"] else "")
        )
        (output_dir / "largest_component_unique_smiles.txt").write_text(
            "\n".join(largest_mol_metrics["unique_smiles"])
            + ("\n" if largest_mol_metrics["unique_smiles"] else "")
        )
        (output_dir / "largest_component_novel_smiles.txt").write_text(
            "\n".join(largest_mol_metrics["novel_smiles"])
            + ("\n" if largest_mol_metrics["novel_smiles"] else "")
        )

    torch.save(
        {
            "samples": samples,
            "generated_graphs": generated_graphs,
            "raw_generated_graphs": generated_graphs,
            "largest_component_graphs": largest_component_graphs,
            "largest_component_source_attempt_indices": largest_component_source_attempt_indices,
            "metrics": metrics,
            "metadata": metadata,
            "final_stage": final_stage,
        },
        output_dir / "ring_vae_samples.pt",
    )
    (output_dir / "metrics.json").write_text(json.dumps(json_ready(metrics), indent=2, sort_keys=True))
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))

    print("\n--- Ring WL-VAE Summary ---")
    print(f"Requested samples: {args.n_samples}")
    print(f"Generated graphs:  {len(generated_graphs)} ({len(generated_graphs) / max(args.n_samples, 1):.2%})")
    print(
        f"Raw components:    {metrics.get('graph/raw_mean_component_count', 0.0):.2f} mean; "
        f"{metrics.get('graph/raw_multiple_components_fraction', 0.0):.2%} disconnected"
    )
    print(
        f"Raw isolates:      {metrics.get('graph/raw_mean_isolated_nodes', 0.0):.2f} mean; "
        f"largest component {metrics.get('graph/raw_mean_largest_component_fraction', 0.0):.2%} of nodes"
    )
    print(f"Complete graphs:   {complete} ({complete / max(args.n_samples, 1):.2%} of attempts)")
    print(f"Ring complete:     {complete_before_closure} ({complete_before_closure / max(args.n_samples, 1):.2%} of attempts)")
    if metrics["sampling/mean_plan_prefix_match_fraction"] is None:
        print("Plan match:        n/a (no external/latent plan)")
    else:
        print(f"Plan match:        {metrics['sampling/mean_plan_prefix_match_fraction']:.2%}")
    print(f"Mean closure edges:{metrics['sampling/mean_closure_edges']:.2f}")
    print(f"Frontier closures: {metrics['sampling/mean_frontier_closure_edges']:.2f}")
    print(f"Final closures:    {metrics['sampling/mean_final_closure_edges']:.2f}")
    print(f"Mean |N-target|:   {metrics['sampling/mean_abs_target_node_error']:.2f}")
    if metrics.get("config/use_action_head"):
        print(f"Learned actions:   {metrics.get('config/use_learned_actions')}")
    print(f"Node-count TV:     {_fmt_metric(metrics.get('dist/node_count/tv'))}")
    print(f"Edge-count TV:     {_fmt_metric(metrics.get('dist/edge_count/tv'))}")
    print(f"Degree TV:         {_fmt_metric(metrics.get('dist/degree/tv'))}")
    print(f"WL-color TV:       {_fmt_metric(metrics.get('dist/wl_color/tv'))}")
    if not largest_component_matches_raw:
        print(
            f"Largest-comp TV:   node {_fmt_metric(metrics.get('largest_component/dist/node_count/tv'))}, "
            f"edge {_fmt_metric(metrics.get('largest_component/dist/edge_count/tv'))}"
        )
    if "mol/validity" in metrics:
        print(f"Mol validity:      {metrics['mol/validity']:.2%}")
        print(f"Largest-comp valid:{metrics['largest_component/mol/validity']:.2%}")
        print(f"Mol uniqueness:    {metrics['mol/uniqueness']:.2%}")
        print(f"Mol novelty:       {metrics['mol/novelty']:.2%}")
        if args.dataset == "QM9_H":
            print(f"Explicit-H unique: {metrics['mol/explicit_h/uniqueness']:.2%}")
            print(f"H-stripped unique: {metrics['mol/h_stripped/uniqueness']:.2%}")
        if metrics.get("mol/fcd_available"):
            print(f"Mol FCD:           {metrics['mol/fcd']:.6f}")
        elif args.fcd_command or os.environ.get("FCD_COMMAND"):
            print(f"Mol FCD:           unavailable ({metrics.get('mol/fcd_status')})")


if __name__ == "__main__":
    main()
