"""Train the Ring WL-VAE and write validation-selected checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from gpt_datasets import (
    PAPER_SPLIT_PROTOCOLS,
    QM9_PROTOCOLS,
    SUPPORTED_DATASETS,
    ZINC250K_PROTOCOLS,
    load_graph_dataset_bundle,
    normalize_dataset_name,
)
from ring_traces import (
    build_training_ring_traces,
    infer_ring_model_sizes,
    load_trace_cache,
    save_trace_cache,
    trace_cache_matches,
    trace_cache_provenance,
)
from ring_vae_model import MODEL_SCHEMA_VERSION, RingWLVAE
from ring_vae_trace_packing import make_packed_ring_batch, pack_ring_trace
from run_wl_witness_ar import json_ready
from wl_color_hierarchy import build_color_hierarchy


def make_parser():
    parser = argparse.ArgumentParser(description="Train a ring-structured WL VAE.")
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
    parser.add_argument("--output-dir", default="checkpoints/ring_vae/default")
    parser.add_argument("--trace-cache", default=None)
    parser.add_argument("--rebuild-trace-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-wl-iters", type=int, default=3)
    parser.add_argument(
        "--decoder-color-min-frequency",
        type=int,
        default=1,
        help="Minimum reusable decoder-color frequency before WL-stage backoff is considered.",
    )
    parser.add_argument(
        "--decoder-rare-fraction-threshold",
        type=float,
        default=1.0,
        help="Back off the decoder to WL1 when the requested-stage rare-color fraction exceeds this value.",
    )
    parser.add_argument("--max-train-graphs", type=int, default=None)
    parser.add_argument("--max-val-graphs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--budget-context", action="store_true",
                        help="Condition decoder heads on target size, remaining budget, unresolved stubs, and ring.")
    parser.add_argument(
        "--edge-budget-context",
        action="store_true",
        help=(
            "Extend budget conditioning with target/current/remaining edges and "
            "current/target mean-degree bins. Requires --budget-context."
        ),
    )
    parser.add_argument("--plan-context", action="store_true",
                        help="Condition decoder heads on the remaining WL/action plan during teacher-forced training.")
    parser.add_argument("--latent-plan", action="store_true",
                        help="Train a latent-conditioned structured plan decoder in front of the constrained AR decoder.")
    parser.add_argument("--plan-dropout", type=float, default=0.10,
                        help="Randomly mask plan colors during decoder training for robust plan following.")
    parser.add_argument("--plan-loss-weight", type=float, default=1.0,
                        help="Weight for the latent structured plan reconstruction loss.")
    parser.add_argument("--no-action-head", action="store_true",
                        help="Disable the learned connect/create/stop decoder head.")
    parser.add_argument("--no-completion-head", action="store_true",
                        help="Disable unresolved-stub prediction from partial decoder states.")
    parser.add_argument("--completion-loss-weight", type=float, default=0.1,
                        help="Weight for the action delta-unresolved auxiliary loss.")
    parser.add_argument("--color-balance-weight", type=float, default=0.01,
                        help="Weight for the train-time WL-color distribution calibration loss.")
    parser.add_argument("--action-connect-weight", type=float, default=3.0,
                        help="Class weight for connect_existing action loss.")
    parser.add_argument("--action-create-weight", type=float, default=1.0,
                        help="Class weight for create_node action loss.")
    parser.add_argument("--action-stop-weight", type=float, default=5.0,
                        help="Class weight for stop action loss.")
    parser.add_argument("--kl-beta", type=float, default=0.1)
    parser.add_argument("--kl-warmup-epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Stop after this many epochs without a meaningful validation improvement; 0 disables.")
    parser.add_argument("--early-stopping-min-epochs", type=int, default=None,
                        help="Minimum epochs before stopping. Defaults to max(15, KL warmup epochs).")
    parser.add_argument("--early-stopping-min-delta", type=float, default=1.0e-3,
                        help="Relative fixed-beta validation-score improvement that resets patience.")
    parser.add_argument("--validation-seed", type=int, default=None,
                        help="Seed for repeatable validation latent draws. Defaults to training seed + 100003.")
    parser.add_argument(
        "--structural-validation-interval",
        type=int,
        default=0,
        help="Generate a validation-only structural audit every N epochs; 0 keeps ELBO-only selection.",
    )
    parser.add_argument(
        "--structural-validation-samples",
        type=int,
        default=8,
        help="Number of validation-only samples used by structural checkpoint selection.",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    return parser


def resolve_device(name):
    if name and str(name).lower() != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(*args):
    print(*args, flush=True)


def make_batches(traces, *, batch_size: int, rng: random.Random, training: bool):
    order = list(range(len(traces)))
    if training:
        rng.shuffle(order)
    batch_size = max(1, int(batch_size))
    return [[traces[idx] for idx in order[start:start + batch_size]] for start in range(0, len(order), batch_size)]


def pack_traces(traces):
    return [pack_ring_trace(trace) for trace in traces]


def make_train_color_probs(color_specs, color_vocab_size: int):
    values = [0.0] * (int(color_vocab_size) + 1)
    for spec in color_specs:
        color_id = int(spec.color_id)
        if 0 < color_id < len(values):
            values[color_id] = max(float(spec.weight), 0.0)
    total = sum(values)
    if total <= 0.0 and int(color_vocab_size) > 0:
        for color_id in range(1, int(color_vocab_size) + 1):
            values[color_id] = 1.0
        total = sum(values)
    return [float(value) / max(total, 1.0e-12) for value in values]


def summarize(rows):
    if not rows:
        return {"loss": 0.0, "seconds": 0.0, "graphs": 0}
    graphs = sum(int(row.get("n_graphs", 0)) for row in rows)
    out = {"graphs": graphs, "seconds": sum(float(row.get("seconds", 0.0)) for row in rows)}
    for key in ["loss", "recon", "center", "color", "prev_edge", "intra_edge", "action", "completion", "color_balance", "plan", "kl"]:
        weighted = [float(row[key]) * int(row.get("n_graphs", 0)) for row in rows if key in row]
        if weighted and graphs > 0:
            out[key] = sum(weighted) / graphs
    out["ppl"] = math.exp(min(float(out.get("recon", out.get("loss", 0.0))), 20.0))
    return out


def validation_selection_score(val_metrics, kl_beta: float):
    """Use the final-beta ELBO for comparable checkpoint selection across epochs."""
    recon = float(val_metrics.get("recon", float("inf")))
    kl = float(val_metrics.get("kl", float("inf")))
    score = recon + float(kl_beta) * kl
    if not math.isfinite(score):
        raise RuntimeError(f"Non-finite validation selection score: recon={recon} kl={kl}")
    return score


def legacy_validation_selection_score(val_metrics):
    """Return the warmup-weighted validation loss used by the old selector."""
    score = float(val_metrics.get("loss", float("inf")))
    if not math.isfinite(score):
        raise RuntimeError(f"Non-finite legacy validation selection score: loss={score}")
    return score


def is_meaningful_improvement(score: float, reference: float, relative_min_delta: float):
    if not math.isfinite(reference):
        return True
    threshold = max(abs(float(reference)), 1.0) * max(0.0, float(relative_min_delta))
    return float(score) < float(reference) - threshold


def run_epoch(
    model,
    traces,
    optimizer,
    *,
    beta: float,
    completion_weight: float,
    color_balance_weight: float,
    plan_loss_weight: float,
    action_class_weights,
    batch_size: int,
    grad_clip: float,
    rng: random.Random,
):
    training = optimizer is not None
    model.train(training)
    rows = []
    t0 = time.perf_counter()
    for batch in make_batches(traces, batch_size=batch_size, rng=rng, training=training):
        if training:
            optimizer.zero_grad(set_to_none=True)
        packed_batch = make_packed_ring_batch(batch)
        with torch.set_grad_enabled(training):
            out = model.compute_loss(
                packed_batch,
                beta=beta,
                completion_weight=completion_weight,
                color_balance_weight=color_balance_weight,
                plan_loss_weight=plan_loss_weight,
                action_class_weights=action_class_weights,
            )
            if training:
                out["loss"].backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()
        rows.append({
            "n_graphs": int(out["n_graphs"]),
            "loss": float(out["loss"].detach().item()),
            "recon": float(out["recon"].item()),
            "center": float(out["center"].item()),
            "color": float(out["color"].item()),
            "prev_edge": float(out["prev_edge"].item()),
            "intra_edge": float(out["intra_edge"].item()),
            "action": float(out["action"].item()),
            "completion": float(out["completion"].item()),
            "color_balance": float(out["color_balance"].item()),
            "plan": float(out["plan"].item()),
            "kl": float(out["kl"].item()),
        })
    summary = summarize(rows)
    summary["seconds"] = time.perf_counter() - t0
    return summary


def structural_validation_score(
    model,
    trace_result,
    bundle,
    *,
    dataset: str,
    n_samples: int,
    seed: int,
    train_smiles_cache=None,
):
    """Validation-only structural proxy; never reads the test split or paper denominators."""
    from ring_vae_generator import RingWLVAEGenerator
    from run_wl_witness_ar import generic_mmd_metrics, graph_metrics
    from wl_witness_ar import sample_graph_budgets
    from zinc_utils import build_training_smiles, compute_mol_metrics

    rng = random.Random(int(seed))
    budgets = sample_graph_budgets(trace_result.metadata, max(1, int(n_samples)), rng)
    generator = RingWLVAEGenerator(
        model=model,
        metadata=trace_result.metadata,
        final_stage=trace_result.final_stage,
        color_specs=trace_result.color_specs,
        require_connected=True,
        require_complete=False,
        max_restarts=(
            5
            if str(trace_result.metadata.get("constraint_kind", "generic")) in {"planar", "sbm"}
            else 1
        ),
        constraint_kind=str(trace_result.metadata.get("constraint_kind", "generic")),
        molecule_constraint_profile=trace_result.metadata.get("molecule_constraint_profile"),
        rng=rng,
    )
    was_training = model.training
    model.eval()
    samples = generator.sample_many(
        [int(item["nodes"]) for item in budgets],
        edge_counts=[item.get("edges") for item in budgets],
        constraint_profiles=[dict(item) for item in budgets],
    )
    model.train(was_training)
    generated = [sample.graph for sample in samples if sample.graph is not None]
    complete = sum(
        int(sample.graph is not None and sample.stats.get("complete", False))
        for sample in samples
    ) / max(len(samples), 1)
    unresolved = sum(
        float(sample.stats.get("unresolved_stubs", 0))
        for sample in samples
        if sample.graph is not None
    ) / max(len(generated), 1)
    mean_target_edges = sum(
        float(item.get("edges") or 0)
        for item in budgets
    ) / max(len(budgets), 1)
    unresolved_fraction = min(1.0, unresolved / max(2.0 * mean_target_edges, 1.0))
    references = list(bundle.val_graphs)[: max(1, int(n_samples))]
    if not references:
        references = list(bundle.train_graphs)[: max(1, int(n_samples))]
    mmd = generic_mmd_metrics(
        references,
        generated,
        trace_result.metadata,
        trace_result.final_stage,
        max_graphs=max(1, int(n_samples)),
    ) if generated and references else {}
    mmd_values = [
        max(0.0, float(mmd.get(key, 1.0)))
        for key in ("mmd/degree", "mmd/clustering", "mmd/orbit")
    ]
    mmd_mean = sum(math.log1p(value) for value in mmd_values) / max(len(mmd_values), 1)

    molecule_datasets = {"ZINC", "ZINC250K", "QM9", "QM9_H", "MOSES", "GUACAMOL"}
    if str(dataset).upper() in molecule_datasets and bundle.atom_decoder:
        train_smiles = train_smiles_cache
        if train_smiles is None:
            train_smiles = list(bundle.train_smiles or [])
        if not train_smiles:
            train_smiles = build_training_smiles(bundle.train_graphs, bundle.atom_decoder)
        quality = compute_mol_metrics(
            generated,
            train_smiles,
            bundle.atom_decoder,
            requested_total=len(samples),
        )
        validity = float(quality["validity"])
        uniqueness = float(quality["uniqueness"])
        novelty = float(quality["novelty"])
        validity_metric = "rdkit_validation_proxy"
    else:
        quality = graph_metrics(
            generated,
            train_graphs=bundle.train_graphs,
            dataset=dataset,
            sbm_validity_backend="heuristic",
        )
        validity = float(quality["graph/valid_fraction"])
        uniqueness = float(quality["graph/unique_fraction"])
        novelty = float(quality["graph/novel_fraction"])
        validity_metric = str(quality["graph/validity_metric"])
    generation_fraction = len(generated) / max(len(samples), 1)
    score = (
        (1.0 - generation_fraction)
        + (1.0 - validity)
        + 0.5 * (1.0 - uniqueness)
        + (1.0 - novelty)
        + 0.5 * (1.0 - complete)
        + 0.5 * unresolved_fraction
        + mmd_mean
    )
    return {
        "score": float(score),
        "generation_fraction": float(generation_fraction),
        "complete_fraction": float(complete),
        "mean_unresolved_stubs": float(unresolved),
        "unresolved_stub_fraction": float(unresolved_fraction),
        "valid_fraction": float(validity),
        "unique_fraction": float(uniqueness),
        "novel_fraction": float(novelty),
        "mmd_degree": float(mmd_values[0]),
        "mmd_clustering": float(mmd_values[1]),
        "mmd_orbit_proxy": float(mmd_values[2]),
        "validity_metric": validity_metric,
        "reference_split": "validation",
        "seed": int(seed),
    }


def load_or_build_traces(args, bundle):
    cache_path = Path(args.trace_cache) if args.trace_cache else None
    expected_provenance = trace_cache_provenance(
        dataset=args.dataset,
        data_root=bundle.data_root,
        seed=args.seed,
        n_wl_iters=args.n_wl_iters,
        max_train_graphs=args.max_train_graphs,
        max_val_graphs=args.max_val_graphs,
        train_graphs=bundle.train_graphs,
        val_graphs=bundle.val_graphs,
        smiles_file=args.smiles_file,
        moses_split=args.moses_split,
        dataset_provenance=bundle.provenance,
        decoder_color_min_frequency=args.decoder_color_min_frequency,
        decoder_rare_fraction_threshold=args.decoder_rare_fraction_threshold,
    )
    if cache_path is not None and cache_path.exists() and not args.rebuild_trace_cache:
        log(f"Loading trace cache: {cache_path}")
        cached = load_trace_cache(cache_path)
        ok, reason = trace_cache_matches(cached, expected_provenance)
        if ok:
            return cached
        log(f"Ignoring stale trace cache ({reason}); rebuilding.")

    log("Fitting WL metadata and building ring traces...")
    result = build_training_ring_traces(
        bundle.train_graphs,
        bundle.val_graphs,
        n_wl_iters=args.n_wl_iters,
        max_train_graphs=args.max_train_graphs,
        max_val_graphs=args.max_val_graphs,
        seed=args.seed,
        decoder_color_min_frequency=args.decoder_color_min_frequency,
        decoder_rare_fraction_threshold=args.decoder_rare_fraction_threshold,
        dataset=args.dataset,
    )
    result.provenance = expected_provenance
    if cache_path is not None:
        save_trace_cache(result, cache_path)
    return result


def packed_count_vocab_size(packed_traces):
    max_count = 0
    for trace in packed_traces:
        values = (
            trace.target_nodes,
            trace.target_edges,
            *trace.degrees,
            *trace.color_unresolved_stubs,
            *trace.prev_edge_unresolved_stubs,
            *trace.intra_edge_unresolved_stubs,
            *trace.action_unresolved_stubs,
            *trace.action_next_unresolved_stubs,
            *trace.color_current_edges,
            *trace.prev_edge_current_edges,
            *trace.intra_edge_current_edges,
            *trace.action_current_edges,
        )
        for value in values:
            max_count = max(max_count, int(value))
    return max(128, max_count + 1)


def main():
    args = make_parser().parse_args()
    if args.edge_budget_context and not args.budget_context:
        raise SystemExit("--edge-budget-context requires --budget-context")
    if args.decoder_color_min_frequency < 1:
        raise SystemExit("--decoder-color-min-frequency must be at least 1")
    if not 0.0 <= args.decoder_rare_fraction_threshold <= 1.0:
        raise SystemExit("--decoder-rare-fraction-threshold must be between 0 and 1")
    if args.structural_validation_interval < 0:
        raise SystemExit("--structural-validation-interval must be non-negative")
    if args.structural_validation_samples < 1:
        raise SystemExit("--structural-validation-samples must be at least 1")
    args.dataset = normalize_dataset_name(args.dataset)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Dataset: {args.dataset}")
    log(f"Output: {output_dir}")
    log(f"Device: {device}")

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
    log(
        f"Graphs: train={len(bundle.train_graphs)} val={len(bundle.val_graphs)} "
        f"test={len(bundle.test_graphs)} mode={bundle.mode} provenance={bundle.provenance}"
    )
    trace_result = load_or_build_traces(args, bundle)
    if len(trace_result.train_traces) < 2:
        raise RuntimeError(
            f"Not enough train traces: {len(trace_result.train_traces)} failures={trace_result.failure_counts}"
        )
    if not trace_result.val_traces:
        raise RuntimeError(f"No validation traces were built; failures={trace_result.failure_counts}")
    log(
        f"Ring traces: train={len(trace_result.train_traces)} val={len(trace_result.val_traces)} "
        f"skipped={trace_result.skipped_graphs} failures={trace_result.failure_counts}"
    )
    log("Packing ring traces for faster training batches...")
    pack_t0 = time.perf_counter()
    packed_train_traces = pack_traces(trace_result.train_traces)
    packed_val_traces = pack_traces(trace_result.val_traces)
    log(f"Packed traces in {time.perf_counter() - pack_t0:.1f}s")

    color_hierarchy = build_color_hierarchy(trace_result.metadata, trace_result.final_stage)
    children_by_prev = {
        int(parent): sorted(int(child) for child in children)
        for parent, children in color_hierarchy.children_by_parent[trace_result.final_stage].items()
    }
    seed_color_ids = [int(spec.color_id) for spec in trace_result.color_specs if int(spec.degree) > 0]
    size_args = infer_ring_model_sizes(trace_result.metadata, trace_result.final_stage, trace_result.color_specs)
    required_seq_len = max([2] + [len(trace.sequence) + 1 for trace in trace_result.train_traces + trace_result.val_traces])
    max_seq_len = max(int(args.max_seq_len), int(required_seq_len))
    if max_seq_len != int(args.max_seq_len):
        log(f"Increasing max_seq_len from {args.max_seq_len} to {max_seq_len} to avoid trace truncation.")
    count_vocab_size = packed_count_vocab_size(packed_train_traces + packed_val_traces)
    model_args = {
        **size_args,
        "children_by_prev": children_by_prev,
        "seed_color_ids": seed_color_ids,
        "latent_dim": int(args.latent_dim),
        "hidden_dim": int(args.hidden_dim),
        "n_layers": int(args.n_layers),
        "n_heads": int(args.n_heads),
        "dropout": float(args.dropout),
        "max_seq_len": int(max_seq_len),
        "count_vocab_size": int(count_vocab_size),
        "use_budget_context": bool(args.budget_context),
        "use_edge_budget_context": bool(args.edge_budget_context),
        "use_action_head": not bool(args.no_action_head),
        "use_completion_head": not bool(args.no_completion_head),
        "use_plan_context": bool(args.plan_context or args.latent_plan),
        "use_latent_plan": bool(args.latent_plan),
        "plan_dropout": float(args.plan_dropout),
        "train_color_probs": make_train_color_probs(trace_result.color_specs, size_args["color_vocab_size"]),
    }
    model = RingWLVAE(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_epoch = None
    best_legacy_val = float("inf")
    best_legacy_epoch = None
    patience_reference = float("inf")
    epochs_without_improvement = 0
    early_stopping_patience = max(0, int(args.early_stopping_patience))
    early_stopping_min_epochs = (
        max(0, int(args.early_stopping_min_epochs))
        if args.early_stopping_min_epochs is not None
        else max(15, int(args.kl_warmup_epochs))
    )
    validation_seed = int(args.validation_seed) if args.validation_seed is not None else int(args.seed) + 100003
    structural_selection_enabled = int(args.structural_validation_interval) > 0
    selection_metric = (
        "validation_structural_budget_v2"
        if structural_selection_enabled
        else "fixed_beta_elbo"
    )
    structural_train_smiles = None
    molecule_datasets = {"ZINC", "ZINC250K", "QM9", "QM9_H", "MOSES", "GUACAMOL"}
    if structural_selection_enabled and args.dataset in molecule_datasets and bundle.atom_decoder:
        from zinc_utils import build_training_smiles

        if bundle.train_smiles:
            structural_train_smiles = frozenset(bundle.train_smiles)
        else:
            smiles_cache = output_dir / "structural_train_smiles_repo_cache.json"
            log(f"Preparing one-time molecular novelty cache: {smiles_cache}")
            structural_train_smiles = frozenset(build_training_smiles(
                bundle.train_graphs,
                bundle.atom_decoder,
                cache_path=smiles_cache,
            ))
    stopped_early = False
    epochs_trained = 0
    history = []
    action_class_weights = (
        float(args.action_connect_weight),
        float(args.action_create_weight),
        float(args.action_stop_weight),
    )
    for epoch in range(1, int(args.epochs) + 1):
        warmup = max(1, int(args.kl_warmup_epochs))
        beta = float(args.kl_beta) * min(1.0, epoch / warmup)
        train_metrics = run_epoch(
            model,
            packed_train_traces,
            optimizer,
            beta=beta,
            completion_weight=args.completion_loss_weight,
            color_balance_weight=args.color_balance_weight,
            plan_loss_weight=args.plan_loss_weight,
            action_class_weights=action_class_weights,
            batch_size=args.batch_size,
            grad_clip=args.grad_clip,
            rng=rng,
        )
        cuda_devices = []
        if device.type == "cuda":
            cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(validation_seed)
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    packed_val_traces,
                    None,
                    beta=beta,
                    completion_weight=args.completion_loss_weight,
                    color_balance_weight=args.color_balance_weight,
                    plan_loss_weight=args.plan_loss_weight,
                    action_class_weights=action_class_weights,
                    batch_size=args.batch_size,
                    grad_clip=args.grad_clip,
                    rng=rng,
                )
        val_score = validation_selection_score(val_metrics, args.kl_beta)
        legacy_val_score = legacy_validation_selection_score(val_metrics)
        structural_metrics = None
        structural_audit_due = structural_selection_enabled and (
            epoch == 1 or epoch % int(args.structural_validation_interval) == 0
        )
        if structural_audit_due:
            # The audit uses a fixed, isolated RNG stream so checkpoint
            # selection cannot alter subsequent training batches or latents.
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(validation_seed + 1)
                with torch.no_grad():
                    structural_metrics = structural_validation_score(
                        model,
                        trace_result,
                        bundle,
                        dataset=args.dataset,
                        n_samples=args.structural_validation_samples,
                        seed=validation_seed + 1,
                        train_smiles_cache=structural_train_smiles,
                    )
        selection_score = (
            float(structural_metrics["score"])
            if structural_metrics is not None
            else None
        ) if structural_selection_enabled else val_score
        row = {
            "epoch": epoch,
            "beta": beta,
            "val/selection_score": selection_score,
            "val/fixed_beta_elbo": val_score,
            "val/legacy_selection_score": legacy_val_score,
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"val/{key}": value for key, value in val_metrics.items()},
        }
        if structural_metrics is not None:
            row.update({
                f"val/structural_{key}": value
                for key, value in structural_metrics.items()
            })
        history.append(row)
        epochs_trained = epoch

        is_best = selection_score is not None and selection_score < best_val
        if is_best:
            best_val = float(selection_score)
            best_epoch = epoch
        is_legacy_best = legacy_val_score < best_legacy_val
        if is_legacy_best:
            best_legacy_val = legacy_val_score
            best_legacy_epoch = epoch
        if is_meaningful_improvement(val_score, patience_reference, args.early_stopping_min_delta):
            patience_reference = val_score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        should_stop = (
            early_stopping_patience > 0
            and epoch >= early_stopping_min_epochs
            and epochs_without_improvement >= early_stopping_patience
        )
        log(
            f"| epoch {epoch:3d} beta {beta:.4f} | "
            f"train {train_metrics.get('loss', 0.0):.4f} recon {train_metrics.get('recon', 0.0):.4f} "
            f"action {train_metrics.get('action', 0.0):.4f} comp {train_metrics.get('completion', 0.0):.4f} "
            f"plan {train_metrics.get('plan', 0.0):.4f} colorbal {train_metrics.get('color_balance', 0.0):.4f} "
            f"kl {train_metrics.get('kl', 0.0):.4f} | "
            f"val {val_metrics.get('loss', 0.0):.4f} recon {val_metrics.get('recon', 0.0):.4f} "
            f"action {val_metrics.get('action', 0.0):.4f} comp {val_metrics.get('completion', 0.0):.4f} "
            f"plan {val_metrics.get('plan', 0.0):.4f} colorbal {val_metrics.get('color_balance', 0.0):.4f} "
            f"kl {val_metrics.get('kl', 0.0):.4f} fixed-elbo {val_score:.4f} "
            f"select {selection_score if selection_score is not None else 'not-audited'} "
            f"legacy {legacy_val_score:.4f} "
            f"patience {epochs_without_improvement}/{early_stopping_patience} | {train_metrics['seconds']:.1f}s"
        )

        ckpt = {
            "model_schema_version": MODEL_SCHEMA_VERSION,
            "model_state": model.state_dict(),
            "model_args": model_args,
            "metadata": trace_result.metadata,
            "final_stage": trace_result.final_stage,
            "color_specs": trace_result.color_specs,
            "args": vars(args),
            "optimizer_state": optimizer.state_dict(),
            "provenance": trace_result.provenance,
            "wl_semantics": trace_result.provenance.get("wl_semantics"),
            "selection": {
                "metric": selection_metric,
                "score": selection_score,
                "fixed_kl_beta": float(args.kl_beta),
                "fixed_beta_elbo": val_score,
                "best_score": best_val,
                "best_epoch": best_epoch,
                "structural_audit": structural_metrics,
                "structural_validation_interval": int(args.structural_validation_interval),
                "structural_validation_samples": int(args.structural_validation_samples),
                "legacy_metric": "warmup_weighted_val_loss",
                "legacy_score": legacy_val_score,
                "legacy_best_score": best_legacy_val,
                "legacy_best_epoch": best_legacy_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "patience": early_stopping_patience,
                "min_epochs": early_stopping_min_epochs,
                "relative_min_delta": float(args.early_stopping_min_delta),
                "validation_seed": validation_seed,
                "stopped_early": should_stop,
            },
            "trace_summary": {
                "n_train_traces": len(trace_result.train_traces),
                "n_val_traces": len(trace_result.val_traces),
                "skipped_graphs": trace_result.skipped_graphs,
                "failure_counts": trace_result.failure_counts,
            },
            "history": history,
        }
        torch.save({**ckpt, "checkpoint_role": "last_epoch"}, output_dir / "last_ring_vae.pt")
        if is_best:
            best_role = "structural_validation_best" if structural_selection_enabled else "fixed_beta_best"
            torch.save({**ckpt, "checkpoint_role": best_role}, output_dir / "best_ring_vae.pt")
        if is_legacy_best:
            torch.save(
                {**ckpt, "checkpoint_role": "legacy_warmup_weighted_val_loss_best"},
                output_dir / "best_legacy_ring_vae.pt",
            )
        if should_stop:
            stopped_early = True
            log(
                f"Early stopping at epoch {epoch}: no meaningful fixed-beta validation improvement "
                f"for {epochs_without_improvement} epochs."
            )
            break

    summary = {
        "config/model": "ring_wl_vae",
        "config/model_schema_version": MODEL_SCHEMA_VERSION,
        "config/use_budget_context": bool(args.budget_context),
        "config/use_edge_budget_context": bool(args.edge_budget_context),
        "config/constraint_kind": trace_result.metadata.get("constraint_kind", "generic"),
        "config/decoder_color_backoff": trace_result.metadata.get("decoder_color_backoff"),
        "config/requested_n_wl_iters": trace_result.metadata.get("requested_n_wl_iters", trace_result.final_stage),
        "config/decoder_wl_stage": trace_result.final_stage,
        "config/use_action_head": not bool(args.no_action_head),
        "config/use_completion_head": not bool(args.no_completion_head),
        "config/use_plan_context": bool(args.plan_context or args.latent_plan),
        "config/use_latent_plan": bool(args.latent_plan),
        "config/plan_dropout": float(args.plan_dropout),
        "config/plan_loss_weight": float(args.plan_loss_weight),
        "config/completion_loss_weight": float(args.completion_loss_weight),
        "config/color_balance_weight": float(args.color_balance_weight),
        "config/action_connect_weight": float(args.action_connect_weight),
        "config/action_create_weight": float(args.action_create_weight),
        "config/action_stop_weight": float(args.action_stop_weight),
        "config/early_stopping_patience": early_stopping_patience,
        "config/early_stopping_min_epochs": early_stopping_min_epochs,
        "config/early_stopping_relative_min_delta": float(args.early_stopping_min_delta),
        "config/validation_seed": validation_seed,
        "config/structural_validation_interval": int(args.structural_validation_interval),
        "config/structural_validation_samples": int(args.structural_validation_samples),
        "config/dataset": args.dataset,
        "best_val_loss": best_val,
        "best_validation_metric": selection_metric,
        "best_validation_score": best_val,
        "best_epoch": best_epoch,
        "best_legacy_validation_metric": "warmup_weighted_val_loss",
        "best_legacy_validation_score": best_legacy_val,
        "best_legacy_epoch": best_legacy_epoch,
        "epochs_trained": epochs_trained,
        "stopped_early": stopped_early,
        "model_args": model_args,
        "final_stage": trace_result.final_stage,
        "provenance": trace_result.provenance,
        "wl_semantics": trace_result.provenance.get("wl_semantics"),
        "n_train_traces": len(trace_result.train_traces),
        "n_val_traces": len(trace_result.val_traces),
        "skipped_graphs": trace_result.skipped_graphs,
        "failure_counts": trace_result.failure_counts,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(json_ready(summary), indent=2, sort_keys=True))
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    log(
        f"Training complete after {epochs_trained} epochs. "
        f"Best {selection_metric} score: {best_val:.4f} at epoch {best_epoch}."
    )
    log(f"Best checkpoint: {output_dir / 'best_ring_vae.pt'}")
    log(
        f"Best legacy validation loss: {best_legacy_val:.4f} at epoch {best_legacy_epoch}. "
        f"Checkpoint: {output_dir / 'best_legacy_ring_vae.pt'}"
    )
    log(f"Last checkpoint: {output_dir / 'last_ring_vae.pt'}")


if __name__ == "__main__":
    main()
