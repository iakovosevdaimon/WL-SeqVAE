"""Validate checkpoint schema, feature configuration, and dataset provenance."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import json

import torch

from gpt_datasets import (
    PAPER_SPLIT_PROTOCOLS,
    QM9_PROTOCOLS,
    ZINC250K_PROTOCOLS,
    load_graph_dataset_bundle,
    normalize_dataset_name,
)
from ring_traces import WL_SEMANTICS_VERSION, graph_collection_fingerprint, validate_final_stage
from ring_vae_model import MODEL_SCHEMA_VERSION


FEATURE_KEYS = {
    "budget_context": "use_budget_context",
    "edge_budget_context": "use_edge_budget_context",
    "action_head": "use_action_head",
    "completion_head": "use_completion_head",
    "plan_context": "use_plan_context",
    "latent_plan": "use_latent_plan",
}


def checkpoint_compatibility_errors(
    checkpoint: dict[str, Any],
    *,
    dataset: str | None = None,
    n_wl_iters: int | None = None,
    expected_features: dict[str, bool | None] | None = None,
    expected_data_provenance: dict[str, Any] | None = None,
    expected_train_fingerprint: str | None = None,
    expected_val_fingerprint: str | None = None,
    expected_training_args: dict[str, Any] | None = None,
):
    errors = []
    schema_version = checkpoint.get("model_schema_version")
    if schema_version != MODEL_SCHEMA_VERSION:
        errors.append(
            f"model schema {schema_version!r} != current schema {MODEL_SCHEMA_VERSION}; retraining is required"
        )

    checkpoint_args = checkpoint.get("args") or {}
    checkpoint_dataset = checkpoint_args.get("dataset")
    if dataset is not None and checkpoint_dataset is not None:
        if normalize_dataset_name(checkpoint_dataset) != normalize_dataset_name(dataset):
            errors.append(f"checkpoint dataset {checkpoint_dataset!r} != requested {dataset!r}")
    for key, expected in (expected_training_args or {}).items():
        actual = checkpoint_args.get(key)
        if actual != expected:
            errors.append(f"checkpoint training arg {key}={actual!r} != expected {expected!r}")

    metadata = checkpoint.get("metadata") or {}
    try:
        validate_final_stage(metadata, checkpoint["final_stage"])
    except Exception as exc:
        errors.append(str(exc))

    actual_n_wl_iters = metadata.get("n_wl_iters")
    if n_wl_iters is not None and actual_n_wl_iters != int(n_wl_iters):
        errors.append(f"checkpoint n_wl_iters {actual_n_wl_iters!r} != requested {int(n_wl_iters)}")

    wl_semantics = checkpoint.get("wl_semantics") or (checkpoint.get("provenance") or {}).get("wl_semantics")
    if wl_semantics != WL_SEMANTICS_VERSION:
        if wl_semantics is None:
            errors.append("checkpoint does not record WL semantics/provenance")
        else:
            errors.append(f"checkpoint WL semantics {wl_semantics!r} != {WL_SEMANTICS_VERSION!r}")

    provenance = checkpoint.get("provenance") or {}
    checkpoint_data_provenance = provenance.get("dataset_provenance") or {}
    for key, expected in (expected_data_provenance or {}).items():
        actual = checkpoint_data_provenance.get(key)
        if actual != expected:
            errors.append(f"checkpoint dataset provenance {key}={actual!r} != expected {expected!r}")
    if expected_train_fingerprint is not None:
        actual = provenance.get("train_fingerprint")
        if actual != expected_train_fingerprint:
            errors.append("checkpoint training-graph fingerprint does not match the requested split")
    if expected_val_fingerprint is not None:
        actual = provenance.get("val_fingerprint")
        if actual != expected_val_fingerprint:
            errors.append("checkpoint validation-graph fingerprint does not match the requested split")

    model_args = checkpoint.get("model_args") or {}
    for feature, expected in (expected_features or {}).items():
        if expected is None:
            continue
        model_key = FEATURE_KEYS[feature]
        actual = bool(model_args.get(model_key, False))
        if actual != bool(expected):
            errors.append(f"checkpoint {model_key}={actual} != expected {bool(expected)}")
    return errors


def make_parser():
    parser = argparse.ArgumentParser(description="Check whether a ring-VAE checkpoint is safe to reuse.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--n-wl-iters", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--zinc-subset", default=None, choices=("small", "large"))
    parser.add_argument("--max-train-graphs", type=int, default=None)
    parser.add_argument("--max-val-graphs", type=int, default=None)
    parser.add_argument("--smiles-file", default=None)
    parser.add_argument("--moses-split", default="train", choices=("train", "test", "test_scaffolds"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--paper-split-protocol", choices=PAPER_SPLIT_PROTOCOLS, default="legacy_local")
    parser.add_argument("--qm9-protocol", choices=QM9_PROTOCOLS, default="legacy_local")
    parser.add_argument("--zinc250k-protocol", choices=ZINC250K_PROTOCOLS, default="legacy_pyg")
    parser.add_argument("--verify-data-provenance", action="store_true")
    parser.add_argument(
        "--expected-training-arg",
        action="append",
        default=[],
        metavar="KEY=JSON_VALUE",
        help="Require a checkpoint training argument to match exactly; repeat as needed.",
    )
    for option in FEATURE_KEYS:
        parser.add_argument(f"--{option.replace('_', '-')}", choices=("any", "on", "off"), default="any")
    return parser


def main():
    args = make_parser().parse_args()
    path = Path(args.checkpoint)
    if not path.is_file():
        print(f"Checkpoint is missing: {path}")
        raise SystemExit(1)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"Checkpoint could not be loaded: {exc}")
        raise SystemExit(1)

    expected_features = {}
    for feature in FEATURE_KEYS:
        value = getattr(args, feature)
        expected_features[feature] = None if value == "any" else value == "on"
    expected_training_args = {}
    for item in args.expected_training_arg:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise SystemExit(f"Invalid --expected-training-arg {item!r}; expected KEY=JSON_VALUE")
        try:
            expected_training_args[key] = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON value in --expected-training-arg {item!r}: {exc}") from exc

    # Reject an obviously stale/incompatible checkpoint before touching the
    # dataset.  In particular, provenance verification can require a large
    # download or preprocessing pass that is pointless when the schema or
    # requested model features already disagree.
    basic_errors = checkpoint_compatibility_errors(
        checkpoint,
        dataset=args.dataset,
        n_wl_iters=args.n_wl_iters,
        expected_features=expected_features,
        expected_training_args=expected_training_args,
    )
    if basic_errors:
        print(f"Checkpoint is incompatible: {path}")
        for error in basic_errors:
            print(f"- {error}")
        raise SystemExit(1)

    expected_data_provenance = None
    expected_train_fingerprint = None
    expected_val_fingerprint = None
    if args.verify_data_provenance:
        if args.dataset is None:
            raise SystemExit("--verify-data-provenance requires --dataset")
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
        expected_data_provenance = bundle.provenance
        expected_train_fingerprint = graph_collection_fingerprint(bundle.train_graphs)
        expected_val_fingerprint = graph_collection_fingerprint(bundle.val_graphs)

    errors = checkpoint_compatibility_errors(
        checkpoint,
        dataset=args.dataset,
        n_wl_iters=args.n_wl_iters,
        expected_features=expected_features,
        expected_data_provenance=expected_data_provenance,
        expected_train_fingerprint=expected_train_fingerprint,
        expected_val_fingerprint=expected_val_fingerprint,
        expected_training_args=expected_training_args,
    )
    if errors:
        print(f"Checkpoint is incompatible: {path}")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print(f"Checkpoint is compatible: {path}")


if __name__ == "__main__":
    main()
