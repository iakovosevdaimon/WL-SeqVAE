#!/usr/bin/env python3
"""Recompute current graph benchmarks with the DiGress/SPECTRE MMD protocol.

The implementation mirrors the public DiGress ``spectre_utils.py`` and
``dist_helper.py`` behavior for degree, clustering, and ORCA orbit MMDs:

* Community-small: Gaussian EMD kernels.
* Other abstract graphs: Gaussian total-variation kernels.
* Orbit features: exact ORCA node orbit counts for graphlets up to size four.
* MMD: the biased squared estimate, including diagonal kernel terms.

For datasets covered by the SPECTRE tables, ratios use the table's published
train-vs-test denominators, matching the convention stated by DiGress.  Other
datasets use locally recomputed train-vs-test denominators and are labeled as
protocol-compatible rather than directly paper-comparable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import networkx as nx
import numpy as np
import torch
from scipy.spatial.distance import cdist

try:
    import pyemd
except ImportError:  # Checked explicitly when Community-small is evaluated.
    pyemd = None

from gpt_datasets import load_graph_dataset_bundle
from project_paths import PROJECT_ROOT, portable_path
from run_wl_witness_ar import _graph_data_to_nx


DEFAULT_RUN_ROOT = PROJECT_ROOT / "benchmark_runs" / "corrected_graphs_wl1_wl2_wl3_seed0"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_ROOT / "digress_spectre_compatible"
DEFAULT_MARKDOWN = PROJECT_ROOT / "current_corrected_digress_spectre_metrics.md"

OFFICIAL_DIGRESS_PAPER = "https://arxiv.org/abs/2209.14734"
OFFICIAL_SPECTRE_PAPER = "https://arxiv.org/abs/2204.01613"
OFFICIAL_EVALUATOR = "https://github.com/cvignac/DiGress/blob/main/src/analysis/spectre_utils.py"
OFFICIAL_DISTANCE_HELPER = "https://github.com/cvignac/DiGress/blob/main/src/analysis/dist_helper.py"

# Values printed in SPECTRE Tables 1, 2, and 4.  DiGress Appendix F.1 states
# that these printed values are used as MMD^2 denominators.
PUBLISHED_DENOMINATORS: dict[str, dict[str, float]] = {
    "COMMUNITY_SMALL": {"degree": 0.02, "clustering": 0.07, "orbit": 0.01},
    "PLANAR": {"degree": 0.0002, "clustering": 0.0310, "orbit": 0.0005},
    "SBM": {"degree": 0.0008, "clustering": 0.0332, "orbit": 0.0255},
    "PROTEINS": {"degree": 0.0003, "clustering": 0.0068, "orbit": 0.0032},
}

CHECKPOINT_VARIANTS = {
    "budget_allow_incomplete": "fixed_beta_best",
    "budget_legacy_val_loss_best_allow_incomplete": "legacy_val_loss_best",
    "budget_last_epoch_allow_incomplete": "last_epoch",
}

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
EXPECTED_PAPER_SAMPLES = {
    "COMMUNITY_SMALL": 20,
    "EGO_SMALL": 40,
    "ENZYMES": 117,
    "GRID": 20,
    "PLANAR": 40,
    "SBM": 40,
    "PROTEINS": 184,
}


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    wl: int
    group: str
    checkpoint_role: str
    sample_dir: Path
    sample_file: Path
    metric_file: Path
    strict_metric_file: Path
    run_config_file: Path


@dataclass
class CompatibleResult:
    dataset: str
    wl: int
    group: str
    checkpoint_role: str
    checkpoint_epoch: int | None
    checkpoint_selection_metric: str | None
    checkpoint_selection_score: float | None
    samples: int
    target_samples: int
    strict_generated: int
    strict_generation_pct: float
    complete_pct: float
    vun_pct: float
    validity_metric: str | None
    sbm_accuracy_refinement_steps: int | None
    sbm_vun_refinement_steps: int | None
    degree_mmd2: float
    clustering_mmd2: float
    orbit_mmd2: float
    degree_denominator: float
    clustering_denominator: float
    orbit_denominator: float
    degree_local_denominator: float
    clustering_local_denominator: float
    orbit_local_denominator: float
    degree_ratio: float | None
    clustering_ratio: float | None
    orbit_ratio: float | None
    mean_ratio: float | None
    ratio_metrics_used: int
    ratio_status: str
    denominator_source: str
    reference_split: str
    comparison_protocol: str | None
    paper_dataset_match: bool
    directly_paper_comparable: bool
    comparability_failures: list[str]
    kernel_family: str
    orbit_backend: str
    sample_file: str
    original_metric_file: str
    compatible_metric_file: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--orca", type=Path, default=PROJECT_ROOT / "third_party" / "orca" / "orca")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--datasets", nargs="*", default=None)
    return parser.parse_args()


def json_number(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value) -> float:
    return 100.0 * float(value or 0.0)


def scan_runs(run_root: Path, datasets: set[str] | None = None) -> list[RunSpec]:
    runs: list[RunSpec] = []
    pattern = "*_allow_incomplete/samples/metrics.json"
    for metric_file in sorted(run_root.rglob(pattern)):
        variant = metric_file.parents[1].name
        fallback_role = CHECKPOINT_VARIANTS.get(variant)
        if fallback_role is None:
            continue
        original = read_json(metric_file)
        embedded_role = str(original.get("config/checkpoint_role") or "")
        role = {
            "legacy_warmup_weighted_val_loss_best": "legacy_val_loss_best",
            "legacy_val_loss_best": "legacy_val_loss_best",
            "fixed_beta_best": "fixed_beta_best",
            "structural_validation_best": "structural_validation_best",
            "last_epoch": "last_epoch",
        }.get(embedded_role, fallback_role)
        dataset = str(original["config/dataset"])
        if datasets and dataset not in datasets:
            continue
        sample_dir = metric_file.parent
        sample_file = sample_dir / "ring_vae_samples.pt"
        run_config_file = sample_dir / "run_config.json"
        if not sample_file.exists() or not run_config_file.exists():
            continue
        strict_variant = variant.removesuffix("_allow_incomplete")
        strict_metric_file = metric_file.parents[2] / strict_variant / "samples" / "metrics.json"
        runs.append(
            RunSpec(
                dataset=dataset,
                wl=int(original.get("config/requested_n_wl_iters", original["config/final_stage"])),
                group=metric_file.parents[3].name,
                checkpoint_role=role,
                sample_dir=sample_dir,
                sample_file=sample_file,
                metric_file=metric_file,
                strict_metric_file=strict_metric_file,
                run_config_file=run_config_file,
            )
        )
    return sorted(
        runs,
        key=lambda run: (
            DATASET_ORDER.index(run.dataset) if run.dataset in DATASET_ORDER else 999,
            run.wl,
            ROLE_ORDER[run.checkpoint_role],
        ),
    )


def simple_graph(graph) -> nx.Graph:
    if isinstance(graph, nx.Graph):
        clean = nx.Graph(graph)
    else:
        clean = nx.Graph(_graph_data_to_nx(graph))
    clean.remove_edges_from(nx.selfloop_edges(clean))
    return clean


def graph_fingerprint(graph: nx.Graph) -> str:
    nodes = list(graph.nodes())
    index = {node: idx for idx, node in enumerate(nodes)}
    edges = sorted((min(index[u], index[v]), max(index[u], index[v])) for u, v in graph.edges())
    payload = f"{len(nodes)}|" + ";".join(f"{u},{v}" for u, v in edges)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


class OrbitCache:
    def __init__(self, path: Path, orca: Path, workers: int):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.orca = orca.resolve()
        self.workers = max(1, int(workers))
        if not self.orca.is_file() or not os.access(self.orca, os.X_OK):
            raise FileNotFoundError(f"ORCA executable is missing or not executable: {self.orca}")
        self.orca_sha256 = hashlib.sha256(self.orca.read_bytes()).hexdigest()
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        existing = self.conn.execute(
            "SELECT value FROM cache_metadata WHERE key='orca_sha256'"
        ).fetchone()
        if existing is not None and existing[0] != self.orca_sha256:
            raise RuntimeError(
                f"Orbit cache {self.path} was built with ORCA {existing[0]}, "
                f"but the requested executable is {self.orca_sha256}. Use a fresh output directory."
            )
        self.conn.execute(
            "INSERT OR REPLACE INTO cache_metadata(key,value) VALUES('orca_sha256',?)",
            (self.orca_sha256,),
        )
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS orbit_features "
            "(fingerprint TEXT PRIMARY KEY, nodes INTEGER NOT NULL, edges INTEGER NOT NULL, feature TEXT NOT NULL)"
        )
        self.conn.commit()

    def close(self):
        self.conn.close()

    def _read(self, fingerprints: Sequence[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        batch_size = 500
        for start in range(0, len(fingerprints), batch_size):
            batch = fingerprints[start:start + batch_size]
            if not batch:
                continue
            marks = ",".join("?" for _ in batch)
            rows = self.conn.execute(
                f"SELECT fingerprint, feature FROM orbit_features WHERE fingerprint IN ({marks})",
                list(batch),
            )
            for fingerprint, feature in rows:
                found[fingerprint] = np.asarray(json.loads(feature), dtype=np.float64)
        return found

    def _compute_one(self, graph: nx.Graph) -> np.ndarray:
        nodes = list(graph.nodes())
        index = {node: idx for idx, node in enumerate(nodes)}
        edges = [(index[u], index[v]) for u, v in graph.edges()]
        input_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".orca", delete=False, encoding="utf-8") as handle:
                input_path = Path(handle.name)
                handle.write(f"{len(nodes)} {len(edges)}\n")
                for u, v in edges:
                    handle.write(f"{u} {v}\n")
            proc = subprocess.run(
                [str(self.orca), "node", "4", str(input_path), "std"],
                check=True,
                capture_output=True,
                text=True,
            )
            output = proc.stdout
            marker = "orbit counts:"
            marker_index = output.find(marker)
            if marker_index < 0:
                raise RuntimeError(f"ORCA output has no orbit marker: {output[-500:]}")
            rows = []
            for line in output[marker_index + len(marker):].strip().splitlines():
                values = line.strip().split()
                if values:
                    rows.append([int(value) for value in values])
            counts = np.asarray(rows, dtype=np.float64)
            if counts.shape != (len(nodes), 15):
                raise RuntimeError(f"Expected ORCA shape {(len(nodes), 15)}, received {counts.shape}")
            return counts.sum(axis=0) / max(len(nodes), 1)
        finally:
            if input_path is not None:
                input_path.unlink(missing_ok=True)

    def features(self, graphs: Sequence[nx.Graph], label: str) -> np.ndarray:
        fingerprints = [graph_fingerprint(graph) for graph in graphs]
        unique_graphs: dict[str, nx.Graph] = {}
        for fingerprint, graph in zip(fingerprints, graphs):
            unique_graphs.setdefault(fingerprint, graph)
        cached = self._read(list(unique_graphs))
        missing = [(key, graph) for key, graph in unique_graphs.items() if key not in cached]
        if missing:
            print(f"ORCA {label}: computing {len(missing)} new graph features; {len(cached)} cached.", flush=True)
            completed: dict[str, np.ndarray] = {}
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = {pool.submit(self._compute_one, graph): (key, graph) for key, graph in missing}
                for idx, future in enumerate(as_completed(futures), start=1):
                    key, graph = futures[future]
                    completed[key] = future.result()
                    if idx % 250 == 0 or idx == len(missing):
                        print(f"ORCA {label}: {idx}/{len(missing)}", flush=True)
            self.conn.executemany(
                "INSERT OR REPLACE INTO orbit_features(fingerprint,nodes,edges,feature) VALUES(?,?,?,?)",
                [
                    (
                        key,
                        int(unique_graphs[key].number_of_nodes()),
                        int(unique_graphs[key].number_of_edges()),
                        json.dumps(feature.tolist(), separators=(",", ":")),
                    )
                    for key, feature in completed.items()
                ],
            )
            self.conn.commit()
            cached.update(completed)
        return np.stack([cached[fingerprint] for fingerprint in fingerprints], axis=0)


def degree_features(graphs: Sequence[nx.Graph]) -> list[np.ndarray]:
    return [np.asarray(nx.degree_histogram(graph), dtype=np.float64) for graph in graphs]


def clustering_features(graphs: Sequence[nx.Graph]) -> list[np.ndarray]:
    features = []
    for graph in graphs:
        values = list(nx.clustering(graph).values())
        hist, _ = np.histogram(values, bins=100, range=(0.0, 1.0), density=False)
        features.append(hist.astype(np.float64))
    return features


def pad_features(samples1: Sequence[np.ndarray], samples2: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    width = max([1] + [len(sample) for sample in samples1] + [len(sample) for sample in samples2])
    first = np.zeros((len(samples1), width), dtype=np.float64)
    second = np.zeros((len(samples2), width), dtype=np.float64)
    for idx, sample in enumerate(samples1):
        first[idx, :len(sample)] = sample
    for idx, sample in enumerate(samples2):
        second[idx, :len(sample)] = sample
    return first, second


def normalize_histograms(values: np.ndarray) -> np.ndarray:
    return values / (values.sum(axis=1, keepdims=True) + 1.0e-6)


def kernel_mean(x: np.ndarray, y: np.ndarray, kernel: str, sigma: float, distance_scaling: float = 1.0) -> float:
    if kernel == "gaussian_tv":
        distances = cdist(x, y, metric="cityblock") / 2.0
    elif kernel == "gaussian_emd":
        if pyemd is None:
            raise RuntimeError(
                "Exact Community-small Gaussian-EMD compatibility requires pyemd. "
                "Install pyemd in the evaluation environment before recomputing metrics."
            )
        support = max(int(x.shape[1]), int(y.shape[1]))
        coordinates = np.arange(support, dtype=np.float64)
        distance_matrix = np.abs(coordinates[:, None] - coordinates[None, :])
        distance_matrix /= float(distance_scaling)
        distances = np.empty((x.shape[0], y.shape[0]), dtype=np.float64)
        for first_idx, first in enumerate(x):
            for second_idx, second in enumerate(y):
                distances[first_idx, second_idx] = abs(
                    float(pyemd.emd(first, second, distance_matrix))
                )
    elif kernel == "gaussian":
        distances = np.sqrt(cdist(x, y, metric="sqeuclidean"))
    else:
        raise ValueError(f"Unknown kernel {kernel}")
    return float(np.exp(-(distances * distances) / (2.0 * sigma * sigma)).mean())


def compute_mmd2(
    samples1: Sequence[np.ndarray] | np.ndarray,
    samples2: Sequence[np.ndarray] | np.ndarray,
    *,
    kernel: str,
    sigma: float,
    is_hist: bool,
    distance_scaling: float = 1.0,
) -> float:
    if len(samples1) == 0 or len(samples2) == 0:
        raise RuntimeError("Paper-compatible MMD is undefined for an empty graph collection")
    first_seq = [np.asarray(sample, dtype=np.float64) for sample in samples1]
    second_seq = [np.asarray(sample, dtype=np.float64) for sample in samples2]
    first, second = pad_features(first_seq, second_seq)
    if is_hist:
        first = normalize_histograms(first)
        second = normalize_histograms(second)
    value = (
        kernel_mean(first, first, kernel, sigma, distance_scaling)
        + kernel_mean(second, second, kernel, sigma, distance_scaling)
        - 2.0 * kernel_mean(first, second, kernel, sigma, distance_scaling)
    )
    return float(value)


def metric_protocol(dataset: str) -> dict[str, dict]:
    if dataset == "COMMUNITY_SMALL":
        return {
            "degree": {"kernel": "gaussian_emd", "sigma": 1.0, "is_hist": True, "distance_scaling": 1.0},
            "clustering": {"kernel": "gaussian_emd", "sigma": 0.1, "is_hist": True, "distance_scaling": 100.0},
            "orbit": {"kernel": "gaussian", "sigma": 30.0, "is_hist": False, "distance_scaling": 1.0},
        }
    return {
        "degree": {"kernel": "gaussian_tv", "sigma": 1.0, "is_hist": True, "distance_scaling": 1.0},
        "clustering": {"kernel": "gaussian_tv", "sigma": 0.1, "is_hist": True, "distance_scaling": 1.0},
        "orbit": {"kernel": "gaussian_tv", "sigma": 30.0, "is_hist": False, "distance_scaling": 1.0},
    }


def compute_metric_set(
    dataset: str,
    reference: dict[str, Sequence[np.ndarray] | np.ndarray],
    generated: dict[str, Sequence[np.ndarray] | np.ndarray],
) -> dict[str, float]:
    protocol = metric_protocol(dataset)
    return {
        name: compute_mmd2(reference[name], generated[name], **protocol[name])
        for name in ("degree", "clustering", "orbit")
    }


DATASET_LOADER_CONFIG_KEYS = (
    "data_root",
    "zinc_subset",
    "max_train_graphs",
    "max_val_graphs",
    "smiles_file",
    "moses_split",
    "seed",
    "paper_split_protocol",
    "qm9_protocol",
    "zinc250k_protocol",
)


def _normalized_dataset_loader_config(config: dict) -> dict:
    values = {key: config.get(key) for key in DATASET_LOADER_CONFIG_KEYS}
    for key in ("data_root", "smiles_file"):
        if values[key] is not None:
            values[key] = str(Path(values[key]).expanduser().resolve())
    values["seed"] = int(values["seed"] or 0)
    values["moses_split"] = values["moses_split"] or "train"
    values["paper_split_protocol"] = values["paper_split_protocol"] or "legacy_local"
    values["qm9_protocol"] = values["qm9_protocol"] or "legacy_local"
    values["zinc250k_protocol"] = values["zinc250k_protocol"] or "legacy_pyg"
    return values


def _require_matching_dataset_loader_config(dataset: str, expected: dict, actual: dict, source: Path):
    expected_comparable = dict(expected)
    actual_comparable = dict(actual)
    # For the official PROTEINS/DD protocol, seed=1234 is embedded in the
    # loader and the run-config seed is only the model-training seed.  A
    # multiseed report must therefore compare the fixed loader protocol and
    # recorded provenance, not require identical model seeds.
    if (
        dataset == "PROTEINS"
        and expected_comparable.get("paper_split_protocol") == "spectre_seed1234"
        and actual_comparable.get("paper_split_protocol") == "spectre_seed1234"
    ):
        expected_comparable.pop("seed", None)
        actual_comparable.pop("seed", None)
    if expected_comparable != actual_comparable:
        raise RuntimeError(
            f"Mixed dataset loader configurations for {dataset}: "
            f"first={expected} current={actual} ({source})"
        )


def _require_matching_recorded_dataset_provenance(
    dataset: str,
    loaded: dict,
    recorded: dict | None,
    source: Path,
):
    # Older artifacts without recorded provenance can still be recomputed, but
    # they remain explicitly non-comparable below.  A recorded mismatch is
    # stronger evidence: using the loaded split would produce metrics against
    # data other than the run's own reference, so fail instead of publishing a
    # misleading recomputation.
    if recorded is not None and recorded != loaded:
        raise RuntimeError(
            f"Recorded dataset provenance does not match the loaded data for {dataset}: "
            f"recorded={recorded} loaded={loaded} ({source})"
        )


def load_dataset_graphs(run: RunSpec, config: dict | None = None) -> tuple[list[nx.Graph], list[nx.Graph], dict, bool]:
    config = dict(config or read_json(run.run_config_file))
    bundle = load_graph_dataset_bundle(
        run.dataset,
        data_root=config.get("data_root"),
        zinc_subset=config.get("zinc_subset"),
        max_train_graphs=config.get("max_train_graphs"),
        max_val_graphs=config.get("max_val_graphs"),
        smiles_file=config.get("smiles_file"),
        moses_split=config.get("moses_split", "train"),
        seed=int(config.get("seed", 0)),
        paper_split_protocol=config.get("paper_split_protocol", "legacy_local"),
        qm9_protocol=config.get("qm9_protocol", "legacy_local"),
        zinc250k_protocol=config.get("zinc250k_protocol", "legacy_pyg"),
    )
    return (
        [simple_graph(graph) for graph in bundle.train_graphs],
        [simple_graph(graph) for graph in bundle.test_graphs],
        dict(bundle.provenance),
        bool(config.get("verify_data_provenance", False)),
    )


def comparison_protocol(dataset: str, provenance: dict) -> str | None:
    protocol = provenance.get("paper_split_protocol")
    source = provenance.get("source")
    if dataset in {"COMMUNITY_SMALL", "PLANAR", "SBM"}:
        if source != "official_spectre_asset":
            return None
        if protocol == "digress_seed0":
            return "digress_current_main_seed0_membership"
        if protocol == "spectre_seed1234":
            return "digress_paper_era_and_spectre_seed1234"
    if dataset == "PROTEINS":
        if source == "official_spectre_dd_filtered_100_500" and protocol == "spectre_seed1234":
            return "spectre_seed1234"
    return None


def exact_paper_dataset_match(dataset: str, provenance: dict) -> bool:
    return comparison_protocol(dataset, provenance) is not None


def paper_comparability_failures(
    *,
    dataset: str,
    denominator_source: str,
    provenance_verified: bool,
    paper_dataset_match: bool,
    reference_split: str,
    target_samples: int,
    generated_samples: int,
    validity_metric: str | None = None,
    sbm_accuracy_refinement_steps: int | None = None,
    sbm_vun_refinement_steps: int | None = None,
) -> list[str]:
    failures = []
    if denominator_source != "spectre_published_table":
        failures.append("no_published_spectre_denominator")
    if not provenance_verified:
        failures.append("checkpoint_or_dataset_provenance_unverified")
    if not paper_dataset_match:
        failures.append("non_official_asset_or_split")
    if reference_split != "test":
        failures.append(f"reference_split_is_{reference_split or 'missing'}")
    expected_samples = EXPECTED_PAPER_SAMPLES.get(dataset)
    if expected_samples is not None and target_samples != expected_samples:
        failures.append(f"target_samples_{target_samples}_expected_{expected_samples}")
    if expected_samples is not None and generated_samples != expected_samples:
        failures.append(f"generated_samples_{generated_samples}_expected_{expected_samples}")
    if dataset == "SBM":
        if validity_metric != "sbm_graph_tool_exact":
            failures.append("sbm_backend_not_graph_tool_exact")
        if sbm_accuracy_refinement_steps != 100:
            failures.append("sbm_accuracy_refinements_not_100")
        if sbm_vun_refinement_steps != 1000:
            failures.append("sbm_vun_refinements_not_1000")
    return failures


def extract_generated_graphs(sample_file: Path) -> list[nx.Graph]:
    payload = torch.load(sample_file, map_location="cpu", weights_only=False)
    graphs = payload.get("raw_generated_graphs") or payload.get("generated_graphs")
    if graphs is None:
        raise KeyError(f"No generated graph collection in {sample_file}")
    return [simple_graph(graph) for graph in graphs]


def feature_bundle(graphs: Sequence[nx.Graph], cache: OrbitCache, label: str) -> dict:
    return {
        "degree": degree_features(graphs),
        "clustering": clustering_features(graphs),
        "orbit": cache.features(graphs, label),
    }


def relative(path: Path) -> str:
    return str(portable_path(path))


def write_per_run(output_dir: Path, result: CompatibleResult, payload: dict) -> Path:
    path = output_dir / "per_run" / f"wl{result.wl}" / result.dataset.lower() / f"{result.checkpoint_role}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_number) + "\n", encoding="utf-8")
    return path


def recompute(runs: list[RunSpec], output_dir: Path, cache: OrbitCache) -> tuple[list[CompatibleResult], dict]:
    dataset_context: dict[str, dict] = {}
    results: list[CompatibleResult] = []
    for run_idx, run in enumerate(runs, start=1):
        print(
            f"[{run_idx}/{len(runs)}] {run.dataset} WL{run.wl} {run.checkpoint_role}",
            flush=True,
        )
        original = read_json(run.metric_file)
        run_config = read_json(run.run_config_file)
        loader_config = _normalized_dataset_loader_config(run_config)
        if run.dataset not in dataset_context:
            train_graphs, test_graphs, dataset_provenance, _config_requests_verification = load_dataset_graphs(
                run,
                run_config,
            )
            train_features = feature_bundle(train_graphs, cache, f"{run.dataset}/train")
            test_features = feature_bundle(test_graphs, cache, f"{run.dataset}/test")
            local_denominators = compute_metric_set(run.dataset, train_features, test_features)
            published = PUBLISHED_DENOMINATORS.get(run.dataset)
            denominators = dict(published) if published is not None else dict(local_denominators)
            source = "spectre_published_table" if published is not None else "local_train_test"
            dataset_context[run.dataset] = {
                "loader_config": loader_config,
                "train_graphs": len(train_graphs),
                "test_graphs": len(test_graphs),
                "test_features": test_features,
                "local_denominators": local_denominators,
                "denominators": denominators,
                "denominator_source": source,
                "paper_dataset_match": True,
                "checkpoint_data_provenance_verified": True,
                "dataset_provenance": dataset_provenance,
            }
            print(
                f"{run.dataset}: local denominators {local_denominators}; selected {source} {denominators}",
                flush=True,
            )
        context = dataset_context[run.dataset]
        _require_matching_dataset_loader_config(
            run.dataset,
            context["loader_config"],
            loader_config,
            run.run_config_file,
        )
        recorded_dataset_provenance = original.get("config/dataset_provenance")
        _require_matching_recorded_dataset_provenance(
            run.dataset,
            context["dataset_provenance"],
            recorded_dataset_provenance,
            run.metric_file,
        )
        run_provenance_verified = (
            bool(run_config.get("verify_data_provenance", False))
            and not original.get("config/checkpoint_mismatch_warnings")
            and recorded_dataset_provenance == context["dataset_provenance"]
        )
        run_paper_dataset_match = (
            run_provenance_verified
            and exact_paper_dataset_match(run.dataset, context["dataset_provenance"])
        )
        context["checkpoint_data_provenance_verified"] = bool(
            context["checkpoint_data_provenance_verified"] and run_provenance_verified
        )
        context["paper_dataset_match"] = bool(
            context["paper_dataset_match"] and run_paper_dataset_match
        )
        generated_graphs = extract_generated_graphs(run.sample_file)
        recorded_generated = int(original.get("sampling/generated_graphs", -1))
        if recorded_generated != len(generated_graphs):
            raise RuntimeError(
                f"Generated payload/metrics count mismatch for {run.metric_file}: "
                f"payload={len(generated_graphs)} metrics={recorded_generated}"
            )
        if not generated_graphs:
            raise RuntimeError(f"No generated graphs in paper-compatible artifact {run.sample_file}")
        generated_features = feature_bundle(
            generated_graphs,
            cache,
            f"{run.dataset}/WL{run.wl}/{run.checkpoint_role}",
        )
        numerator = compute_metric_set(run.dataset, context["test_features"], generated_features)
        denominators = context["denominators"]
        ratios = {
            name: numerator[name] / denominators[name] if denominators[name] > 0.0 else None
            for name in numerator
        }
        available_ratios = [value for value in ratios.values() if value is not None]
        mean_ratio = (
            sum(available_ratios) / len(available_ratios)
            if available_ratios
            else None
        )
        undefined_metrics = [name for name, value in ratios.items() if value is None]
        ratio_status = (
            "all_three_defined"
            if not undefined_metrics
            else "undefined_zero_denominator:" + ",".join(undefined_metrics)
        )

        strict = read_json(run.strict_metric_file) if run.strict_metric_file.exists() else {}
        kernel_family = "gaussian_emd" if run.dataset == "COMMUNITY_SMALL" else "gaussian_tv"
        target_samples = int(original.get("sampling/target_n_samples", len(generated_graphs)))
        reference_split = str(original.get("config/reference_split") or run_config.get("reference_split") or "")
        protocol_label = comparison_protocol(run.dataset, context["dataset_provenance"])
        comparability_failures = paper_comparability_failures(
            dataset=run.dataset,
            denominator_source=context["denominator_source"],
            provenance_verified=run_provenance_verified,
            paper_dataset_match=run_paper_dataset_match,
            reference_split=reference_split,
            target_samples=target_samples,
            generated_samples=len(generated_graphs),
            validity_metric=original.get("graph/validity_metric"),
            sbm_accuracy_refinement_steps=original.get("graph/sbm_accuracy_refinement_steps"),
            sbm_vun_refinement_steps=original.get("graph/sbm_vun_refinement_steps"),
        )
        directly_comparable = not comparability_failures
        provisional_path = (
            output_dir / "per_run" / f"wl{run.wl}" / run.dataset.lower() / f"{run.checkpoint_role}.json"
        )
        result = CompatibleResult(
            dataset=run.dataset,
            wl=run.wl,
            group=run.group,
            checkpoint_role=run.checkpoint_role,
            checkpoint_epoch=original.get("config/checkpoint_epoch"),
            checkpoint_selection_metric=original.get("config/checkpoint_selection_metric"),
            checkpoint_selection_score=original.get("config/checkpoint_selection_score"),
            samples=recorded_generated,
            target_samples=target_samples,
            strict_generated=int(strict.get("sampling/generated_graphs", 0)),
            strict_generation_pct=pct(strict.get("sampling/strict_generation_fraction")),
            complete_pct=pct(original.get("sampling/complete_fraction")),
            vun_pct=pct(original.get("graph/valid_unique_novel_fraction")),
            validity_metric=original.get("graph/validity_metric"),
            sbm_accuracy_refinement_steps=original.get("graph/sbm_accuracy_refinement_steps"),
            sbm_vun_refinement_steps=original.get("graph/sbm_vun_refinement_steps"),
            degree_mmd2=numerator["degree"],
            clustering_mmd2=numerator["clustering"],
            orbit_mmd2=numerator["orbit"],
            degree_denominator=denominators["degree"],
            clustering_denominator=denominators["clustering"],
            orbit_denominator=denominators["orbit"],
            degree_local_denominator=context["local_denominators"]["degree"],
            clustering_local_denominator=context["local_denominators"]["clustering"],
            orbit_local_denominator=context["local_denominators"]["orbit"],
            degree_ratio=ratios["degree"],
            clustering_ratio=ratios["clustering"],
            orbit_ratio=ratios["orbit"],
            mean_ratio=mean_ratio,
            ratio_metrics_used=len(available_ratios),
            ratio_status=ratio_status,
            denominator_source=context["denominator_source"],
            reference_split=reference_split,
            comparison_protocol=protocol_label,
            paper_dataset_match=bool(run_paper_dataset_match),
            directly_paper_comparable=directly_comparable,
            comparability_failures=comparability_failures,
            kernel_family=kernel_family,
            orbit_backend="ORCA node-orbit graphlets up to size 4, exact",
            sample_file=relative(run.sample_file),
            original_metric_file=relative(run.metric_file),
            compatible_metric_file=relative(provisional_path),
        )
        per_run_payload = {
            "protocol": {
                "name": "DiGress/SPECTRE degree-clustering-orbit MMD compatibility",
                "mmd_estimator": "biased_mmd_squared",
                "kernel_family": kernel_family,
                "orbit_backend": result.orbit_backend,
                "orca_sha256": cache.orca_sha256,
                "denominator_source": result.denominator_source,
                "paper_dataset_match": result.paper_dataset_match,
                "directly_paper_comparable": result.directly_paper_comparable,
                "published_denominators": PUBLISHED_DENOMINATORS.get(run.dataset),
                "locally_recomputed_denominators": context["local_denominators"],
            },
            "run": asdict(result),
        }
        write_per_run(output_dir, result, per_run_payload)
        results.append(result)
    return results, dataset_context


def fmt(value: float | None, digits: int = 6) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def fmt_ratio(value: float | None) -> str:
    if value is None:
        return "--"
    if abs(value) >= 1000:
        return f"{value:.1f}"
    if abs(value) >= 100:
        return f"{value:.2f}"
    return f"{value:.3f}"


def role_label(role: str) -> str:
    return {
        "structural_validation_best": "structural-validation best",
        "fixed_beta_best": "fixed-beta best",
        "legacy_val_loss_best": "legacy-val best",
        "last_epoch": "last epoch",
    }[role]


def markdown_link(label: str, path: str) -> str:
    return f"[{label}]({path})"


def write_markdown(path: Path, results: list[CompatibleResult], contexts: dict, run_root: Path, timestamp: str):
    published_datasets = [
        dataset for dataset in DATASET_ORDER
        if dataset in contexts and dataset in PUBLISHED_DENOMINATORS
    ]
    exact_datasets = [
        dataset for dataset in published_datasets
        if (dataset_results := [result for result in results if result.dataset == dataset])
        and all(result.directly_paper_comparable for result in dataset_results)
    ]
    local_paper_scale_datasets = [
        dataset for dataset in published_datasets
        if dataset not in exact_datasets
    ]
    local_only_datasets = [
        dataset for dataset in DATASET_ORDER
        if dataset in contexts and dataset not in PUBLISHED_DENOMINATORS
    ]
    exact_text = ", ".join(exact_datasets) if exact_datasets else "none"
    local_paper_text = ", ".join(local_paper_scale_datasets) if local_paper_scale_datasets else "none"
    local_only_text = ", ".join(local_only_datasets) if local_only_datasets else "none"
    sbm_results = [result for result in results if result.dataset == "SBM"]
    exact_sbm_rows = sum(
        result.validity_metric == "sbm_graph_tool_exact"
        and result.sbm_accuracy_refinement_steps == 100
        and result.sbm_vun_refinement_steps == 1000
        for result in sbm_results
    )
    sbm_note = (
        f"All {len(sbm_results)} SBM rows use `sbm_graph_tool_exact` with 100 refinements for "
        "validity and 1,000 for V.U.N."
        if sbm_results and exact_sbm_rows == len(sbm_results)
        else f"Only {exact_sbm_rows}/{len(sbm_results)} SBM rows use the exact graph-tool backend."
    )

    lines = [
        "# Current Corrected Benchmarks: DiGress/SPECTRE-Compatible Metrics",
        "",
        f"Snapshot: **{timestamp}**",
        "",
        f"Source sweep: `{relative(run_root)}`",
        "",
        "This report recomputes degree, clustering, and orbit MMD using the public DiGress/SPECTRE evaluation convention. It covers every finished corrected permissive sample artifact for the primary best checkpoint (structural-validation best in budget-v2, fixed-beta best in legacy runs), the legacy validation-loss best checkpoint, and the last checkpoint at every finished WL stage.",
        "",
        "## Interpretation and comparability",
        "",
        "- Values named `MMD²` are the biased squared MMD estimator used by the public code. They must **not** be squared again.",
        "- `Degree ratio`, `Clustering ratio`, and `Orbit ratio` are `MMD²(generated,test) / MMD²(train,test)`.",
        "- `3-metric selection mean` is our arithmetic mean of the defined degree/clustering/orbit ratios. Lower is better. It is **not** SPECTRE's five-metric Ratio for Planar, SBM, or Proteins.",
        "- When a locally recomputed train-test denominator is exactly zero, that ratio is mathematically undefined and is printed as `--`; the mean is then over the remaining metrics and its coverage is shown. This occurs for MUTAG clustering and does not affect datasets using published SPECTRE denominators.",
        f"- {', '.join(published_datasets) or 'No loaded datasets'} use the printed SPECTRE denominators, so their ratio columns are on the DiGress paper's reported scale.",
        f"- Every row passes the full direct-comparison gate for: **{exact_text}**. The gate checks asset/split and checkpoint provenance, test reference, paper-scale sample count, and exact SBM settings.",
        f"- Paper-scale ratios without verified official dataset/split provenance: **{local_paper_text}**. These remain metric-compatible comparisons, not exact end-to-end reproductions.",
        f"- Datasets without a published DiGress/SPECTRE denominator: **{local_only_text}**. They use the same evaluator with locally recomputed train-test denominators and are comparable only within this report.",
        "- MMD is evaluated over the full permissive output set because DiGress evaluates every produced graph. `Strict generated` and `Complete` remain separate decoder diagnostics.",
        f"- {sbm_note} The official rerun scripts require the exact backend and stop before training if `graph-tool` is unavailable. PLANAR validity uses connectedness plus NetworkX planarity and matches the paper definition.",
        "",
        "Primary sources: [DiGress paper](https://arxiv.org/abs/2209.14734), [SPECTRE paper](https://arxiv.org/abs/2204.01613), [official evaluator](https://github.com/cvignac/DiGress/blob/main/src/analysis/spectre_utils.py), and [official MMD helper](https://github.com/cvignac/DiGress/blob/main/src/analysis/dist_helper.py).",
        "",
        "## Exact paper-comparability status",
        "",
        f"- Complete official asset/split runs in this report: **{exact_text}**.",
        f"- Still requiring retraining through the official rerun path: **{local_paper_text}**.",
        "- `digress_seed0` reproduces the source-membership split in current DiGress `main` (introduced in July 2023), with the official experiment counts of 20 Community and 40 Planar/SBM samples. It does not reproduce the current loader's accidental double append.",
        "- `spectre_seed1234` reproduces the SPECTRE and paper-era DiGress split (64/16/20 for Community; 128/32/40 for Planar/SBM). A result cannot use both split protocols at once.",
        "- Exact SPECTRE PROTEINS reproduction means the paper's filtered DD graph collection, Torch seed-1234 split, training and evaluation on that same split, and test-set-sized sampling.",
        "- EGO_SMALL, ENZYMES, and GRID use a full-universe 64/16/20 Torch split with seed 0 when `digress_seed0` is requested. They use the DiGress evaluator, exact ORCA, and test-set-sized sampling, but DiGress does not publish official loaders or SPECTRE denominators for them. Their normalized ratios therefore use locally recomputed train--test denominators and are protocol-compatible, not direct DiGress paper ratios.",
        "",
        "A row is labeled directly paper-comparable only when its published denominator, recorded training/evaluation provenance, test reference, paper-scale sample count, and (for SBM) graph-tool 100/1,000-step settings all pass. Changing only the evaluation test split never upgrades an old checkpoint.",
        "",
        "## Denominator audit",
        "",
        "Published values are the rounded values printed in the SPECTRE result tables, which DiGress says it uses as denominators. Local values are recomputed with the same kernels and the exact local train/test split.",
        "",
        "| Dataset | Train / test | Kernel | Ratio denominator | Published Degree / Clustering / Orbit | Local Degree / Clustering / Orbit | Exact dataset/split verified? |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for dataset in DATASET_ORDER:
        if dataset not in contexts:
            continue
        context = contexts[dataset]
        published = PUBLISHED_DENOMINATORS.get(dataset)
        published_text = "--" if published is None else " / ".join(fmt(published[key]) for key in ("degree", "clustering", "orbit"))
        local_text = " / ".join(fmt(context["local_denominators"][key]) for key in ("degree", "clustering", "orbit"))
        kernel = "Gaussian EMD" if dataset == "COMMUNITY_SMALL" else "Gaussian TV"
        direct = "Yes" if context["denominator_source"] == "spectre_published_table" and context["paper_dataset_match"] else "No"
        lines.append(
            f"| {dataset} | {context['train_graphs']} / {context['test_graphs']} | {kernel} | "
            f"{context['denominator_source']} | {published_text} | {local_text} | {direct} |"
        )

    lines.extend([
        "",
        "## Best finished compatible result per dataset",
        "",
        "This selects exactly one whole row per dataset by the lowest compatible mean ratio. Every raw MMD², normalized ratio, completion, strict-generation, and V.U.N. value in a dataset row comes from that same WL iteration and checkpoint. No metric-wise mixing is used. Ties are resolved by the lower WL iteration and then checkpoint order: structural-validation best, fixed-beta best, legacy-validation best, last epoch.",
        "",
        "| Dataset | Selected WL | Selected checkpoint | Epoch | Degree MMD² | Clustering MMD² | Orbit MMD² | Degree ratio | Clustering ratio | Orbit ratio | 3-metric selection mean | Ratio coverage | Complete | Strict generated | V.U.N. | Comparison scope |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for dataset in DATASET_ORDER:
        candidates = [result for result in results if result.dataset == dataset]
        if not candidates:
            continue
        result = min(
            candidates,
            key=lambda item: (
                float("inf") if item.mean_ratio is None else item.mean_ratio,
                item.wl,
                ROLE_ORDER[item.checkpoint_role],
            ),
        )
        if result.directly_paper_comparable:
            scope = f"direct: {result.comparison_protocol}"
        elif result.denominator_source == "spectre_published_table":
            scope = "not direct: " + ", ".join(result.comparability_failures)
        else:
            scope = "official kernels; local denominator"
        lines.append(
            f"| {dataset} | {result.wl} | {role_label(result.checkpoint_role)} | {result.checkpoint_epoch} | "
            f"{fmt(result.degree_mmd2)} | {fmt(result.clustering_mmd2)} | {fmt(result.orbit_mmd2)} | "
            f"{fmt_ratio(result.degree_ratio)} | {fmt_ratio(result.clustering_ratio)} | {fmt_ratio(result.orbit_ratio)} | "
            f"{fmt_ratio(result.mean_ratio)} | {result.ratio_metrics_used}/3 | {result.complete_pct:.2f}% | {result.strict_generated}/{result.target_samples} | "
            f"{result.vun_pct:.2f}% | {scope} |"
        )

    lines.extend([
        "",
        "## All current corrected compatible results",
        "",
        f"Total rows: **{len(results)}**. These tables are the complete selection audit. Rows not selected above remain here for comparison, but values from them are not mixed into the selected benchmark rows.",
        "",
    ])
    for dataset in DATASET_ORDER:
        dataset_results = [result for result in results if result.dataset == dataset]
        if not dataset_results:
            continue
        lines.extend([
            f"### {dataset}",
            "",
            "| WL | Checkpoint | Epoch | Samples | Strict generated | Complete | Degree MMD² | Clustering MMD² | Orbit MMD² | Degree ratio | Clustering ratio | Orbit ratio | 3-metric selection mean | Ratio coverage | V.U.N. | Direct? | Exact JSON |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ])
        for result in dataset_results:
            lines.append(
                f"| {result.wl} | {role_label(result.checkpoint_role)} | {result.checkpoint_epoch} | "
                f"{result.samples}/{result.target_samples} | {result.strict_generated}/{result.target_samples} | "
                f"{result.complete_pct:.2f}% | {fmt(result.degree_mmd2)} | {fmt(result.clustering_mmd2)} | "
                f"{fmt(result.orbit_mmd2)} | {fmt_ratio(result.degree_ratio)} | {fmt_ratio(result.clustering_ratio)} | "
                f"{fmt_ratio(result.orbit_ratio)} | {fmt_ratio(result.mean_ratio)} | {result.ratio_metrics_used}/3 | {result.vun_pct:.2f}% | "
                f"{'Yes' if result.directly_paper_comparable else 'No'} | "
                f"{markdown_link('JSON', result.compatible_metric_file)} |"
            )
        lines.append("")

    present_stages = {(result.dataset, result.wl) for result in results}
    expected_stages = {
        (dataset, wl)
        for dataset in contexts
        for wl in (1, 2, 3)
    }
    missing_stages = sorted(
        expected_stages - present_stages,
        key=lambda value: (
            DATASET_ORDER.index(value[0]) if value[0] in DATASET_ORDER else 999,
            value[1],
        ),
    )
    lines.extend([
        "## Coverage",
        "",
        "Finished dataset/WL stages included: " + str(len(present_stages)) + ".",
        "",
        "Missing expected WL1--WL3 stages: "
        + (", ".join(f"{dataset} WL{wl}" for dataset, wl in missing_stages) if missing_stages else "none")
        + ".",
        "",
        "## Reproduction",
        "",
        "```bash",
        f"{sys.executable} recompute_digress_spectre_metrics.py --run-root {run_root}",
        f"{sys.executable} generate_current_best_results_tex.py --metrics {run_root}/digress_spectre_compatible/metrics.json",
        "```",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    output_dir: Path,
    markdown: Path,
    results: list[CompatibleResult],
    contexts: dict,
    run_root: Path,
    orca_sha256: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    aggregate = {
        "snapshot": timestamp,
        "run_root": relative(run_root),
        "protocol": {
            "name": "DiGress/SPECTRE degree-clustering-orbit MMD compatibility",
            "mmd_estimator": "biased_mmd_squared",
            "selection_score": "custom_mean_of_degree_clustering_orbit_ratios",
            "orca_sha256": orca_sha256,
            "published_denominators": PUBLISHED_DENOMINATORS,
            "official_sources": {
                "digress_paper": OFFICIAL_DIGRESS_PAPER,
                "spectre_paper": OFFICIAL_SPECTRE_PAPER,
                "evaluator": OFFICIAL_EVALUATOR,
                "distance_helper": OFFICIAL_DISTANCE_HELPER,
            },
        },
        "dataset_context": {
            dataset: {
                key: value
                for key, value in context.items()
                if key not in {"test_features"}
            }
            for dataset, context in contexts.items()
        },
        "rows": [asdict(result) for result in results],
    }
    aggregate_path = output_dir / "metrics.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True, default=json_number) + "\n", encoding="utf-8")
    csv_path = output_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    write_markdown(markdown, results, contexts, run_root, timestamp)
    print(f"Wrote {aggregate_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {markdown}")


def main():
    args = parse_args()
    selected = {name.upper() for name in args.datasets} if args.datasets else None
    runs = scan_runs(args.run_root, selected)
    if not runs:
        raise SystemExit(f"No finished permissive checkpoint artifacts found under {args.run_root}")
    if any(run.dataset == "COMMUNITY_SMALL" for run in runs) and pyemd is None:
        raise SystemExit(
            "Community-small requires pyemd for literal DiGress Gaussian-EMD evaluation. "
            "Run setup_exact_graph_tool_env.sh or install pyemd in this Python environment."
        )
    expected_roles: dict[tuple[str, int], set[str]] = {}
    for run in runs:
        expected_roles.setdefault((run.dataset, run.wl), set()).add(run.checkpoint_role)
    secondary_roles = {"legacy_val_loss_best", "last_epoch"}
    primary_roles = {"fixed_beta_best", "structural_validation_best"}
    incomplete = {}
    for key, roles in expected_roles.items():
        missing = sorted(secondary_roles - roles)
        if not roles.intersection(primary_roles):
            missing.append("fixed_beta_best_or_structural_validation_best")
        unexpected = roles - secondary_roles - primary_roles
        if missing or unexpected:
            incomplete[key] = {
                "missing": missing,
                "unexpected": sorted(unexpected),
            }
    if incomplete:
        raise SystemExit(f"Finished stages missing checkpoint roles: {incomplete}")

    started = time.time()
    cache = OrbitCache(args.output_dir / "orbit_cache.sqlite3", args.orca, args.workers)
    orca_sha256 = cache.orca_sha256
    try:
        results, contexts = recompute(runs, args.output_dir, cache)
    finally:
        cache.close()
    write_outputs(args.output_dir, args.markdown, results, contexts, args.run_root, orca_sha256)
    print(f"Completed {len(results)} rows in {(time.time() - started) / 60.0:.1f} minutes.")


if __name__ == "__main__":
    main()
