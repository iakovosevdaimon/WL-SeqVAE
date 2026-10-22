"""Focused regression tests for data protocols, generation, and metrics."""

from __future__ import annotations

import json
import random
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import torch

from ring_vae_actions import ACTION_CREATE_NODE
from dataset import split_graph_dataset
from evaluate_molecule_benchmark_suites import load_run
from check_ring_vae_checkpoint import checkpoint_compatibility_errors
from gpt_datasets import (
    DIGRESS_STYLE_SPLIT_INDEX_SHA256,
    OFFICIAL_SPLIT_INDEX_SHA256,
    _official_split_provenance,
    _digress_style_generic_provenance,
    _pyg_data_to_simple_data,
    _read_smiles_file,
    _rdkit_atomic_number_vocabulary,
    _split_catflow_gdss,
    _split_digress_seed0,
    _split_full_universe_torch_seed,
    _split_spectre_seed1234,
    graph_to_simple_data,
)
from recompute_digress_spectre_metrics import (
    compute_mmd2,
    paper_comparability_failures,
)
from recompute_catflow_metrics import compute_metrics as compute_catflow_metrics
from ring_traces import (
    WL_SEMANTICS_VERSION,
    RingGraphTrace,
    RingNodeTrace,
    build_training_ring_traces,
    graph_collection_fingerprint,
    infer_ring_model_sizes,
)
from ring_vae_generator import RingWLVAEGenerator
from ring_vae_model import MODEL_SCHEMA_VERSION, RingWLVAE
from ring_vae_trace_packing import make_packed_ring_batch, pack_ring_trace
from project_paths import PROJECT_ROOT, portable_path
from run_ring_vae import largest_component_views, make_parser as make_ring_run_parser
from check_ring_vae_checkpoint import make_parser as make_checkpoint_parser
from recompute_digress_spectre_metrics import (
    _normalized_dataset_loader_config,
    _require_matching_dataset_loader_config,
    _require_matching_recorded_dataset_provenance,
)
from run_wl_witness_ar import graph_metrics
import run_wl_witness_ar as graph_metric_module
from aggregate_catflow_molecule_metrics import (
    aggregate as aggregate_catflow_molecule_values,
    load_run as load_catflow_molecule_run,
)
from train_ring_vae import (
    is_meaningful_improvement,
    legacy_validation_selection_score,
    validation_selection_score,
)
from wl_witness_ar import ColorSpec, PairedStubARGenerator, sample_graph_budgets
from zinc_utils import (
    _parse_fcd_value,
    build_smiles,
    compute_mol_metrics,
    smiles_from_mol,
)


class TinyData:
    def __init__(self, num_nodes, edges, labels=None):
        self.num_nodes = int(num_nodes)
        self.x = torch.tensor(labels or [1] * int(num_nodes), dtype=torch.long)
        rows = []
        attrs = []
        for u, v, edge_type in edges:
            rows.append((int(u), int(v)))
            rows.append((int(v), int(u)))
            attrs.extend([int(edge_type), int(edge_type)])
        self.edge_index = torch.tensor(rows, dtype=torch.long).t().contiguous()
        self.edge_attr = torch.tensor(attrs, dtype=torch.long)


def test_wl_final_stage_matches_requested_iterations():
    graph = TinyData(3, [(0, 1, 1), (1, 2, 1)])
    result = build_training_ring_traces([graph, graph], [graph], n_wl_iters=2)
    assert result.final_stage == 2
    assert result.final_stage == result.metadata["n_wl_iters"]
    assert len(result.metadata["label_dict"]) == 3


def test_raw_connectivity_is_not_measured_after_cleanup():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (2, 3)])
    metrics = graph_metrics([graph], dataset="COMMUNITY_SMALL")
    assert metrics["graph/raw_connected_fraction"] == 0.0
    assert metrics["graph/clean_connected_fraction"] == 1.0
    assert metrics["graph/connected_fraction"] == 0.0


def test_official_paper_split_protocols():
    items_100 = list(range(100))
    train, val, test = _split_digress_seed0(items_100)
    assert (len(train), len(val), len(test)) == (68, 15, 17)
    order = torch.randperm(200, generator=torch.Generator().manual_seed(0)).tolist()
    train_members = set(order[:128])
    val_members = set(order[128:160])
    test_members = set(order[160:])
    assert train == [item for item in items_100 if item in train_members]
    assert val == [item for item in items_100 if item in val_members]
    assert test == [item for item in items_100 if item in test_members]
    provenance = _official_split_provenance(100, "digress_seed0")
    assert provenance["split_index_sha256"] == OFFICIAL_SPLIT_INDEX_SHA256[("digress_seed0", 100)]
    assert provenance["split_protocol_revision"] == "23d816fc65dce923424fed1dbd33dcf4a7d2a025"

    items_918 = list(range(918))
    train, val, test = _split_spectre_seed1234(items_918)
    assert (len(train), len(val), len(test)) == (587, 147, 184)
    expected = torch.randperm(918, generator=torch.Generator().manual_seed(1234)).tolist()
    assert train + val + test == expected
    provenance = _official_split_provenance(918, "spectre_seed1234")
    assert provenance["split_index_sha256"] == OFFICIAL_SPLIT_INDEX_SHA256[("spectre_seed1234", 918)]


def test_catflow_gdss_split_is_ordered_and_has_no_split_seed():
    for size, expected_sizes in (
        (200, (160, 32, 40)),
        (100, (80, 16, 20)),
        (587, (470, 94, 117)),
    ):
        items = list(range(size))
        train, val, test, provenance = _split_catflow_gdss(items)
        test_size = int(0.2 * size)
        assert train == items[test_size:]
        assert test == items[:test_size]
        assert val == train[: len(val)]
        assert (len(train), len(val), len(test)) == expected_sizes
        assert provenance["split_rng"] == "none_ordered_collection"
        assert provenance["split_seed"] is None
        assert provenance["outer_split_policy"] == (
            "gdss_first_20_percent_test_remaining_80_percent_train"
        )


def test_digress_style_generic_seed0_split_is_full_universe_and_nonofficial():
    for dataset, size, expected_sizes in (
        ("EGO_SMALL", 200, (128, 32, 40)),
        ("ENZYMES", 587, (376, 94, 117)),
        ("GRID", 100, (64, 16, 20)),
    ):
        items = list(range(size))
        train, val, test = _split_full_universe_torch_seed(items, seed=0)
        assert (len(train), len(val), len(test)) == expected_sizes
        assert sorted(train + val + test) == items
        expected = torch.randperm(
            size,
            generator=torch.Generator().manual_seed(0),
        ).tolist()
        assert train + val + test == expected
        provenance = _digress_style_generic_provenance(
            dataset,
            size,
            protocol="digress_seed0",
            seed=0,
        )
        assert provenance["official_digress_split"] is False
        assert provenance["split_scope"] == "digress_style_full_universe_nonofficial_dataset"
        assert provenance["split_universe"] == size
        assert provenance["split_seed"] == 0
        assert provenance["split_index_sha256"] == DIGRESS_STYLE_SPLIT_INDEX_SHA256[size]


def test_paper_comparability_gate_fails_closed():
    exact = paper_comparability_failures(
        dataset="SBM",
        denominator_source="spectre_published_table",
        provenance_verified=True,
        paper_dataset_match=True,
        reference_split="test",
        target_samples=40,
        generated_samples=40,
        validity_metric="sbm_graph_tool_exact",
        sbm_accuracy_refinement_steps=100,
        sbm_vun_refinement_steps=1000,
    )
    assert exact == []
    failures = paper_comparability_failures(
        dataset="SBM",
        denominator_source="spectre_published_table",
        provenance_verified=False,
        paper_dataset_match=True,
        reference_split="train",
        target_samples=39,
        generated_samples=38,
        validity_metric="sbm_heuristic",
        sbm_accuracy_refinement_steps=1000,
        sbm_vun_refinement_steps=100,
    )
    assert failures == [
        "checkpoint_or_dataset_provenance_unverified",
        "reference_split_is_train",
        "target_samples_39_expected_40",
        "generated_samples_38_expected_40",
        "sbm_backend_not_graph_tool_exact",
        "sbm_accuracy_refinements_not_100",
        "sbm_vun_refinements_not_1000",
    ]


def test_compatible_mmd_rejects_empty_and_uses_biased_squared_estimator():
    samples = [torch.tensor([1.0, 0.0]).numpy(), torch.tensor([0.0, 1.0]).numpy()]
    value = compute_mmd2(samples, samples, kernel="gaussian_tv", sigma=1.0, is_hist=True)
    assert abs(value) < 1.0e-12
    try:
        compute_mmd2([], samples, kernel="gaussian_tv", sigma=1.0, is_hist=True)
    except RuntimeError as exc:
        assert "empty graph collection" in str(exc)
    else:
        raise AssertionError("Empty compatible MMD input should fail closed")


def test_catflow_metric_protocol_is_raw_biased_mmd_squared():
    features = {
        "degree": [
            torch.tensor([1.0, 2.0]).numpy(),
            torch.tensor([2.0, 1.0]).numpy(),
        ],
        "clustering": [
            torch.tensor([3.0, 0.0]).numpy(),
            torch.tensor([0.0, 3.0]).numpy(),
        ],
        "orbit": torch.tensor(
            [[1.0, 0.0, 2.0], [0.0, 1.0, 2.0]]
        ).numpy(),
    }
    values = compute_catflow_metrics(features, features)
    assert set(values) == {"degree", "clustering", "orbit"}
    assert all(abs(value) < 1.0e-12 for value in values.values())


def test_sbm_vun_uses_paper_refinement_count_after_unique_and_novel_checks():
    old_kind = graph_metric_module._graph_validity_kind
    old_valid = graph_metric_module._is_valid_for_dataset
    calls = []
    try:
        graph_metric_module._graph_validity_kind = lambda *_args, **_kwargs: "sbm_graph_tool_exact"

        def record_validity(_graph, _dataset, *, validity_kind=None, sbm_refinement_steps=100):
            calls.append((validity_kind, sbm_refinement_steps))
            return True

        graph_metric_module._is_valid_for_dataset = record_validity
        training_graph = nx.path_graph(3)
        novel_graph = nx.cycle_graph(4)
        metrics = graph_metric_module.graph_metrics(
            [training_graph.copy(), training_graph.copy(), novel_graph],
            train_graphs=[training_graph],
            dataset="SBM",
            sbm_validity_backend="exact",
            sbm_refinement_steps=100,
            sbm_vun_refinement_steps=1000,
        )
    finally:
        graph_metric_module._graph_validity_kind = old_kind
        graph_metric_module._is_valid_for_dataset = old_valid
    assert metrics["graph/n_valid_unique_novel"] == 1
    assert metrics["graph/sbm_accuracy_refinement_steps"] == 100
    assert metrics["graph/sbm_vun_refinement_steps"] == 1000
    assert sum(steps == 1000 for _kind, steps in calls) == 1


def test_clean_validity_and_planarity_are_measured_on_clean_graph():
    graph = nx.path_graph(4)
    metrics = graph_metrics([graph], dataset="PLANAR")
    assert metrics["graph/after_cleanup"] == 1
    assert metrics["graph/clean_valid"] == 1
    assert metrics["graph/valid_clean_fraction"] == 1.0
    assert metrics["graph/clean_planar_fraction"] == 1.0


def test_component_diagnostics_and_largest_component_view():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (2, 3)])
    graph.add_node(4)
    metrics = graph_metrics([graph], dataset="COMMUNITY_SMALL")
    assert metrics["graph/raw_mean_component_count"] == 3.0
    assert metrics["graph/raw_mean_isolated_nodes"] == 1.0
    assert metrics["graph/raw_multiple_components_fraction"] == 1.0
    assert metrics["graph/raw_isolate_fraction"] == 1.0
    assert metrics["graph/raw_mean_largest_component_fraction"] == 2.0 / 5.0
    assert metrics["graph/cleanup_mean_removed_nodes"] == 3.0

    sample = SimpleNamespace(graph=graph, stats={})
    cleaned, source_indices = largest_component_views([sample])
    assert source_indices == [0]
    assert len(cleaned) == 1
    assert cleaned[0].number_of_nodes() == 2
    assert nx.is_connected(cleaned[0])
    assert sample.stats["raw_n_components"] == 3
    assert sample.stats["raw_n_isolates"] == 1
    assert sample.stats["cleanup_removed_nodes"] == 3


def test_fixed_beta_checkpoint_selection_score():
    metrics = {"recon": 2.0, "kl": 3.0, "loss": 2.03}
    assert validation_selection_score(metrics, 0.1) == 2.3
    assert legacy_validation_selection_score(metrics) == 2.03
    assert is_meaningful_improvement(9.98, 10.0, 0.001)
    assert not is_meaningful_improvement(9.995, 10.0, 0.001)


def test_fixed_and_legacy_selectors_can_choose_different_epochs():
    epochs = [
        {"epoch": 1, "recon": 2.0, "kl": 10.0, "loss": 2.1},
        {"epoch": 2, "recon": 2.5, "kl": 0.0, "loss": 2.5},
    ]
    fixed_best = min(epochs, key=lambda row: validation_selection_score(row, 0.1))
    legacy_best = min(epochs, key=legacy_validation_selection_score)
    assert fixed_best["epoch"] == 2
    assert legacy_best["epoch"] == 1


def test_fcd_parser_rejects_unlabeled_progress_numbers():
    assert _parse_fcd_value("Processed 100 molecules in 2.5 seconds") is None
    assert _parse_fcd_value("fcd: 1.25") == 1.25
    assert _parse_fcd_value("3.5") == 3.5
    assert _parse_fcd_value(json.dumps({"fcd": 0.75})) == 0.75


def test_headerless_csv_keeps_first_smiles():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mols.csv"
        path.write_text("CCO\nCCN\nCCC\n")
        assert _read_smiles_file(path) == ["CCO", "CCN", "CCC"]

        headed = Path(tmp) / "headed.csv"
        headed.write_text("smiles,name\nCO,methanol\nCCO,ethanol\n")
        assert _read_smiles_file(headed) == ["CO", "CCO"]


def test_pyg_tu_fallback_preserves_label_categories():
    data = SimpleNamespace(
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        y=torch.tensor([1]),
        num_nodes=2,
    )
    graph = _pyg_data_to_simple_data(data, preserve_edge_labels=True)
    assert graph.x.tolist() == [1, 2]
    assert graph.edge_attr.tolist() == [2, 2]
    assert graph.y.tolist() == [1]


def test_smiles_atom_ids_follow_atomic_numbers_independent_of_input_order():
    from rdkit import Chem

    atom_vocab, atom_decoder = _rdkit_atomic_number_vocabulary()
    first = graph_to_simple_data(Chem.MolFromSmiles("OCCl"), atom_vocab=atom_vocab)
    second = graph_to_simple_data(Chem.MolFromSmiles("ClCO"), atom_vocab=atom_vocab)
    assert first.x.tolist() == [8, 6, 17]
    assert second.x.tolist() == [17, 6, 8]
    assert atom_decoder[6] == "C"
    assert atom_decoder[8] == "O"
    assert atom_decoder[17] == "Cl"


def test_no_include_invalid_placeholders_does_not_pad():
    with tempfile.TemporaryDirectory() as tmp:
        samples = Path(tmp) / "variant" / "samples"
        samples.mkdir(parents=True)
        (samples / "metrics.json").write_text(json.dumps({
            "config/dataset": "MOSES",
            "config/mode": "molecule",
            "mol/fcd_attempts": 4,
        }))
        (samples / "valid_smiles.txt").write_text("CCO\nCCN\n")
        run = load_run(samples, include_invalid=False)
        assert run is not None
        assert run["generated_smiles"] == ["CCO", "CCN"]

        padded = load_run(samples, include_invalid=True)
        assert padded is not None
        assert len(padded["generated_smiles"]) == 4


def test_action_targets_are_chronological():
    nodes = [
        RingNodeTrace(0, 0, 1, 2, 6, ((1, 1),) * 3, ((1, 1),) * 3),
        RingNodeTrace(1, 1, 1, 2, 6, ((1, 1),) * 3, ((1, 1),) * 3),
        RingNodeTrace(2, 1, 1, 2, 6, ((1, 1),) * 3, ((1, 1),) * 3),
        RingNodeTrace(3, 1, 1, 2, 6, ((1, 1),) * 3, ((1, 1),) * 3),
    ]
    trace = RingGraphTrace(
        nodes=nodes,
        rings=[[0], [1, 2, 3]],
        edges=[
            (0, 1, 1),
            (0, 2, 1),
            (0, 3, 1),
            (1, 2, 1),
            (1, 3, 1),
            (2, 3, 1),
        ],
        sequence=[2, 2, 2, 2],
        target_nodes=4,
        center_color=2,
    )
    packed = pack_ring_trace(trace)
    currents = packed.action_unresolved_stubs
    nexts = packed.action_next_unresolved_stubs
    deltas = packed.action_delta_unresolved_stubs
    assert all(next_value == current + delta for current, next_value, delta in zip(currents, nexts, deltas))
    assert all(currents[idx] == nexts[idx - 1] for idx in range(1, len(currents)))


def test_edge_contexts_follow_decoder_timeline_without_target_leakage():
    nodes = [
        RingNodeTrace(i, 0 if i == 0 else 1, 1, 2, 6, ((1, 1),) * 3, ((1, 1),) * 3)
        for i in range(4)
    ]
    trace = RingGraphTrace(
        nodes=nodes,
        rings=[[0], [1, 2, 3]],
        edges=[
            (0, 1, 1),
            (0, 2, 1),
            (0, 3, 1),
            (1, 2, 1),
            (1, 3, 1),
            (2, 3, 1),
        ],
        sequence=[2, 2, 2, 2],
        target_nodes=4,
        center_color=2,
    )
    packed = pack_ring_trace(trace)
    assert packed.prev_edge_targets == ()  # Mandatory parent edges are not learned decisions.
    assert packed.color_unresolved_stubs == (3, 4, 5)
    assert packed.intra_edge_prefix_counts == (4, 4, 4)
    assert packed.intra_edge_unresolved_stubs == (6, 4, 2)
    assert packed.intra_edge_targets == (1.0, 1.0, 1.0)
    assert packed.target_edges == 6
    assert all(0 <= value < packed.target_edges for value in packed.color_current_edges)
    assert all(0 <= value < packed.target_edges for value in packed.intra_edge_current_edges)


def test_rare_wl_backoff_keeps_requested_stage_in_encoder_only():
    graph = TinyData(
        5,
        [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 4, 1)],
    )
    result = build_training_ring_traces(
        [graph, graph, graph],
        [graph],
        n_wl_iters=2,
        decoder_color_min_frequency=100,
        decoder_rare_fraction_threshold=0.0,
        dataset="PLANAR",
    )
    assert result.metadata["requested_n_wl_iters"] == 2
    assert result.final_stage == 1
    assert result.metadata["decoder_color_backoff"]["applied"] is True
    packed = make_packed_ring_batch(result.train_traces[:1])
    assert packed.sequences[0] == tuple(result.train_traces[0].sequence)
    assert packed.encoder_sequences[0] == tuple(result.train_traces[0].encoder_sequence)
    sizes = infer_ring_model_sizes(result.metadata, result.final_stage, result.color_specs)
    assert sizes["encoder_color_vocab_size"] >= sizes["color_vocab_size"]
    model = RingWLVAE(
        **sizes,
        latent_dim=4,
        hidden_dim=8,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        max_seq_len=8,
        count_vocab_size=32,
        use_budget_context=True,
        use_edge_budget_context=True,
    )
    mu, logvar = model.encode_sequences(packed.encoder_sequences)
    assert mu.shape == logvar.shape == (1, 4)
    loss = model.compute_loss(packed)["loss"]
    assert torch.isfinite(loss)


def test_joint_graph_budget_sampling_preserves_profiles():
    metadata = {
        "train_graph_budgets": [
            {"nodes": 10, "edges": 20, "block_sizes": [5, 5]},
            {"nodes": 12, "edges": 33, "block_sizes": [6, 6]},
        ]
    }
    sampled = sample_graph_budgets(metadata, 20, random.Random(17))
    allowed = {(10, 20, (5, 5)), (12, 33, (6, 6))}
    assert {
        (row["nodes"], row["edges"], tuple(row["block_sizes"]))
        for row in sampled
    } <= allowed


def test_planar_budget_repair_rejects_nonplanar_edge():
    generator = object.__new__(RingWLVAEGenerator)
    generator.constraint_kind = "planar"
    generator.enforce_edge_budget = True
    generator._active_target_edges = 9
    generator._active_constraint_profile = {}
    graph = nx.complete_bipartite_graph(3, 3)
    graph.remove_edge(0, 3)
    nodes = [
        {"label": 1, "prev_color": 1, "required": {}}
        for _ in graph.nodes()
    ]
    assert generator._force_budget_edge(graph, nodes, 0, 3, 1) is False
    assert not graph.has_edge(0, 3)


def test_planar_node_budget_repair_stays_connected():
    generator = object.__new__(RingWLVAEGenerator)
    generator.use_budget_repair = True
    generator.constraint_kind = "planar"
    generator.enforce_edge_budget = True
    generator._active_target_nodes = 4
    generator._active_target_edges = 3
    generator._active_degree_targets = [2, 2, 2, 2]
    generator._active_constraint_profile = {}
    generator._active_block_remaining = Counter()
    generator.seed_color_ids = [1]
    generator.color_specs_by_id = {
        1: ColorSpec(1, 1, ((1, 1), (1, 1)), Counter({(1, 1): 2}), 1.0)
    }
    generator.final_stage = 1
    generator.color_hierarchy = SimpleNamespace(
        parent=lambda _stage, _color: 1,
        neighbor_counts={1: {1: Counter({(1, 1): 2})}},
    )
    generator.rng = random.Random(3)
    graph = nx.Graph()
    graph.add_nodes_from([(0, {"label": 1}), (1, {"label": 1})])
    graph.add_edge(0, 1, bond_type=1)
    nodes = [
        {
            "final_color": 1,
            "prev_color": 1,
            "label": 1,
            "ring": node,
            "required": Counter(),
            "degree": 2,
        }
        for node in range(2)
    ]
    added = generator._node_budget_repair(graph, nodes)
    assert added == 2
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 3
    assert nx.is_connected(graph)
    assert nx.check_planarity(graph)[0]


def test_molecule_budget_constraint_checks_bond_order():
    generator = object.__new__(RingWLVAEGenerator)
    generator.constraint_kind = "molecule"
    generator.enforce_edge_budget = True
    generator._active_target_edges = 4
    generator._active_constraint_profile = {}
    generator.molecule_constraint_profile = {
        "allowed_bonds": [[6, 6, 2]],
        "max_degree_by_label": {"6": 3},
        "max_bond_order_by_label": {"6": 3},
    }
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2])
    graph.add_edge(0, 1, bond_type=2)
    nodes = [{"label": 6}, {"label": 6}, {"label": 6}]
    assert generator._constraint_allows_edge(graph, nodes, 0, 2, 2) is False


def _tiny_ring_model(*, use_budget_context=False):
    model = RingWLVAE(
        color_vocab_size=2,
        prev_color_vocab_size=2,
        label_vocab_size=2,
        edge_vocab_size=1,
        latent_dim=4,
        hidden_dim=8,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        max_seq_len=8,
        max_ring=4,
        count_vocab_size=8,
        use_budget_context=use_budget_context,
    )
    model.eval()
    return model


def test_generator_graph_summary_does_not_preapply_budget():
    model = _tiny_ring_model(use_budget_context=True)
    generator = object.__new__(RingWLVAEGenerator)
    generator.model = model
    nodes = [{"final_color": 1, "prev_color": 1, "label": 0, "ring": 0, "degree": 1}]
    z_context = torch.zeros(model.hidden_dim)
    expected = model.graph_summary([generator._node_repr(nodes[0])], z_context, budget_ctx=None)
    actual = generator._graph_summary(nodes, z_context, target_nodes=4, ring=1)
    assert torch.allclose(actual, expected)


def test_zero_node_label_embedding_is_trainable():
    model = _tiny_ring_model()
    assert model.label_emb.padding_idx is None
    model.zero_grad(set_to_none=True)
    model.label_emb(torch.tensor([0])).sum().backward()
    assert torch.count_nonzero(model.label_emb.weight.grad[0]).item() == model.hidden_dim


def test_ring_local_candidates_and_latent_plan_ring_guidance():
    generator = object.__new__(RingWLVAEGenerator)
    generator.rng = random.Random(0)
    generator._can_connect = lambda _nodes, _src, _dst, _edge: True
    nodes = [
        {"ring": 0, "prev_color": 1},
        {"ring": 2, "prev_color": 1},
        {"ring": 1, "prev_color": 1},
    ]
    graph = nx.Graph()
    graph.add_nodes_from(range(3))
    candidates = generator._existing_candidates_for_stub(graph, nodes, 0, 1, 1)
    assert candidates == [2]

    chosen = generator._choose_stub_for_new_node(
        nodes,
        [(0, 1, 1), (2, 1, 1)],
        None,
        1,
        plan_actions=[{}, {"ring": 2, "parent_edge_type": 1, "parent_prev_color": 1}],
    )
    assert chosen == (2, 1, 1)

    generator.use_learned_actions = True
    generator.max_rings = 4
    generator.greedy = True
    generator.action_temperature = 1.0
    generator._action_logits_for_stub = lambda *_args, **_kwargs: torch.tensor([0.0, 1.0, 0.0])
    generator._existing_candidates_for_stub = lambda *_args, **_kwargs: []
    generator._child_candidates = lambda **_kwargs: [1]
    learned = generator._choose_learned_action(
        graph,
        nodes,
        torch.zeros(1),
        [(0, 1, 1), (2, 1, 1)],
        hard_max_nodes=4,
        target_nodes=4,
        ring=2,
        planned_node_action={"ring": 2, "parent_edge_type": 1, "parent_prev_color": 1},
    )
    assert learned == (ACTION_CREATE_NODE, 2, 1, 1)


def _mock_complete_two_node_generator(*, accept_stop: bool):
    generator = object.__new__(RingWLVAEGenerator)
    generator.model = SimpleNamespace(
        eval=lambda: None,
        sample_latent=lambda n: torch.zeros((n, 1)),
        z_context=lambda z: z,
        use_latent_plan=False,
        budget_context=lambda **_kwargs: torch.zeros(1),
        center_logits=lambda *_args, **_kwargs: torch.zeros(2),
    )
    generator.max_restarts = 1
    generator.max_rings = 1
    generator.use_stub_closure = True
    generator.use_learned_actions = True
    generator.connect_existing_first = True
    generator.seed_color_ids = [1]
    generator._hard_node_budget = lambda _target: 2
    generator._plan_context = lambda *_args: None
    generator._sample_color = lambda *_args, **_kwargs: 1

    def add_node(graph, nodes, *, final_color, ring):
        node_id = len(nodes)
        nodes.append({
            "final_color": int(final_color),
            "prev_color": 1,
            "label": 1,
            "ring": int(ring),
            "required": {(1, 1): 1} if node_id == 0 else {},
            "degree": 1,
        })
        graph.add_node(node_id)
        return node_id

    def create_node(graph, nodes, _z_ctx, **kwargs):
        nodes[0]["required"].clear()
        node_id = len(nodes)
        nodes.append({
            "final_color": 1,
            "prev_color": 1,
            "label": 1,
            "ring": int(kwargs["ring"]),
            "required": {},
            "degree": 1,
        })
        graph.add_node(node_id)
        graph.add_edge(0, node_id)
        return True, node_id, None

    generator._add_node = add_node
    generator._choose_learned_action = lambda *_args, **_kwargs: None
    generator._stubs_with_existing_candidates = lambda *_args: []
    generator._choose_stub_for_new_node = lambda *_args, **_kwargs: (0, 1, 1)
    generator._create_node_for_stub = create_node
    generator._learned_stop_decision = lambda *_args, **_kwargs: bool(accept_stop)
    generator._acceptable_partial = lambda *_args: True
    return generator


def test_learned_stop_rejection_restarts_instead_of_accepting():
    rejected = _mock_complete_two_node_generator(accept_stop=False).sample_one(2)
    assert rejected.graph is None
    assert rejected.stats["failure"] == "restarts_exhausted"
    assert rejected.stats["failure_counts"] == {"learned_stop_rejected": 1}

    accepted = _mock_complete_two_node_generator(accept_stop=True).sample_one(2)
    assert accepted.graph is not None
    assert accepted.stats["complete"]
    assert accepted.stats["n_rings"] == 2


def test_stub_closure_respects_max_rings():
    generator = object.__new__(RingWLVAEGenerator)
    generator.max_closure_steps_per_node = 2
    generator.max_rings = 1
    generator.use_learned_closure = False
    generator._unresolved_entries = lambda _nodes: [(0, 1, 1)]
    generator._plan_context = lambda *_args: None
    generator._stubs_with_existing_candidates = lambda *_args: []
    generator._choose_stub_for_new_node = lambda *_args, **_kwargs: (0, 1, 1)
    nodes = [{"ring": 1, "prev_color": 1, "required": {(1, 1): 1}}]
    graph = nx.Graph()
    graph.add_node(0)
    completed, _edges = generator._stub_closure(
        graph,
        nodes,
        torch.zeros(1),
        hard_max_nodes=3,
        target_nodes=2,
    )
    assert not completed
    assert len(nodes) == 1


def test_generic_degree_budget_uses_default_cap():
    generator = SimpleNamespace(max_budget_by_label={"__default__": 0}, edge_weight_mode="unit")
    assert not PairedStubARGenerator._budget_ok(generator, label=99, used=0, edge_type=1)


def test_graph_fingerprint_covers_all_graphs_and_nodes():
    graphs = []
    for _ in range(129):
        graph = nx.path_graph(40)
        nx.set_node_attributes(graph, 0, "label")
        graphs.append(graph)
    before = graph_collection_fingerprint(graphs)
    graphs[128].nodes[39]["label"] = 1
    after = graph_collection_fingerprint(graphs)
    assert before != after

    large_graph = nx.path_graph(80)
    nx.set_node_attributes(large_graph, 0, "label")
    nx.set_edge_attributes(large_graph, 1, "bond_type")
    before = graph_collection_fingerprint([large_graph])
    large_graph.edges[70, 71]["bond_type"] = 2
    after = graph_collection_fingerprint([large_graph])
    assert before != after


def test_smiles_cache_invalidates_when_graphs_change():
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "train_smiles.json"
        carbon = nx.Graph()
        carbon.add_node(0, label=0)
        carbon.add_node(1, label=0)
        carbon.add_edge(0, 1, bond_type=1)
        assert build_smiles([carbon], ["C", "O"], cache_path=cache) == ["CC"]

        alcohol = carbon.copy()
        alcohol.nodes[1]["label"] = 1
        assert build_smiles([alcohol], ["C", "O"], cache_path=cache) == ["CO"]
        assert cache.with_name(f"{cache.name}.meta.json").is_file()


def test_explicit_h_uniqueness_is_not_replaced_by_h_stripped_uniqueness():
    def carbon_with_explicit_hydrogens(count):
        graph = nx.Graph()
        graph.add_node(0, label=1)
        for hydrogen in range(int(count)):
            graph.add_node(hydrogen + 1, label=0)
            graph.add_edge(0, hydrogen + 1, bond_type=1)
        return graph

    graphs = [carbon_with_explicit_hydrogens(3), carbon_with_explicit_hydrogens(4)]
    atom_decoder = ["H", "C", "N", "O", "F"]
    explicit_h = compute_mol_metrics(graphs, [], atom_decoder, preprocess="repo")
    h_stripped = compute_mol_metrics(graphs, [], atom_decoder, preprocess="catflow")
    assert explicit_h["uniqueness"] == 1.0
    assert h_stripped["uniqueness"] == 0.5


def test_catflow_molecule_validation_precedes_largest_fragment_selection():
    from rdkit import Chem

    valid_disconnected = Chem.MolFromSmiles("CC.C", sanitize=False)
    assert smiles_from_mol(valid_disconnected, preprocess="catflow") == "CC"

    invalid_component = Chem.MolFromSmiles("CC.C=[C](C)(C)C", sanitize=False)
    assert invalid_component is not None
    assert smiles_from_mol(invalid_component, preprocess="catflow") is None


def test_checkpoint_schema_and_feature_validation():
    checkpoint = {
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "metadata": {"n_wl_iters": 2, "label_dict": [{}, {}, {}]},
        "final_stage": 2,
        "wl_semantics": WL_SEMANTICS_VERSION,
        "args": {"dataset": "QM9"},
        "model_args": {"use_budget_context": True, "use_action_head": False},
    }
    assert checkpoint_compatibility_errors(
        checkpoint,
        dataset="QM9",
        n_wl_iters=2,
        expected_features={"budget_context": True, "action_head": False},
    ) == []
    stale = dict(checkpoint)
    stale.pop("model_schema_version")
    errors = checkpoint_compatibility_errors(stale, dataset="QM9", n_wl_iters=2)
    assert any("model schema" in error for error in errors)

    checkpoint["provenance"] = {
        "dataset_provenance": {
            "source": "official_spectre_asset",
            "paper_split_protocol": "digress_seed0",
        },
        "train_fingerprint": "train-a",
        "val_fingerprint": "val-a",
    }
    errors = checkpoint_compatibility_errors(
        checkpoint,
        expected_data_provenance={
            "source": "official_spectre_asset",
            "paper_split_protocol": "digress_seed0",
        },
        expected_train_fingerprint="train-b",
        expected_val_fingerprint="val-a",
    )
    assert errors == ["checkpoint training-graph fingerprint does not match the requested split"]
    errors = checkpoint_compatibility_errors(
        checkpoint,
        expected_training_args={"epochs": 75, "batch_size": 32},
    )
    assert errors == [
        "checkpoint training arg epochs=None != expected 75",
        "checkpoint training arg batch_size=None != expected 32",
    ]


def test_provenance_clis_accept_training_graph_caps():
    run_args = make_ring_run_parser().parse_args([
        "--checkpoint", "model.pt",
        "--max-train-graphs", "11",
        "--max-val-graphs", "7",
        "--paper-split-protocol", "catflow_gdss",
        "--qm9-protocol", "gdss_catflow",
        "--zinc250k-protocol", "gdss_catflow",
    ])
    assert (run_args.max_train_graphs, run_args.max_val_graphs) == (11, 7)
    assert run_args.paper_split_protocol == "catflow_gdss"
    assert run_args.qm9_protocol == "gdss_catflow"
    assert run_args.zinc250k_protocol == "gdss_catflow"

    check_args = make_checkpoint_parser().parse_args([
        "--checkpoint", "model.pt",
        "--max-train-graphs", "11",
        "--max-val-graphs", "7",
        "--zinc-subset", "large",
        "--smiles-file", "mols.smi",
        "--moses-split", "test",
        "--qm9-protocol", "gdss_catflow",
        "--zinc250k-protocol", "gdss_catflow",
    ])
    assert (check_args.max_train_graphs, check_args.max_val_graphs) == (11, 7)
    assert check_args.zinc_subset == "large"
    assert check_args.smiles_file == "mols.smi"
    assert check_args.moses_split == "test"
    assert check_args.qm9_protocol == "gdss_catflow"
    assert check_args.zinc250k_protocol == "gdss_catflow"


def test_benchmark_recompute_rejects_mixed_dataset_configs():
    first = _normalized_dataset_loader_config({"seed": 0, "paper_split_protocol": "digress_seed0"})
    same = _normalized_dataset_loader_config({"seed": 0, "paper_split_protocol": "digress_seed0"})
    _require_matching_dataset_loader_config("PLANAR", first, same, Path("same.json"))

    different = _normalized_dataset_loader_config({"seed": 1, "paper_split_protocol": "digress_seed0"})
    try:
        _require_matching_dataset_loader_config("PLANAR", first, different, Path("different.json"))
    except RuntimeError as exc:
        assert "Mixed dataset loader configurations" in str(exc)
    else:
        raise AssertionError("Mixed benchmark dataset configurations were accepted")

    proteins_seed0 = _normalized_dataset_loader_config(
        {"seed": 0, "paper_split_protocol": "spectre_seed1234"}
    )
    proteins_seed1 = _normalized_dataset_loader_config(
        {"seed": 1, "paper_split_protocol": "spectre_seed1234"}
    )
    _require_matching_dataset_loader_config(
        "PROTEINS", proteins_seed0, proteins_seed1, Path("model_seed1.json")
    )

    provenance = {"source": "official_spectre_asset", "split": "test"}
    _require_matching_recorded_dataset_provenance(
        "PLANAR", provenance, dict(provenance), Path("same.json")
    )
    _require_matching_recorded_dataset_provenance("PLANAR", provenance, None, Path("legacy.json"))
    try:
        _require_matching_recorded_dataset_provenance(
            "PLANAR",
            provenance,
            {"source": "different_asset", "split": "test"},
            Path("different.json"),
        )
    except RuntimeError as exc:
        assert "Recorded dataset provenance does not match" in str(exc)
    else:
        raise AssertionError("Mismatched recorded dataset provenance was accepted")


def test_small_graph_split_preserves_test_set_when_possible():
    assert tuple(len(part) for part in split_graph_dataset([], seed=0)) == (0, 0, 0)
    assert tuple(len(part) for part in split_graph_dataset([0], seed=0)) == (1, 0, 0)
    assert tuple(len(part) for part in split_graph_dataset([0, 1], seed=0)) == (1, 1, 0)
    for n in range(3, 9):
        sizes = tuple(len(part) for part in split_graph_dataset(list(range(n)), seed=0))
        assert all(size > 0 for size in sizes)
        assert sum(sizes) == n


def test_portable_metadata_paths_do_not_expose_external_directories():
    internal = PROJECT_ROOT / "checkpoints" / "model.pt"
    assert portable_path(internal) == "checkpoints/model.pt"
    with tempfile.TemporaryDirectory() as temp_dir:
        external = Path(temp_dir) / "model.pt"
        value = portable_path(external)
        assert value == "<external>/model.pt"
        assert temp_dir not in value


def test_catflow_molecule_aggregation_is_seed_wl_and_protocol_strict():
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        metrics_path = (
            root
            / "wl2"
            / "seed7"
            / "qm9"
            / "budget_allow_incomplete"
            / "samples"
            / "metrics.json"
        )
        metrics_path.parent.mkdir(parents=True)
        payload = {
            "config/dataset": "QM9",
            "config/checkpoint_role": "structural_validation_best",
            "config/checkpoint_epoch": 12,
            "config/checkpoint_mismatch_warnings": [],
            "config/verify_data_provenance": True,
            "config/requested_n_wl_iters": 2,
            "config/reference_split": "test",
            "config/require_complete": False,
            "sampling/generated_graphs": 10000,
            "mol/validity": 0.9,
            "mol/uniqueness": 0.8,
            "mol/fcd": 1.25,
            "mol/fcd_attempts": 10000,
            "mol/fcd_reference_split": "test",
            "mol/catflow_table_comparable": True,
            "mol/catflow_comparability_failures": [],
        }
        metrics_path.write_text(json.dumps(payload))
        run = load_catflow_molecule_run(metrics_path, root, "QM9")
        assert run is not None
        assert (run.wl, run.seed) == (2, 7)
        assert run.directly_catflow_comparable
        summary = aggregate_catflow_molecule_values([1.0, 2.0, 3.0])
        assert summary == {"n": 3, "mean": 2.0, "population_std": (2.0 / 3.0) ** 0.5}

        payload["mol/fcd_attempts"] = 9999
        metrics_path.write_text(json.dumps(payload))
        invalid = load_catflow_molecule_run(metrics_path, root, "QM9")
        assert invalid is not None
        assert not invalid.directly_catflow_comparable
        assert "ten_thousand_attempts" in invalid.comparability_failures

        payload["mol/fcd_attempts"] = 10000
        payload["config/checkpoint_role"] = "legacy_warmup_weighted_val_loss_best"
        metrics_path.write_text(json.dumps(payload))
        legacy = load_catflow_molecule_run(metrics_path, root, "QM9")
        assert legacy is not None
        assert legacy.checkpoint_role == "legacy_warmup_weighted_val_loss_best"


def main():
    tests = [
        test_wl_final_stage_matches_requested_iterations,
        test_raw_connectivity_is_not_measured_after_cleanup,
        test_official_paper_split_protocols,
        test_catflow_gdss_split_is_ordered_and_has_no_split_seed,
        test_digress_style_generic_seed0_split_is_full_universe_and_nonofficial,
        test_paper_comparability_gate_fails_closed,
        test_compatible_mmd_rejects_empty_and_uses_biased_squared_estimator,
        test_catflow_metric_protocol_is_raw_biased_mmd_squared,
        test_sbm_vun_uses_paper_refinement_count_after_unique_and_novel_checks,
        test_clean_validity_and_planarity_are_measured_on_clean_graph,
        test_component_diagnostics_and_largest_component_view,
        test_fixed_beta_checkpoint_selection_score,
        test_fixed_and_legacy_selectors_can_choose_different_epochs,
        test_fcd_parser_rejects_unlabeled_progress_numbers,
        test_headerless_csv_keeps_first_smiles,
        test_pyg_tu_fallback_preserves_label_categories,
        test_smiles_atom_ids_follow_atomic_numbers_independent_of_input_order,
        test_no_include_invalid_placeholders_does_not_pad,
        test_action_targets_are_chronological,
        test_edge_contexts_follow_decoder_timeline_without_target_leakage,
        test_rare_wl_backoff_keeps_requested_stage_in_encoder_only,
        test_joint_graph_budget_sampling_preserves_profiles,
        test_planar_budget_repair_rejects_nonplanar_edge,
        test_planar_node_budget_repair_stays_connected,
        test_molecule_budget_constraint_checks_bond_order,
        test_generator_graph_summary_does_not_preapply_budget,
        test_zero_node_label_embedding_is_trainable,
        test_ring_local_candidates_and_latent_plan_ring_guidance,
        test_learned_stop_rejection_restarts_instead_of_accepting,
        test_stub_closure_respects_max_rings,
        test_generic_degree_budget_uses_default_cap,
        test_graph_fingerprint_covers_all_graphs_and_nodes,
        test_smiles_cache_invalidates_when_graphs_change,
        test_explicit_h_uniqueness_is_not_replaced_by_h_stripped_uniqueness,
        test_catflow_molecule_validation_precedes_largest_fragment_selection,
        test_checkpoint_schema_and_feature_validation,
        test_provenance_clis_accept_training_graph_caps,
        test_benchmark_recompute_rejects_mixed_dataset_configs,
        test_small_graph_split_preserves_test_set_when_possible,
        test_portable_metadata_paths_do_not_expose_external_directories,
        test_catflow_molecule_aggregation_is_seed_wl_and_protocol_strict,
    ]
    for test in tests:
        test()
        print(f"ok {test.__name__}")


if __name__ == "__main__":
    main()
