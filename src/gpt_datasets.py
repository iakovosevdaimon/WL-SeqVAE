"""Dataset loaders and provenance checks for graph-generation experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import torch

from dataset import SimpleGraphData, load_tu_graph_dataset, split_graph_dataset
from project_paths import DATASETS_ROOT, portable_path
from warning_filters import configure_warning_filters
from zinc_utils import load_atom_decoder, load_processed_zinc_split


MOLECULE_DATASETS = {"ZINC", "ZINC250K", "QM9", "QM9_H", "MOSES", "GUACAMOL"}
QM9_PROTOCOLS = ("legacy_local", "gdss_catflow")
ZINC250K_PROTOCOLS = ("legacy_pyg", "gdss_catflow")
TU_DATASETS = {"MUTAG", "NCI1"}
PAPER_GENERIC_DATASETS = {
    "COMMUNITY_SMALL",
    "EGO_SMALL",
    "ENZYMES",
    "GRID",
    "PLANAR",
    "SBM",
    "PROTEINS",
}
DIGRESS_STYLE_GENERIC_DATASETS = {"EGO_SMALL", "ENZYMES", "GRID"}
GENERIC_DATASETS = TU_DATASETS | PAPER_GENERIC_DATASETS
SUPPORTED_DATASETS = tuple(sorted(MOLECULE_DATASETS | GENERIC_DATASETS))
PAPER_DATASET_SEED = 1234
PAPER_SPLIT_PROTOCOLS = (
    "legacy_local",
    "catflow_gdss",
    "digress_seed0",
    "spectre_seed1234",
)

SPECTRE_SOURCE_REVISION = "f676c0d55fe0eba2fbf10133291c9aeac89423e6"
DIGRESS_MAIN_SPLIT_REVISION = "23d816fc65dce923424fed1dbd33dcf4a7d2a025"
DIGRESS_PAPER_SPLIT_REVISION = "72f011b36a8a38c3fe5e9cb1530e1ec9b48dc62e"

CATFLOW_GDSS_REVISION = "24cc490e0c5b39cbd265fda33ab2bc8b528d4454"
CATFLOW_GDSS_GENERIC_DATASETS = {
    "EGO_SMALL",
    "COMMUNITY_SMALL",
    "ENZYMES",
    "GRID",
}
CATFLOW_GDSS_GENERIC_ASSETS = {
    "EGO_SMALL": {
        "filename": "ego_small.pkl",
        "sha256": "eb1af8c3094929748a67b75c4f33ad364c1361d711c0fe14fa8c4d1681918dba",
        "graphs": 200,
    },
    "COMMUNITY_SMALL": {
        "filename": "community_small.pkl",
        "sha256": "ee0390de8ca9855e40b3a81cf0a84bfd0114a9a42ace369eb08a925e4d988157",
        "graphs": 100,
    },
    "ENZYMES": {
        "filename": "ENZYMES.pkl",
        "sha256": "d1fb37bd6322be1e7ee5e324672003b2b24b9b2e496d047a42e8f5a404cde110",
        "graphs": 587,
    },
    "GRID": {
        "filename": "grid.pkl",
        "sha256": "1c533398435301faaa1282993e1738069cafcc5e7662b284a3dba23cee680104",
        "graphs": 100,
    },
}
for _asset in CATFLOW_GDSS_GENERIC_ASSETS.values():
    _asset["url"] = (
        "https://raw.githubusercontent.com/harryjo97/GDSS/"
        f"{CATFLOW_GDSS_REVISION}/data/{_asset['filename']}"
    )

GDSS_QM9_REVISION = CATFLOW_GDSS_REVISION
GDSS_QM9_ASSETS = {
    "gdss_qm9.csv": {
        "sha256": "b7d055641f34c2803d0461806af46d263ebc6dec879bf0766657b1d3efd7ff8f",
        "url": f"https://raw.githubusercontent.com/harryjo97/GDSS/{GDSS_QM9_REVISION}/data/qm9.csv",
    },
    "valid_idx_qm9.json": {
        "sha256": "889fd232a33d8a2c00f057c807b119e72abd5baa48f6fab93e8d8b7a79340f4d",
        "url": f"https://raw.githubusercontent.com/harryjo97/GDSS/{GDSS_QM9_REVISION}/data/valid_idx_qm9.json",
    },
}
GDSS_ZINC250K_ASSETS = {
    "gdss_zinc250k.csv": {
        "sha256": "8dfc7f364bdd0c6f89d9dbaf9d1812f58d5f8b55dac9b970b26bbe597e5115f2",
        "url": (
            "https://raw.githubusercontent.com/harryjo97/GDSS/"
            f"{CATFLOW_GDSS_REVISION}/data/zinc250k.csv"
        ),
    },
    "valid_idx_zinc250k.json": {
        "sha256": "cf1af3a8588c947493ff0a2b60178db6726e9a4dbb445235b149cd3bbe6831df",
        "url": (
            "https://raw.githubusercontent.com/harryjo97/GDSS/"
            f"{CATFLOW_GDSS_REVISION}/data/valid_idx_zinc250k.json"
        ),
    },
}

# SHA-256 of compact-JSON source-index lists.  These fixtures make a PyTorch
# randperm change fail loudly instead of silently changing an "official" split.
OFFICIAL_SPLIT_INDEX_SHA256 = {
    ("digress_seed0", 100): {
        "train": "29a39200a98180eb9fbb61acb0b159e3d533a0bea10c084c8497fccd155aca8f",
        "val": "07581870706a9c7d585b8ec493aa12a47e286f02756f19f5f6d1b736186c6b56",
        "test": "c8b1e0b045f8127f0d3717ca249af97ca936695c0789af98c4ae11a01bdb08ae",
    },
    ("digress_seed0", 200): {
        "train": "9bce31573f173ff038abd35e256293318999465d5228dcaa8448c793801543c9",
        "val": "6ca485a9d8179ab90d52b1a0330f26ec30a273fd0ddd2c44b8490a9d04824da3",
        "test": "fcffa2dcf53928c1e886fe63239853e6605c750391e5bf9574826cd2eedb7d5e",
    },
    ("spectre_seed1234", 100): {
        "train": "e7faba34935d4b3e5c5195d6a1dfc2f185b10b6d31e62e995037ea799959c0a3",
        "val": "287feb400362bed6606dc5d2e39ae3f5eae3c56f71a5b5825eed50c02ff48792",
        "test": "48d4a11a6d107889a1a44903ae653d6fdd1212cab479b6a0a89a077a43eaf26c",
    },
    ("spectre_seed1234", 200): {
        "train": "ee28a68db26ecd4f1d6fd101cc4affe2b6d810001e8e837fa81ca9661a97aef9",
        "val": "51527c1a8f8e8b4c3002ca910e5b4fae05ee001d6e36b5d9369574923c383377",
        "test": "18b8e0d8d0b97237e4d9a8e83a52685058dc19964829def7ba9211b31a2df187",
    },
    ("spectre_seed1234", 918): {
        "train": "e45fffecd2fa39cf911919e24d0a30c20a48f41e4be47392244ae74964ab1132",
        "val": "396688feae71836e77063e0b3254533c0df37d5af03d2a1628dbcbb92ee4b66b",
        "test": "dd5b8a1a9d323f3192f4f012d4ceee5d4fe113bb7734aa9223906461ed6d8204",
    },
}

# Full-universe Torch seed-0 splits for the three standard abstract-graph
# collections that DiGress does not load itself.  These fixtures pin our
# explicitly derived metric/seed-compatible protocol without calling it an
# official DiGress split.
DIGRESS_STYLE_SPLIT_INDEX_SHA256 = {
    100: {
        "train": "674e9c605021a932437f2aa5775e177415e59012f9324cceacc5557f8cafdf95",
        "val": "8faeb1e86d4167f9de744a6ff0805a9c4ce146ef3863dfaa6fc5763355ff81e4",
        "test": "f56407d31e9b7eb927fb879d29cfa9b238a32ab4d8ee14bee8db7b17ae52a436",
    },
    200: {
        "train": "8291086ae81e4bb0727281a41090b45e212a37a0709d34a7271ca1e7006e9e45",
        "val": "5b0767ba1355183d1793f0ac67bbc220f361df8f10f5f4f3adc96aa452dc450a",
        "test": "21f7f39fe2bc051bed0d8d40036c2570e830a89c7a0758ec53844b26ec0b00d7",
    },
    587: {
        "train": "4266cd7b61280106032d10addc30748f8461043621aa0ab24908d51f5035daba",
        "val": "7ed2361a4fc3e8160d2e089fbf20e9851710b226aa5eca8de47d7497311222a3",
        "test": "ac0c6e8fbbccfd918279752de76a5ffb5c7103119e72ace32e6513e1cd3d8719",
    },
}

DIGRESS_STYLE_GENERIC_ASSETS = {
    "EGO_SMALL": {
        "filename": "ego_small_paper_graphs.pkl",
        "graphs": 200,
        "sha256": "6d80018492eba2c9479c7a5a81aa615b3cfb7cc5d539d247eb6c826ab417ad6c",
    },
    "ENZYMES": {
        "filename": "enzymes_paper_graphs.pkl",
        "graphs": 587,
        "sha256": "673b85239273d58df983093f875cde43a74c111062d9562c5e8dbf0a1ba54a01",
    },
    "GRID": {
        "filename": "grid_paper_graphs.pkl",
        "graphs": 100,
        "sha256": "06cb268663fe1476395ed2c68dbc57f4f11b0214c49c351b8dbcdb86803e59f7",
    },
}

SPECTRE_OFFICIAL_ASSETS = {
    "COMMUNITY_SMALL": {
        "filename": "community_12_21_100.pt",
        "sha256": "cf4407da67a29eb4fdf4c337b7b7ff30bc56e97a3849ff4e98b45ace0d599ee0",
        "url": f"https://raw.githubusercontent.com/KarolisMart/SPECTRE/{SPECTRE_SOURCE_REVISION}/data/community_12_21_100.pt",
    },
    "PLANAR": {
        "filename": "planar_64_200.pt",
        "sha256": "063dc3e675a5c63144e56aa974ca961abfcba02914368192a09b21a364df38fc",
        "url": f"https://raw.githubusercontent.com/KarolisMart/SPECTRE/{SPECTRE_SOURCE_REVISION}/data/planar_64_200.pt",
    },
    "SBM": {
        "filename": "sbm_200.pt",
        "sha256": "76e39c72aee626c2b642dfa8d5b47f07029f82391701b90b0378e04e1ac0760d",
        "url": f"https://raw.githubusercontent.com/KarolisMart/SPECTRE/{SPECTRE_SOURCE_REVISION}/data/sbm_200.pt",
    },
}

# Raw DD files committed in the public SPECTRE repository.  SPECTRE calls the
# filtered 100--500 node subset of this collection its ``protein`` benchmark.
SPECTRE_DD_RAW_ASSETS = {
    "DD_A.txt": {
        "sha256": "e538ec13a262b84a0664db2f22e6ff720fb5bb74965536c37f4b8e962d156e4a",
        "url": f"https://raw.githubusercontent.com/KarolisMart/SPECTRE/{SPECTRE_SOURCE_REVISION}/data/DD/DD_A.txt",
    },
    "DD_graph_indicator.txt": {
        "sha256": "95aabe75f6537326a14e4efc0d006854c20769f3cfc4877b862803d21c51aff2",
        "url": f"https://raw.githubusercontent.com/KarolisMart/SPECTRE/{SPECTRE_SOURCE_REVISION}/data/DD/DD_graph_indicator.txt",
    },
    "DD_node_labels.txt": {
        "sha256": "b0d10b05702c2d655d691b5a6f5585081993860013eeab801572202df1c5c108",
        "url": f"https://raw.githubusercontent.com/KarolisMart/SPECTRE/{SPECTRE_SOURCE_REVISION}/data/DD/DD_node_labels.txt",
    },
    "DD_graph_labels.txt": {
        "sha256": "b9df6495feb673f79dea357cca30e08f8602e52d978c77b53ec2552d1cb2ab4e",
        "url": f"https://raw.githubusercontent.com/KarolisMart/SPECTRE/{SPECTRE_SOURCE_REVISION}/data/DD/DD_graph_labels.txt",
    },
}

DATASET_ALIASES = {
    "COMMUNITY-SMALL": "COMMUNITY_SMALL",
    "COMMUNITYSMALL": "COMMUNITY_SMALL",
    "EGO-SMALL": "EGO_SMALL",
    "EGOSMALL": "EGO_SMALL",
    "ZINC250K": "ZINC250K",
    "ZINC-250K": "ZINC250K",
    "ZINC_250K": "ZINC250K",
    "ZINC250": "ZINC250K",
    "QM9-H": "QM9_H",
    "QM9_H": "QM9_H",
    "QM9_WITH_H": "QM9_H",
    "QM9_WITH_HYDROGEN": "QM9_H",
    "QM9_WITH_HYDROGENS": "QM9_H",
    "QM9_ORIGINAL": "QM9_H",
    "QM9_FULL": "QM9_H",
    "GUACAMOL": "GUACAMOL",
    "GUACA_MOL": "GUACAMOL",
    "CHEMBL": "GUACAMOL",
    "PROTEIN": "PROTEINS",
    "DD": "PROTEINS",
    "ENZYME": "ENZYMES",
    "COMMUNITY_SMALL": "COMMUNITY_SMALL",
    "EGO_SMALL": "EGO_SMALL",
}


@dataclass
class GraphDatasetBundle:
    train_graphs: list
    val_graphs: list
    test_graphs: list
    atom_decoder: list[str] | None
    train_smiles: list[str]
    mode: str
    data_root: Path
    provenance: dict = field(default_factory=dict)


def default_data_root(dataset: str) -> Path:
    dataset = normalize_dataset_name(dataset)
    if dataset == "PROTEINS":
        folder = "DD"
    elif dataset == "QM9_H":
        folder = "QM9"
    else:
        folder = dataset
    return DATASETS_ROOT / folder


def normalize_dataset_name(dataset: str) -> str:
    name = str(dataset).strip()
    key = name.upper().replace(" ", "_").replace("-", "_")
    compact_key = key.replace("_", "")
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    if compact_key in DATASET_ALIASES:
        return DATASET_ALIASES[compact_key]
    return key


def graph_to_simple_data(graph, *, atom_vocab: dict[str, int] | None = None):
    """Convert an RDKit molecule into the small graph data object used here."""
    from rdkit import Chem

    if graph is None:
        return None

    mol = graph
    labels = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if atom_vocab is None:
            label = atom.GetAtomicNum()
        else:
            if symbol not in atom_vocab:
                atom_vocab[symbol] = len(atom_vocab)
            label = atom_vocab[symbol]
        labels.append(int(label))

    edges = []
    edge_labels = []
    for bond in mol.GetBonds():
        u = int(bond.GetBeginAtomIdx())
        v = int(bond.GetEndAtomIdx())
        edge_type = _rdkit_bond_to_int(bond)
        edges.append((u, v))
        edges.append((v, u))
        edge_labels.append(edge_type)
        edge_labels.append(edge_type)

    x = torch.tensor(labels, dtype=torch.long)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_labels, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0,), dtype=torch.long)
    return SimpleGraphData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=None)


def _rdkit_atomic_number_vocabulary():
    """Return a subset-independent atom vocabulary for SMILES datasets."""
    from rdkit import Chem

    periodic_table = Chem.GetPeriodicTable()
    atom_decoder = [periodic_table.GetElementSymbol(atomic_number) for atomic_number in range(119)]
    atom_vocab = {symbol: atomic_number for atomic_number, symbol in enumerate(atom_decoder)}
    return atom_vocab, atom_decoder


def nx_to_simple_data(graph, *, node_label: int = 1):
    import networkx as nx

    graph = nx.convert_node_labels_to_integers(graph)
    num_nodes = int(graph.number_of_nodes())
    x = torch.full((num_nodes,), int(node_label), dtype=torch.long)
    edges = []
    edge_labels = []
    for u, v in graph.edges():
        u = int(u)
        v = int(v)
        if u == v:
            continue
        edges.append((u, v))
        edges.append((v, u))
        edge_labels.append(1)
        edge_labels.append(1)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_labels, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0,), dtype=torch.long)
    return SimpleGraphData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=None)


def load_graph_dataset_bundle(
    dataset: str,
    *,
    data_root: str | Path | None = None,
    zinc_subset: str | None = None,
    max_train_graphs: int | None = None,
    max_val_graphs: int | None = None,
    max_test_graphs: int | None = None,
    smiles_file: str | Path | None = None,
    moses_split: str = "train",
    seed: int = 0,
    paper_split_protocol: str = "legacy_local",
    qm9_protocol: str = "legacy_local",
    zinc250k_protocol: str = "legacy_pyg",
) -> GraphDatasetBundle:
    dataset = normalize_dataset_name(dataset)
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}; expected one of {SUPPORTED_DATASETS}")
    if paper_split_protocol not in PAPER_SPLIT_PROTOCOLS:
        raise ValueError(
            f"Unsupported paper split protocol {paper_split_protocol!r}; "
            f"expected one of {PAPER_SPLIT_PROTOCOLS}"
        )
    if qm9_protocol not in QM9_PROTOCOLS:
        raise ValueError(
            f"Unsupported QM9 protocol {qm9_protocol!r}; expected one of {QM9_PROTOCOLS}"
        )
    if zinc250k_protocol not in ZINC250K_PROTOCOLS:
        raise ValueError(
            f"Unsupported ZINC250K protocol {zinc250k_protocol!r}; "
            f"expected one of {ZINC250K_PROTOCOLS}"
        )
    root = Path(data_root) if data_root is not None else default_data_root(dataset)

    if dataset in {"ZINC", "ZINC250K"}:
        if dataset == "ZINC250K" and zinc250k_protocol == "gdss_catflow":
            return _load_zinc250k_gdss_catflow_bundle(
                root,
                max_train_graphs=max_train_graphs,
                max_val_graphs=max_val_graphs,
                max_test_graphs=max_test_graphs,
            )
        subset = zinc_subset or ("large" if dataset == "ZINC250K" else "small")
        train_graphs = load_processed_zinc_split("train", subset=subset, root=root, max_graphs=max_train_graphs)
        val_graphs = load_processed_zinc_split("val", subset=subset, root=root, max_graphs=max_val_graphs)
        test_graphs = load_processed_zinc_split("test", subset=subset, root=root, max_graphs=max_test_graphs)
        return GraphDatasetBundle(
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            test_graphs=test_graphs,
            atom_decoder=load_atom_decoder(root=root),
            train_smiles=[],
            mode="molecule",
            data_root=root,
        )

    if dataset in TU_DATASETS:
        tu_root = root.parent if root.name.upper() == dataset else root
        if (tu_root / dataset / f"{dataset}_A.txt").exists():
            graphs = load_tu_graph_dataset(tu_root, dataset)
        else:
            configure_warning_filters()
            try:
                from torch_geometric.datasets import TUDataset
            except Exception as exc:
                raise RuntimeError(
                    f"{dataset} requires local TU files or torch_geometric.datasets.TUDataset."
                ) from exc
            pyg_dataset = TUDataset(root=str(root), name=dataset, use_node_attr=False, use_edge_attr=False)
            graphs = [
                _pyg_data_to_simple_data(data, preserve_edge_labels=True)
                for data in pyg_dataset
            ]
        train_graphs, val_graphs, test_graphs = split_graph_dataset(graphs, seed=seed)
        return GraphDatasetBundle(
            train_graphs=_cap(train_graphs, max_train_graphs),
            val_graphs=_cap(val_graphs, max_val_graphs),
            test_graphs=_cap(test_graphs, max_test_graphs),
            atom_decoder=None,
            train_smiles=[],
            mode="generic",
            data_root=root,
        )

    if dataset in PAPER_GENERIC_DATASETS:
        provenance = {
            "dataset": dataset,
            "paper_split_protocol": paper_split_protocol,
        }
        if paper_split_protocol == "catflow_gdss":
            if dataset not in CATFLOW_GDSS_GENERIC_DATASETS:
                raise ValueError(
                    "The CatFlow/GDSS ordered split is defined only for "
                    "EGO_SMALL, COMMUNITY_SMALL, ENZYMES, and GRID. "
                    f"Received {dataset}."
                )
            graphs, asset_provenance = _load_catflow_gdss_graphs(dataset, root=root)
            train_graphs, val_graphs, test_graphs, split_provenance = _split_catflow_gdss(
                graphs
            )
            provenance.update(asset_provenance)
            provenance.update(split_provenance)
        elif dataset in SPECTRE_OFFICIAL_ASSETS and paper_split_protocol != "legacy_local":
            graphs, asset_provenance = _load_official_spectre_graphs(dataset, root=root)
            provenance.update(asset_provenance)
            train_graphs, val_graphs, test_graphs, split_provenance = _split_official_paper_graphs(
                graphs,
                protocol=paper_split_protocol,
            )
            provenance.update(split_provenance)
        elif dataset in DIGRESS_STYLE_GENERIC_DATASETS and paper_split_protocol != "legacy_local":
            graphs = _load_paper_generic_graphs(dataset, root=root)
            split_seed = 0 if paper_split_protocol == "digress_seed0" else 1234
            train_graphs, val_graphs, test_graphs = _split_full_universe_torch_seed(
                graphs,
                seed=split_seed,
            )
            provenance.update(_digress_style_generic_provenance(
                dataset,
                len(graphs),
                protocol=paper_split_protocol,
                seed=split_seed,
            ))
            provenance.update(_generic_collection_asset_provenance(dataset, root, len(graphs)))
        elif dataset == "PROTEINS" and paper_split_protocol == "spectre_seed1234":
            graphs, asset_provenance = _load_official_spectre_proteins_graphs(root=root)
            train_graphs, val_graphs, test_graphs = _split_spectre_seed1234(graphs)
            provenance.update(asset_provenance)
            provenance.update(_official_split_provenance(len(graphs), "spectre_seed1234"))
        elif dataset == "PROTEINS" and paper_split_protocol == "digress_seed0":
            raise ValueError(
                "DiGress does not define a PROTEINS loader/split. Use "
                "--paper-split-protocol spectre_seed1234 for the SPECTRE PROTEINS benchmark."
            )
        elif paper_split_protocol != "legacy_local":
            raise ValueError(
                f"No official {paper_split_protocol} loader is implemented for {dataset}. "
                "Use legacy_local explicitly for repository-local/generated data."
            )
        else:
            graphs = _load_paper_generic_graphs(dataset, root=root)
            train_graphs, val_graphs, test_graphs = _split_paper_graphs(graphs)
            provenance.update({
                "source": "repository_local",
                "split_seed": PAPER_DATASET_SEED,
                "split_rng": "python_random",
            })
        provenance["split_sizes"] = {
            "train": len(train_graphs),
            "val": len(val_graphs),
            "test": len(test_graphs),
        }
        return GraphDatasetBundle(
            train_graphs=_cap(train_graphs, max_train_graphs),
            val_graphs=_cap(val_graphs, max_val_graphs),
            test_graphs=_cap(test_graphs, max_test_graphs),
            atom_decoder=None,
            train_smiles=[],
            mode="generic",
            data_root=root,
            provenance=provenance,
        )

    if dataset in {"QM9", "QM9_H"}:
        if dataset == "QM9_H" and qm9_protocol != "legacy_local":
            raise ValueError(
                "QM9_H is the project-specific explicit-hydrogen dataset and cannot use the "
                "heavy-atom GDSS/CatFlow protocol. Use --qm9-protocol legacy_local."
            )
        return _load_qm9_bundle(
            root=root,
            max_train_graphs=max_train_graphs,
            max_val_graphs=max_val_graphs,
            max_test_graphs=max_test_graphs,
            seed=seed,
            include_hydrogens=(dataset == "QM9_H"),
            protocol=qm9_protocol,
        )

    if dataset == "GUACAMOL":
        return _load_moses_bundle(
            root=root,
            smiles_file=smiles_file,
            moses_split=moses_split,
            max_train_graphs=max_train_graphs,
            max_val_graphs=max_val_graphs,
            max_test_graphs=max_test_graphs,
            seed=seed,
            dataset_label="GuacaMol/ChEMBL",
            require_smiles_file=True,
        )

    return _load_moses_bundle(
        root=root,
        smiles_file=smiles_file,
        moses_split=moses_split,
        max_train_graphs=max_train_graphs,
        max_val_graphs=max_val_graphs,
        max_test_graphs=max_test_graphs,
        seed=seed,
        dataset_label="MOSES",
        require_smiles_file=False,
    )


def _cap(items: list, max_items: int | None):
    if max_items is None:
        return list(items)
    return list(items)[: max(0, int(max_items))]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_official_spectre_graphs(dataset: str, *, root: Path):
    """Load and verify the serialized graph assets published by SPECTRE."""
    asset = SPECTRE_OFFICIAL_ASSETS[dataset]
    path = Path(root) / asset["filename"]
    if not path.is_file():
        raise FileNotFoundError(
            f"Official SPECTRE asset is missing: {path}. Run "
            f"`python prepare_official_spectre_data.py --output-dir {Path(root)}` first."
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != asset["sha256"]:
        raise RuntimeError(
            f"Official SPECTRE asset checksum mismatch for {path}: "
            f"received {actual_sha256}, expected {asset['sha256']}"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, (list, tuple)) or not payload:
        raise RuntimeError(f"Unexpected SPECTRE payload in {path}")
    adjs = payload[0]
    graphs = []
    for adj in adjs:
        matrix = torch.as_tensor(adj).detach().cpu()
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise RuntimeError(f"Invalid adjacency shape {tuple(matrix.shape)} in {path}")
        graph = nx.from_numpy_array(matrix.numpy())
        graph.remove_edges_from(list(nx.selfloop_edges(graph)))
        graphs.append(nx_to_simple_data(graph))
    expected_count = 100 if dataset == "COMMUNITY_SMALL" else 200
    if len(graphs) != expected_count:
        raise RuntimeError(
            f"Official SPECTRE {dataset} asset has {len(graphs)} graphs; expected {expected_count}"
        )
    return graphs, {
        "source": "official_spectre_asset",
        "asset_file": asset["filename"],
        "asset_sha256": actual_sha256,
        "asset_url": asset["url"],
        "asset_graphs": len(graphs),
    }


def _find_official_dd_dir(root: Path) -> Path:
    candidates = (Path(root) / "DD", Path(root) / "DD" / "raw", Path(root))
    for candidate in candidates:
        if all((candidate / filename).is_file() for filename in SPECTRE_DD_RAW_ASSETS):
            return candidate
    expected = Path(root) / "DD"
    raise FileNotFoundError(
        f"Official SPECTRE DD files are missing under {root}. Run "
        f"`python prepare_official_spectre_data.py --output-dir {Path(root)}` first "
        f"(expected files in {expected})."
    )


def _load_official_spectre_proteins_graphs(*, root: Path):
    """Reproduce SPECTRE's ProteinDataset topology preprocessing exactly."""
    import numpy as np

    dd_dir = _find_official_dd_dir(Path(root))
    verified_hashes = {}
    for filename, spec in SPECTRE_DD_RAW_ASSETS.items():
        path = dd_dir / filename
        actual = _sha256_file(path)
        if actual != spec["sha256"]:
            raise RuntimeError(
                f"Official SPECTRE DD checksum mismatch for {path}: "
                f"received {actual}, expected {spec['sha256']}"
            )
        verified_hashes[filename] = actual

    data_adj = np.loadtxt(dd_dir / "DD_A.txt", delimiter=",", dtype=np.int64)
    graph_indicator = np.loadtxt(
        dd_dir / "DD_graph_indicator.txt", delimiter=",", dtype=np.int64
    )
    graph_labels = np.loadtxt(
        dd_dir / "DD_graph_labels.txt", delimiter=",", dtype=np.int64
    )
    graph = nx.Graph()
    graph.add_edges_from(map(tuple, data_adj.tolist()))
    graph.remove_nodes_from(list(nx.isolates(graph)))
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))

    node_ids = np.arange(graph_indicator.shape[0], dtype=np.int64) + 1
    graphs = []
    for graph_idx in range(int(graph_indicator.max())):
        selected_nodes = node_ids[graph_indicator == graph_idx + 1].tolist()
        subgraph = graph.subgraph(selected_nodes).copy()
        subgraph.graph["label"] = int(graph_labels[graph_idx])
        if 100 <= subgraph.number_of_nodes() <= 500:
            graphs.append(nx_to_simple_data(subgraph))
    if len(graphs) != 918:
        raise RuntimeError(
            f"Official SPECTRE DD preprocessing produced {len(graphs)} graphs; expected 918"
        )
    return graphs, {
        "source": "official_spectre_dd_filtered_100_500",
        "asset_dir": portable_path(dd_dir),
        "asset_graphs": len(graphs),
        "asset_sha256": verified_hashes,
        "asset_urls": {
            filename: spec["url"] for filename, spec in SPECTRE_DD_RAW_ASSETS.items()
        },
        "preprocessing": "SPECTRE ProteinDataset: remove isolates/self-loops; keep 100<=nodes<=500",
    }


def _split_sizes(n: int) -> tuple[int, int, int]:
    test_len = int(round(int(n) * 0.2))
    train_len = int(round((int(n) - test_len) * 0.8))
    val_len = int(n) - train_len - test_len
    return train_len, val_len, test_len


def _split_spectre_seed1234(items: list):
    """Exact SPECTRE README/data.py split: random_split with torch seed 1234."""
    items = list(items)
    train_len, val_len, test_len = _split_sizes(len(items))
    order = torch.randperm(len(items), generator=torch.Generator().manual_seed(1234)).tolist()
    return (
        [items[i] for i in order[:train_len]],
        [items[i] for i in order[train_len:train_len + val_len]],
        [items[i] for i in order[train_len + val_len:]],
    )


def _split_full_universe_torch_seed(items: list, *, seed: int):
    """Reproducible 64/16/20 split over the complete supplied collection.

    This is used for the standard Ego-small, Enzymes, and Grid collections when
    a DiGress-style seed protocol is requested.  DiGress does not publish
    loaders for these datasets, so unlike ``_split_digress_seed0`` this helper
    never applies the SPECTRE loader's hard-coded 200-item membership universe.
    """
    items = list(items)
    train_len, val_len, test_len = _split_sizes(len(items))
    order = torch.randperm(
        len(items),
        generator=torch.Generator().manual_seed(int(seed)),
    ).tolist()
    if train_len + val_len + test_len != len(items):
        raise RuntimeError("Full-universe split sizes do not cover the dataset")
    return (
        [items[i] for i in order[:train_len]],
        [items[i] for i in order[train_len:train_len + val_len]],
        [items[i] for i in order[train_len + val_len:]],
    )


def _digress_style_generic_provenance(
    dataset: str,
    n_items: int,
    *,
    protocol: str,
    seed: int,
) -> dict:
    """Record a fail-closed, explicitly non-official DiGress-style split."""
    dataset = normalize_dataset_name(dataset)
    if dataset not in DIGRESS_STYLE_GENERIC_DATASETS:
        raise ValueError(f"No DiGress-style generic provenance for {dataset}")
    indices = list(range(int(n_items)))
    train, val, test = _split_full_universe_torch_seed(indices, seed=int(seed))
    split_hashes = {
        "train": _index_list_sha256(train),
        "val": _index_list_sha256(val),
        "test": _index_list_sha256(test),
    }
    expected_hashes = DIGRESS_STYLE_SPLIT_INDEX_SHA256.get(int(n_items)) if int(seed) == 0 else None
    if expected_hashes is not None and split_hashes != expected_hashes:
        raise RuntimeError(
            "DiGress-style full-universe seed-0 split membership changed: "
            f"actual={split_hashes}, expected={expected_hashes}"
        )
    source_details = {
        "EGO_SMALL": "PyG CiteSeer radius-1 ego graphs with 4--18 nodes, first 200",
        "ENZYMES": "TU ENZYMES topology graphs with 10--1000 nodes",
        "GRID": "10x10 through 19x19 two-dimensional NetworkX grids",
    }
    return {
        "source": "standard_abstract_graph_collection",
        "source_detail": source_details[dataset],
        "asset_graphs": int(n_items),
        "split_seed": int(seed),
        "split_rng": "torch_randperm_full_dataset",
        "split_scope": "digress_style_full_universe_nonofficial_dataset",
        "split_universe": int(n_items),
        "split_index_sha256": split_hashes,
        "torch_version": str(torch.__version__),
        "official_digress_split": False,
        "comparability_scope": "digress_metric_and_seed_style_local_denominator",
        "protocol_note": (
            f"{protocol} selects Torch seed {int(seed)} over the full dataset; "
            "DiGress does not publish an official split for this dataset"
        ),
    }


def _index_list_sha256(indices: list[int]) -> str:
    payload = json.dumps([int(index) for index in indices], separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _official_split_indices(n_items: int, protocol: str) -> dict[str, list[int]]:
    n_items = int(n_items)
    if protocol == "digress_seed0":
        split_universe = 200
        train_len, val_len, _test_len = _split_sizes(split_universe)
        order = torch.randperm(
            split_universe,
            generator=torch.Generator().manual_seed(0),
        ).tolist()
        memberships = {
            "train": set(order[:train_len]),
            "val": set(order[train_len:train_len + val_len]),
            "test": set(order[train_len + val_len:]),
        }
        return {
            split: [index for index in range(n_items) if index in members]
            for split, members in memberships.items()
        }
    if protocol == "spectre_seed1234":
        train_len, val_len, _test_len = _split_sizes(n_items)
        order = torch.randperm(
            n_items,
            generator=torch.Generator().manual_seed(1234),
        ).tolist()
        return {
            "train": order[:train_len],
            "val": order[train_len:train_len + val_len],
            "test": order[train_len + val_len:],
        }
    raise ValueError(protocol)


def _official_split_provenance(n_items: int, protocol: str) -> dict:
    split_indices = _official_split_indices(n_items, protocol)
    actual_hashes = {
        split: _index_list_sha256(indices)
        for split, indices in split_indices.items()
    }
    expected_hashes = OFFICIAL_SPLIT_INDEX_SHA256.get((protocol, int(n_items)))
    if expected_hashes is None:
        raise RuntimeError(
            f"No pinned official split-index fixture for protocol={protocol} n={n_items}"
        )
    if actual_hashes != expected_hashes:
        raise RuntimeError(
            "Official split membership changed in this PyTorch runtime: "
            f"actual={actual_hashes}, expected={expected_hashes}"
        )
    if protocol == "digress_seed0":
        source_revision = DIGRESS_MAIN_SPLIT_REVISION
        source_url = (
            "https://github.com/cvignac/DiGress/blob/"
            f"{source_revision}/src/datasets/spectre_dataset.py"
        )
        split_seed = 0
        split_rng = "torch_randperm_200_then_source_membership"
        split_scope = "digress_current_main_loader_membership"
        split_universe = 200
    else:
        source_revision = DIGRESS_PAPER_SPLIT_REVISION
        source_url = (
            "https://github.com/cvignac/DiGress/blob/"
            f"{source_revision}/src/datasets/spectre_dataset.py"
        )
        split_seed = 1234
        split_rng = "torch_random_split_order"
        split_scope = "spectre_and_digress_paper_era"
        split_universe = int(n_items)
    return {
        "split_seed": split_seed,
        "split_rng": split_rng,
        "split_scope": split_scope,
        "split_universe": split_universe,
        "split_protocol_revision": source_revision,
        "split_protocol_url": source_url,
        "split_index_sha256": actual_hashes,
        "torch_version": str(torch.__version__),
    }


def _split_digress_seed0(items: list):
    """Exact DiGress loader split, including its source-order membership routing.

    DiGress hard-codes a 200-graph split universe for these datasets.  The official
    Community asset contains 100 graphs, so faithfully reproducing the public
    loader yields a 68/15/17 split instead of SPECTRE's 64/16/20 split.
    """
    items = list(items)
    split_indices = _official_split_indices(len(items), "digress_seed0")
    train = [items[idx] for idx in split_indices["train"]]
    val = [items[idx] for idx in split_indices["val"]]
    test = [items[idx] for idx in split_indices["test"]]
    if len(train) + len(val) + len(test) != len(items):
        raise RuntimeError("DiGress split did not assign every official SPECTRE graph")
    return train, val, test


def _split_official_paper_graphs(items: list, *, protocol: str):
    if protocol == "digress_seed0":
        return (*_split_digress_seed0(items), _official_split_provenance(len(items), protocol))
    if protocol == "spectre_seed1234":
        return (*_split_spectre_seed1234(items), _official_split_provenance(len(items), protocol))
    raise ValueError(f"Official assets require an explicit paper split protocol, received {protocol!r}")


def _split_catflow_gdss(items: list):
    """Reproduce the ordered GDSS split cited by CatFlow.

    GDSS evaluates the first ``int(0.2 * N)`` graphs and trains on every
    remaining graph.  It has no validation split.  The Ring VAE requires
    validation traces, so ``val`` is a deterministic *view* of the official
    training collection and is not removed from ``train``.  This preserves the
    published outer train/test membership and keeps the test set untouched.
    """
    items = list(items)
    if len(items) < 2:
        raise ValueError("CatFlow/GDSS splitting requires at least two graphs")
    test_size = int(0.2 * len(items))
    if test_size < 1 or test_size >= len(items):
        raise ValueError(
            f"Invalid CatFlow/GDSS test size {test_size} for {len(items)} graphs"
        )
    test = items[:test_size]
    train = items[test_size:]
    validation_size = max(1, int(round(0.2 * len(train))))
    val = train[:validation_size]
    provenance = {
        "source_revision": CATFLOW_GDSS_REVISION,
        "split_rng": "none_ordered_collection",
        "split_seed": None,
        "outer_split_policy": "gdss_first_20_percent_test_remaining_80_percent_train",
        "official_train_graphs": len(train),
        "official_test_graphs": len(test),
        "internal_validation_policy": "deterministic_training_subset_view_not_held_out",
        "internal_validation_graphs": len(val),
        "comparison_scope": "CatFlow main abstract-graph table",
    }
    return train, val, test, provenance


def _split_paper_graphs(items: list):
    """SPECTRE/GraphRNN-style split: 20% test, then 20% of remaining train as val."""
    items = list(items)
    rng = random.Random(PAPER_DATASET_SEED)
    order = list(range(len(items)))
    rng.shuffle(order)
    n = len(order)
    if n == 0:
        return [], [], []
    test_len = max(1, int(round(0.2 * n))) if n > 2 else 0
    remaining = max(0, n - test_len)
    val_len = max(1, int(round(0.2 * remaining))) if remaining > 2 else 0
    if test_len + val_len >= n:
        val_len = max(0, n - test_len - 1)
    train_len = max(0, n - test_len - val_len)
    train = [items[i] for i in order[:train_len]]
    val = [items[i] for i in order[train_len:train_len + val_len]]
    test = [items[i] for i in order[train_len + val_len:]]
    return train, val, test


def _split_qm9_paper(items: list, *, seed: int, max_train=None, max_val=None, max_test=None):
    items = list(items)
    rng = random.Random(seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    n = len(order)
    if n >= 20000:
        val_len = 10000
        test_len = 10000
        train_len = n - val_len - test_len
    else:
        test_len = max(1, int(round(0.2 * n))) if n > 2 else 0
        val_len = max(1, int(round(0.1 * n))) if n > 2 else 0
        if test_len + val_len >= n:
            val_len = max(0, n - test_len - 1)
        train_len = max(0, n - val_len - test_len)
    train = [items[i] for i in order[:train_len]]
    val = [items[i] for i in order[train_len:train_len + val_len]]
    test = [items[i] for i in order[train_len + val_len:]]
    return _cap(train, max_train), _cap(val, max_val), _cap(test, max_test)


def _split_items(items: list, *, seed: int, max_train=None, max_val=None, max_test=None):
    rng = random.Random(seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    n = len(order)
    n_train = max(1, int(round(0.8 * n))) if n else 0
    n_val = max(1, int(round(0.1 * n))) if n > 2 else 0
    if n_train + n_val >= n:
        n_val = max(0, n - n_train)
    train = [items[i] for i in order[:n_train]]
    val = [items[i] for i in order[n_train:n_train + n_val]]
    test = [items[i] for i in order[n_train + n_val:]]
    return _cap(train, max_train), _cap(val, max_val), _cap(test, max_test)


def _precap_smiles_for_split(smiles: list[str], *, seed: int, max_train=None, max_val=None, max_test=None):
    required = []
    if max_train is not None:
        required.append(int(max_train / 0.8) + 1)
    if max_val is not None:
        required.append(int(max_val / 0.1) + 1)
    if max_test is not None:
        required.append(int(max_test / 0.1) + 1)
    if not required:
        return smiles

    target = max(required)
    target = max(target + 1000, int(target * 1.25))
    if target >= len(smiles):
        return smiles

    rng = random.Random(seed)
    order = list(range(len(smiles)))
    rng.shuffle(order)
    selected = sorted(order[:target])
    return [smiles[i] for i in selected]


def _load_paper_generic_graphs(dataset: str, *, root: Path):
    root = Path(root)
    cached = _load_cached_paper_graphs(root, dataset)
    if cached:
        return cached

    if dataset == "COMMUNITY_SMALL":
        graphs = _generate_community_small()
    elif dataset == "GRID":
        graphs = _generate_grid_graphs()
    elif dataset == "PLANAR":
        graphs = _generate_planar_graphs()
    elif dataset == "SBM":
        graphs = _generate_sbm_graphs()
    elif dataset == "EGO_SMALL":
        graphs = _load_ego_small_from_citeseer(root)
    elif dataset == "ENZYMES":
        graphs = _load_tu_or_pyg_topology_graphs(root, "ENZYMES", min_nodes=10, max_nodes=1000)
    elif dataset == "PROTEINS":
        graphs = _load_tu_or_pyg_topology_graphs(root, "DD", min_nodes=100, max_nodes=500)
    else:
        raise ValueError(dataset)

    root.mkdir(parents=True, exist_ok=True)
    cache_path = root / f"{dataset.lower()}_paper_graphs.pkl"
    try:
        with open(cache_path, "wb") as handle:
            pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return graphs


def _paper_generic_cache_candidates(root: Path, dataset: str) -> list[Path]:
    candidates = [
        root / f"{dataset.lower()}_paper_graphs.pkl",
        root / f"{dataset.lower()}.pkl",
    ]
    if dataset == "COMMUNITY_SMALL":
        candidates.extend([root / "community_small.pkl", root / "community_12_21_100.pkl"])
    elif dataset == "EGO_SMALL":
        candidates.append(root / "ego_small.pkl")
    elif dataset == "GRID":
        candidates.append(root / "grid.pkl")
    elif dataset == "PLANAR":
        candidates.extend([root / "planar.pkl", root / "planar_64_200.pkl"])
    elif dataset == "SBM":
        candidates.extend([root / "sbm.pkl", root / "sbm_200.pkl"])
    elif dataset == "ENZYMES":
        candidates.append(root / "ENZYMES.pkl")
    elif dataset == "PROTEINS":
        candidates.extend([root / "proteins.pkl", root / "DD.pkl"])
    return candidates


def _generic_collection_asset_provenance(dataset: str, root: Path, n_items: int) -> dict:
    spec = DIGRESS_STYLE_GENERIC_ASSETS[normalize_dataset_name(dataset)]
    path = Path(root) / spec["filename"]
    if not path.is_file():
        raise RuntimeError(f"Pinned DiGress-style source asset is missing: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != spec["sha256"]:
        raise RuntimeError(
            f"Pinned DiGress-style asset checksum mismatch for {path}: "
            f"received {actual_sha256}, expected {spec['sha256']}"
        )
    if int(n_items) != int(spec["graphs"]):
        raise RuntimeError(
            f"Pinned DiGress-style asset {path} loaded {n_items} graphs; "
            f"expected {spec['graphs']}"
        )
    return {
        "asset_file": path.name,
        "asset_sha256": actual_sha256,
        "asset_graphs": int(n_items),
    }


def _load_cached_paper_graphs(root: Path, dataset: str):
    candidates = _paper_generic_cache_candidates(root, dataset)

    for path in candidates:
        if path.exists():
            with open(path, "rb") as handle:
                obj = pickle.load(handle)
            return [_coerce_graph_to_simple_data(item) for item in obj]
    return []


def _load_catflow_gdss_graphs(dataset: str, *, root: Path):
    dataset = normalize_dataset_name(dataset)
    spec = CATFLOW_GDSS_GENERIC_ASSETS[dataset]
    path = Path(root) / spec["filename"]
    if not path.is_file():
        raise FileNotFoundError(
            f"Pinned CatFlow/GDSS asset is missing: {path}. "
            "Run `python prepare_official_catflow_data.py` first."
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != spec["sha256"]:
        raise RuntimeError(
            f"Pinned CatFlow/GDSS asset checksum mismatch for {path}: "
            f"received {actual_sha256}, expected {spec['sha256']}"
        )
    with path.open("rb") as handle:
        raw_graphs = pickle.load(handle)
    if len(raw_graphs) != int(spec["graphs"]):
        raise RuntimeError(
            f"Pinned CatFlow/GDSS asset {path} contains {len(raw_graphs)} graphs; "
            f"expected {spec['graphs']}"
        )
    graphs = [_coerce_graph_to_simple_data(graph) for graph in raw_graphs]
    provenance = {
        "source": "official_gdss_asset",
        "source_revision": CATFLOW_GDSS_REVISION,
        "source_url": spec["url"],
        "asset_file": spec["filename"],
        "asset_sha256": actual_sha256,
        "asset_graphs": len(graphs),
    }
    return graphs, provenance


def _coerce_graph_to_simple_data(item):
    if isinstance(item, SimpleGraphData):
        return item
    if hasattr(item, "edge_index"):
        return _pyg_data_to_simple_data(item, constant_node_label=True)
    return nx_to_simple_data(item)


def _generate_community_small():
    rng = random.Random(PAPER_DATASET_SEED)
    graphs = []
    for _ in range(100):
        max_nodes = rng.choice(list(range(12, 21)))
        graphs.append(nx_to_simple_data(_n_community(2, max_nodes, rng=rng)))
    return graphs


def _n_community(num_communities: int, max_nodes: int, *, rng: random.Random, p_inter: float = 0.05):
    import networkx as nx

    one_community_size = int(max_nodes) // int(num_communities)
    sizes = [one_community_size] * int(num_communities)
    p_bridge = float(p_inter) * 2.0 / max((int(num_communities) - 1) * one_community_size, 1)
    parts = [
        nx.gnp_random_graph(size, 0.7, seed=rng.randrange(2**32))
        for size in sizes
    ]
    graph = nx.disjoint_union_all(parts)
    components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
    for i, sub_a in enumerate(components):
        nodes_a = list(sub_a.nodes())
        for sub_b in components[i + 1:]:
            nodes_b = list(sub_b.nodes())
            has_bridge = False
            for node_a in nodes_a:
                for node_b in nodes_b:
                    if rng.random() < p_bridge:
                        graph.add_edge(node_a, node_b)
                        has_bridge = True
            if not has_bridge and nodes_a and nodes_b:
                graph.add_edge(nodes_a[0], nodes_b[0])
    graph.remove_edges_from(list(nx.selfloop_edges(graph)))
    return nx.convert_node_labels_to_integers(graph)


def _generate_grid_graphs():
    import networkx as nx

    return [
        nx_to_simple_data(nx.grid_2d_graph(m, n))
        for m in range(10, 20)
        for n in range(10, 20)
    ]


def _generate_planar_graphs():
    import networkx as nx
    import numpy as np
    from scipy.spatial import Delaunay

    rng = np.random.RandomState(PAPER_DATASET_SEED)
    graphs = []
    for _ in range(200):
        points = rng.rand(64, 2)
        tri = Delaunay(points)
        graph = nx.Graph()
        graph.add_nodes_from(range(64))
        for simplex in tri.simplices:
            a, b, c = [int(v) for v in simplex]
            graph.add_edge(a, b)
            graph.add_edge(b, c)
            graph.add_edge(c, a)
        graphs.append(nx_to_simple_data(graph))
    return graphs


def _generate_sbm_graphs():
    import networkx as nx
    import numpy as np

    rng = np.random.RandomState(PAPER_DATASET_SEED)
    graphs = []
    for seed in range(200):
        n_communities = int(rng.randint(2, 6))
        community_sizes = rng.randint(20, 41, size=n_communities).tolist()
        probs = np.full((n_communities, n_communities), 0.005, dtype=float)
        np.fill_diagonal(probs, 0.3)
        graph = nx.stochastic_block_model(community_sizes, probs, seed=seed)
        graph.remove_edges_from(list(nx.selfloop_edges(graph)))
        graphs.append(nx_to_simple_data(graph))
    return graphs


def _load_ego_small_from_citeseer(root: Path):
    import networkx as nx

    configure_warning_filters()
    try:
        from torch_geometric.datasets import Planetoid
    except Exception as exc:
        raise RuntimeError("EGO_SMALL support requires torch_geometric.datasets.Planetoid.") from exc

    dataset = Planetoid(root=str(root / "citeseer"), name="CiteSeer")
    data = dataset[0]
    graph = nx.Graph()
    num_nodes = int(getattr(data, "num_nodes", 0) or data.x.size(0))
    graph.add_nodes_from(range(num_nodes))
    edge_index = data.edge_index.detach().cpu()
    for pos in range(edge_index.size(1)):
        u = int(edge_index[0, pos].item())
        v = int(edge_index[1, pos].item())
        if u != v:
            graph.add_edge(u, v)
    largest = max((graph.subgraph(c).copy() for c in nx.connected_components(graph)), key=lambda g: g.number_of_nodes())
    largest = nx.convert_node_labels_to_integers(largest)
    ego_graphs = []
    for node in range(largest.number_of_nodes()):
        ego = nx.ego_graph(largest, node, radius=1)
        ego.remove_edges_from(list(nx.selfloop_edges(ego)))
        if 4 <= ego.number_of_nodes() <= 18:
            ego_graphs.append(nx_to_simple_data(ego))
        if len(ego_graphs) >= 200:
            break
    if len(ego_graphs) < 1:
        raise RuntimeError("CiteSeer did not produce any EGO_SMALL graphs with 4-18 nodes.")
    return ego_graphs


def _load_tu_or_pyg_topology_graphs(root: Path, name: str, *, min_nodes: int, max_nodes: int):
    root = Path(root)
    tu_root = root.parent if root.name.upper() == name.upper() else root
    if (tu_root / name / f"{name}_A.txt").exists():
        graphs = load_tu_graph_dataset(tu_root, name)
        return [
            SimpleGraphData(
                x=torch.ones_like(graph.x, dtype=torch.long),
                edge_index=graph.edge_index.clone(),
                edge_attr=torch.ones_like(graph.edge_attr, dtype=torch.long),
                y=graph.y,
            )
            for graph in graphs
            if int(graph.num_nodes) >= int(min_nodes) and int(graph.num_nodes) <= int(max_nodes)
        ]

    configure_warning_filters()
    try:
        from torch_geometric.datasets import TUDataset
    except Exception as exc:
        raise RuntimeError(f"{name} support requires torch_geometric.datasets.TUDataset or local TU files.") from exc

    pyg_dataset = TUDataset(root=str(root), name=name, use_node_attr=False)
    graphs = [
        _pyg_data_to_simple_data(data, constant_node_label=True)
        for data in pyg_dataset
        if int(getattr(data, "num_nodes", 0) or data.x.size(0)) >= int(min_nodes)
        and int(getattr(data, "num_nodes", 0) or data.x.size(0)) <= int(max_nodes)
    ]
    if not graphs:
        raise RuntimeError(f"No {name} graphs found after filtering to {min_nodes}-{max_nodes} nodes.")
    return graphs


def _pyg_data_to_simple_data(
    data,
    *,
    constant_node_label: bool = False,
    preserve_edge_labels: bool = False,
):
    num_nodes = int(getattr(data, "num_nodes", 0) or (data.x.size(0) if getattr(data, "x", None) is not None else 0))
    if constant_node_label or getattr(data, "x", None) is None:
        x = torch.ones(num_nodes, dtype=torch.long)
    else:
        values = data.x.detach().cpu()
        if values.dim() == 1:
            x = values.to(torch.long)
            if x.numel() > 0 and int(x.min().item()) <= 0:
                x = x - int(x.min().item()) + 1
        else:
            x = values.argmax(dim=-1).to(torch.long) + 1
    edge_index = data.edge_index.detach().cpu().to(torch.long).clone()
    raw_edge_attr = getattr(data, "edge_attr", None)
    if preserve_edge_labels and raw_edge_attr is not None and raw_edge_attr.numel() > 0:
        edge_values = raw_edge_attr.detach().cpu()
        if edge_values.dim() == 1:
            edge_attr = edge_values.to(torch.long)
            if edge_attr.numel() > 0 and int(edge_attr.min().item()) <= 0:
                edge_attr = edge_attr - int(edge_attr.min().item()) + 1
        else:
            edge_attr = edge_values.argmax(dim=-1).to(torch.long) + 1
    else:
        edge_attr = torch.ones(edge_index.size(1), dtype=torch.long)
    y = getattr(data, "y", None)
    y = y.detach().cpu().clone() if torch.is_tensor(y) else None
    return SimpleGraphData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def _load_qm9_bundle(
    root: Path,
    *,
    max_train_graphs,
    max_val_graphs,
    max_test_graphs,
    seed,
    include_hydrogens=False,
    protocol="legacy_local",
):
    if protocol == "gdss_catflow":
        return _load_qm9_gdss_catflow_bundle(
            root,
            max_train_graphs=max_train_graphs,
            max_val_graphs=max_val_graphs,
            max_test_graphs=max_test_graphs,
            seed=seed,
        )

    configure_warning_filters()
    try:
        from torch_geometric.datasets import QM9
    except Exception as exc:
        raise RuntimeError("QM9 support requires torch_geometric.datasets.QM9.") from exc

    atom_numbers = [1, 6, 7, 8, 9] if include_hydrogens else [6, 7, 8, 9]
    atom_to_label = {z: i for i, z in enumerate(atom_numbers)}
    atom_decoder = ["H", "C", "N", "O", "F"] if include_hydrogens else ["C", "N", "O", "F"]

    raw_sdf = Path(root) / "raw" / "gdb9.sdf"
    graphs = []
    pyg_exc = None
    loader_source = "pyg_qm9"
    force_raw_fallback = os.environ.get("QM9_FORCE_RAW_FALLBACK", "0") == "1"
    if force_raw_fallback:
        print("QM9_FORCE_RAW_FALLBACK=1; using raw SDF fallback.", flush=True)
    else:
        try:
            dataset = QM9(root=str(root))
            for data in dataset:
                graph = _qm9_pyg_data_to_simple(data, atom_to_label)
                if graph is None:
                    continue
                graphs.append(graph)
        except Exception as exc:
            pyg_exc = exc

    if not graphs:
        if not raw_sdf.exists():
            if pyg_exc is not None:
                raise RuntimeError("PyG QM9 processing failed and no raw SDF fallback exists.") from pyg_exc
            raise RuntimeError(f"No QM9 graphs loaded and no raw SDF fallback exists at {raw_sdf}")
        if pyg_exc is not None:
            print(
                f"QM9 PyG processing failed ({type(pyg_exc).__name__}: {pyg_exc}); using raw SDF fallback.",
                flush=True,
            )
        else:
            print("QM9 PyG processing produced no usable graphs; using raw SDF fallback.", flush=True)
        graphs = _load_qm9_from_raw_sdf(root, atom_to_label, include_hydrogens=include_hydrogens)
        loader_source = "qm9_raw_sdf"
        if not graphs:
            raise RuntimeError("QM9 raw SDF fallback produced no graphs.")

    train, val, test = _split_qm9_paper(
        graphs,
        seed=PAPER_DATASET_SEED,
        max_train=max_train_graphs,
        max_val=max_val_graphs,
        max_test=max_test_graphs,
    )
    node_counts = [int(graph.x.numel()) for graph in graphs]
    provenance = {
        "dataset": "QM9_H" if include_hydrogens else "QM9",
        "source": loader_source,
        "qm9_protocol": "legacy_local",
        "hydrogen_policy": "explicit" if include_hydrogens else "removed",
        "kekulized_before_graph_extraction": bool(
            loader_source == "qm9_raw_sdf" and not include_hydrogens
        ),
        "atom_types": list(atom_decoder),
        "node_count_min": min(node_counts) if node_counts else None,
        "node_count_max": max(node_counts) if node_counts else None,
        "node_count_mean": (sum(node_counts) / len(node_counts)) if node_counts else None,
        "split_seed": PAPER_DATASET_SEED,
        "split_rng": "python_random",
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
    }
    return GraphDatasetBundle(train, val, test, atom_decoder, [], "molecule", root, provenance)


def _load_qm9_gdss_catflow_bundle(
    root: Path,
    *,
    max_train_graphs,
    max_val_graphs,
    max_test_graphs,
    seed,
):
    """Load the heavy-atom QM9 representation and official GDSS test holdout.

    GDSS publishes a 13,082-item test-index list over its 133,885-row QM9 CSV.
    The remaining rows are its training universe.  This VAE requires validation
    traces, so a deterministic 10,000-row validation view is selected from that
    universe without removing those rows from training; the official test
    membership is left untouched.
    """
    from rdkit import Chem

    root = Path(root)
    asset_dir = root / "raw"
    csv_path = asset_dir / "gdss_qm9.csv"
    test_index_path = asset_dir / "valid_idx_qm9.json"
    verified_hashes = {}
    for filename, path in (("gdss_qm9.csv", csv_path), ("valid_idx_qm9.json", test_index_path)):
        asset = GDSS_QM9_ASSETS[filename]
        if not path.is_file():
            raise FileNotFoundError(
                f"GDSS/CatFlow QM9 asset is missing: {path}. Run "
                "`PREPARE_QM9=1 ./run_corrected_qm9_wl1_wl2_wl3.sh` or download "
                f"{asset['url']}."
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != asset["sha256"]:
            raise RuntimeError(
                f"GDSS/CatFlow QM9 asset checksum mismatch for {path}: "
                f"received {actual_sha256}, expected {asset['sha256']}"
            )
        verified_hashes[filename] = actual_sha256

    payload = json.loads(test_index_path.read_text())
    raw_test_indices = payload.get("valid_idxs") if isinstance(payload, dict) else None
    if not isinstance(raw_test_indices, list):
        raise RuntimeError(f"Unexpected GDSS QM9 test-index payload in {test_index_path}")
    test_indices = [int(index) for index in raw_test_indices]
    if len(test_indices) != 13082 or len(set(test_indices)) != len(test_indices):
        raise RuntimeError(
            f"GDSS QM9 test index must contain 13,082 distinct rows, received {len(test_indices)}"
        )
    test_set = set(test_indices)

    atom_to_label = {6: 0, 7: 1, 8: 2, 9: 3}
    atom_decoder = ["C", "N", "O", "F"]
    cache_path = root / "processed" / "gdss_catflow_qm9_graphs_v1.pt"
    cache_metadata = {"schema_version": 2, "asset_sha256": verified_hashes}
    graphs_by_index = None
    canonical_smiles_by_index = None
    if cache_path.is_file():
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cached.get("metadata") == cache_metadata:
                graphs_by_index = cached.get("graphs_by_index")
                canonical_smiles_by_index = cached.get("canonical_smiles_by_index")
        except Exception:
            graphs_by_index = None
            canonical_smiles_by_index = None

    if graphs_by_index is None or canonical_smiles_by_index is None:
        records = []
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for source_index, row in enumerate(reader):
                smiles = str(row.get("SMILES1") or "").strip()
                if not smiles:
                    raise RuntimeError(f"Missing SMILES1 at GDSS QM9 source row {source_index}")
                records.append((source_index, smiles))
        if len(records) != 133885:
            raise RuntimeError(f"GDSS QM9 CSV has {len(records)} rows; expected 133,885")

        graphs_by_index = {}
        canonical_smiles_by_index = {}
        atom_vocab = {symbol: label for label, symbol in enumerate(atom_decoder)}
        for source_index, smiles in records:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise RuntimeError(f"RDKit could not parse GDSS QM9 row {source_index}: {smiles!r}")
            mol = Chem.RemoveHs(mol)
            Chem.Kekulize(mol, clearAromaticFlags=True)
            if mol.GetNumAtoms() > 9:
                raise RuntimeError(
                    f"GDSS QM9 row {source_index} has {mol.GetNumAtoms()} heavy atoms; expected at most 9"
                )
            graph = graph_to_simple_data(mol, atom_vocab=atom_vocab)
            if graph is None:
                raise RuntimeError(f"Could not extract GDSS QM9 graph at row {source_index}")
            labels = {int(value) for value in graph.x.tolist()}
            if not labels.issubset(set(atom_to_label.values())):
                raise RuntimeError(f"Unexpected atom label(s) at GDSS QM9 row {source_index}: {sorted(labels)}")
            edge_types = {int(value) for value in graph.edge_attr.tolist()}
            if not edge_types.issubset({1, 2, 3}):
                raise RuntimeError(
                    f"Unexpected post-kekulization bond type(s) at GDSS QM9 row "
                    f"{source_index}: {sorted(edge_types)}"
                )
            graphs_by_index[source_index] = graph
            canonical_smiles_by_index[source_index] = Chem.MolToSmiles(
                mol,
                canonical=True,
                isomericSmiles=True,
                kekuleSmiles=True,
            )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_cache_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}")
        torch.save(
            {
                "metadata": cache_metadata,
                "graphs_by_index": graphs_by_index,
                "canonical_smiles_by_index": canonical_smiles_by_index,
            },
            temporary_cache_path,
        )
        os.replace(temporary_cache_path, cache_path)

    if len(graphs_by_index) != 133885:
        raise RuntimeError(f"GDSS QM9 graph cache has {len(graphs_by_index)} rows; expected 133,885")
    if min(test_indices) < 0 or max(test_indices) >= len(graphs_by_index):
        raise RuntimeError("GDSS QM9 test indices fall outside the CSV row universe")

    upstream_train_indices = [index for index in range(len(graphs_by_index)) if index not in test_set]
    validation_size = min(10000, max(1, len(upstream_train_indices) // 10))
    validation_order = torch.randperm(
        len(upstream_train_indices),
        generator=torch.Generator().manual_seed(int(seed)),
    ).tolist()
    validation_positions = set(validation_order[:validation_size])
    val_indices = [
        source_index
        for position, source_index in enumerate(upstream_train_indices)
        if position in validation_positions
    ]
    # Keep the complete GDSS training complement.  Validation is a model-
    # selection view of that training collection, not an additional outer
    # holdout and never includes official test molecules.
    train_indices = upstream_train_indices

    train = [graphs_by_index[index] for index in train_indices]
    val = [graphs_by_index[index] for index in val_indices]
    test = [graphs_by_index[index] for index in test_indices]
    node_counts = [int(graph.x.numel()) for graph in graphs_by_index.values()]
    provenance = {
        "dataset": "QM9",
        "source": "official_gdss_qm9_csv",
        "qm9_protocol": "gdss_catflow",
        "source_revision": GDSS_QM9_REVISION,
        "source_urls": {name: spec["url"] for name, spec in GDSS_QM9_ASSETS.items()},
        "asset_sha256": verified_hashes,
        "source_graphs": len(graphs_by_index),
        "hydrogen_policy": "removed_before_training",
        "kekulized_before_graph_extraction": True,
        "maximum_nodes": 9,
        "atom_types": atom_decoder,
        "bond_types": ["single", "double", "triple"],
        "node_count_min": min(node_counts),
        "node_count_max": max(node_counts),
        "node_count_mean": sum(node_counts) / len(node_counts),
        "official_test_indices": len(test_indices),
        "official_test_index_sha256": _index_list_sha256(test_indices),
        "internal_validation_policy": (
            "torch_randperm_training_subset_view_not_removed_from_gdss_training"
        ),
        "internal_validation_seed": int(seed),
        "internal_validation_indices": len(val_indices),
        "internal_validation_index_sha256": _index_list_sha256(val_indices),
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "comparison_scope": "CatFlow/GDSS heavy-atom QM9 test holdout",
    }
    train_smiles = [canonical_smiles_by_index[index] for index in train_indices]
    return GraphDatasetBundle(
        _cap(train, max_train_graphs),
        _cap(val, max_val_graphs),
        _cap(test, max_test_graphs),
        atom_decoder,
        _cap(train_smiles, max_train_graphs),
        "molecule",
        root,
        provenance,
    )


def _load_zinc250k_gdss_catflow_bundle(
    root: Path,
    *,
    max_train_graphs,
    max_val_graphs,
    max_test_graphs,
):
    """Load the CatFlow/GDSS ZINC250K CSV and official test-index holdout."""
    from rdkit import Chem

    root = Path(root)
    asset_dir = root / "raw"
    csv_path = asset_dir / "gdss_zinc250k.csv"
    test_index_path = asset_dir / "valid_idx_zinc250k.json"
    verified_hashes = {}
    for filename, path in (
        ("gdss_zinc250k.csv", csv_path),
        ("valid_idx_zinc250k.json", test_index_path),
    ):
        asset = GDSS_ZINC250K_ASSETS[filename]
        if not path.is_file():
            raise FileNotFoundError(
                f"GDSS/CatFlow ZINC250K asset is missing: {path}. "
                f"Download {asset['url']}."
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != asset["sha256"]:
            raise RuntimeError(
                f"GDSS/CatFlow ZINC250K asset checksum mismatch for {path}: "
                f"received {actual_sha256}, expected {asset['sha256']}"
            )
        verified_hashes[filename] = actual_sha256

    test_indices = [
        int(index)
        for index in json.loads(test_index_path.read_text(encoding="utf-8"))
    ]
    if len(test_indices) != 24887 or len(set(test_indices)) != len(test_indices):
        raise RuntimeError(
            "GDSS ZINC250K test index must contain 24,887 distinct rows; "
            f"received {len(test_indices)}"
        )
    test_set = set(test_indices)
    atom_decoder = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
    atom_vocab = {symbol: label for label, symbol in enumerate(atom_decoder)}
    cache_path = root / "processed" / "gdss_catflow_zinc250k_graphs_v1.pt"
    cache_metadata = {"schema_version": 1, "asset_sha256": verified_hashes}
    graphs_by_index = None
    canonical_smiles_by_index = None
    if cache_path.is_file():
        try:
            cached = torch.load(cache_path, map_location="cpu", weights_only=False)
            if cached.get("metadata") == cache_metadata:
                graphs_by_index = cached.get("graphs_by_index")
                canonical_smiles_by_index = cached.get("canonical_smiles_by_index")
        except Exception:
            graphs_by_index = None
            canonical_smiles_by_index = None

    if graphs_by_index is None or canonical_smiles_by_index is None:
        records = []
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for source_index, row in enumerate(reader):
                smiles = str(row.get("smiles") or "").strip()
                if not smiles:
                    raise RuntimeError(
                        f"Missing smiles at GDSS ZINC250K source row {source_index}"
                    )
                records.append((source_index, smiles))
        if len(records) != 249455:
            raise RuntimeError(
                f"GDSS ZINC250K CSV has {len(records)} rows; expected 249,455"
            )

        graphs_by_index = {}
        canonical_smiles_by_index = {}
        for source_index, smiles in records:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise RuntimeError(
                    f"RDKit could not parse GDSS ZINC250K row {source_index}: "
                    f"{smiles!r}"
                )
            mol = Chem.RemoveHs(mol)
            Chem.Kekulize(mol, clearAromaticFlags=True)
            if mol.GetNumAtoms() > 38:
                raise RuntimeError(
                    f"GDSS ZINC250K row {source_index} has "
                    f"{mol.GetNumAtoms()} heavy atoms; expected at most 38"
                )
            symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
            unsupported = sorted(symbols - set(atom_vocab))
            if unsupported:
                raise RuntimeError(
                    f"Unexpected atom types at GDSS ZINC250K row {source_index}: "
                    f"{unsupported}"
                )
            graph = graph_to_simple_data(mol, atom_vocab=dict(atom_vocab))
            edge_types = {int(value) for value in graph.edge_attr.tolist()}
            if not edge_types.issubset({1, 2, 3}):
                raise RuntimeError(
                    f"Unexpected bond types at GDSS ZINC250K row {source_index}: "
                    f"{sorted(edge_types)}"
                )
            graphs_by_index[source_index] = graph
            canonical_smiles_by_index[source_index] = Chem.MolToSmiles(
                mol,
                canonical=True,
                isomericSmiles=True,
                kekuleSmiles=True,
            )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_cache_path = cache_path.with_name(
            f"{cache_path.name}.tmp.{os.getpid()}"
        )
        torch.save(
            {
                "metadata": cache_metadata,
                "graphs_by_index": graphs_by_index,
                "canonical_smiles_by_index": canonical_smiles_by_index,
            },
            temporary_cache_path,
        )
        os.replace(temporary_cache_path, cache_path)

    if len(graphs_by_index) != 249455:
        raise RuntimeError(
            f"GDSS ZINC250K graph cache has {len(graphs_by_index)} rows; "
            "expected 249,455"
        )
    if min(test_indices) < 0 or max(test_indices) >= len(graphs_by_index):
        raise RuntimeError(
            "GDSS ZINC250K test indices fall outside the CSV row universe"
        )

    train_indices = [
        index for index in range(len(graphs_by_index)) if index not in test_set
    ]
    validation_size = min(10000, max(1, len(train_indices) // 10))
    val_indices = train_indices[:validation_size]
    train = [graphs_by_index[index] for index in train_indices]
    val = [graphs_by_index[index] for index in val_indices]
    test = [graphs_by_index[index] for index in test_indices]
    node_counts = [int(graph.x.numel()) for graph in graphs_by_index.values()]
    provenance = {
        "dataset": "ZINC250K",
        "source": "official_gdss_zinc250k_csv",
        "zinc250k_protocol": "gdss_catflow",
        "source_revision": CATFLOW_GDSS_REVISION,
        "source_urls": {
            name: spec["url"] for name, spec in GDSS_ZINC250K_ASSETS.items()
        },
        "asset_sha256": verified_hashes,
        "source_graphs": len(graphs_by_index),
        "hydrogen_policy": "removed_before_training",
        "kekulized_before_graph_extraction": True,
        "maximum_nodes": 38,
        "atom_types": atom_decoder,
        "bond_types": ["single", "double", "triple"],
        "node_count_min": min(node_counts),
        "node_count_max": max(node_counts),
        "node_count_mean": sum(node_counts) / len(node_counts),
        "official_test_indices": len(test_indices),
        "official_test_index_sha256": _index_list_sha256(test_indices),
        "internal_validation_policy": (
            "ordered_training_subset_view_not_removed_from_gdss_training"
        ),
        "internal_validation_indices": len(val_indices),
        "internal_validation_index_sha256": _index_list_sha256(val_indices),
        "split_sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "comparison_scope": "CatFlow/GDSS heavy-atom ZINC250K test holdout",
    }
    train_smiles = [
        canonical_smiles_by_index[index] for index in train_indices
    ]
    return GraphDatasetBundle(
        _cap(train, max_train_graphs),
        _cap(val, max_val_graphs),
        _cap(test, max_test_graphs),
        atom_decoder,
        _cap(train_smiles, max_train_graphs),
        "molecule",
        root,
        provenance,
    )


def _load_qm9_from_raw_sdf(root: Path, atom_to_label: dict[int, int], *, include_hydrogens=False):
    raw_path = Path(root) / "raw" / "gdb9.sdf"
    if not raw_path.exists():
        return []
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(str(raw_path), removeHs=False, sanitize=True)
    graphs = []
    skipped = Counter()
    total = 0
    for mol in supplier:
        total += 1
        if mol is None:
            skipped["invalid_mol"] += 1
            continue
        if not include_hydrogens:
            mol = Chem.RemoveHs(mol)
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception:
                skipped["kekulize_failed"] += 1
                continue
        labels = []
        unsupported = False
        for atom in mol.GetAtoms():
            atomic_num = int(atom.GetAtomicNum())
            if atomic_num not in atom_to_label:
                unsupported = True
                break
            labels.append(atom_to_label[atomic_num])
        if unsupported:
            skipped["unsupported_atom"] += 1
            continue
        if len(labels) < 2:
            skipped["too_few_heavy_atoms"] += 1
            continue
        edges = []
        edge_labels = []
        for bond in mol.GetBonds():
            u = int(bond.GetBeginAtomIdx())
            v = int(bond.GetEndAtomIdx())
            edge_type = _rdkit_bond_to_int(bond)
            edges.append((u, v))
            edges.append((v, u))
            edge_labels.append(edge_type)
            edge_labels.append(edge_type)
        if not edges:
            skipped["no_heavy_atom_edges"] += 1
            continue
        graphs.append(SimpleGraphData(
            x=torch.tensor(labels, dtype=torch.long),
            edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(edge_labels, dtype=torch.long),
            y=None,
        ))
    skipped_total = sum(skipped.values())
    if skipped_total:
        mode = "with hydrogens" if include_hydrogens else "heavy atoms only"
        print(
            f"QM9 raw SDF fallback ({mode}) loaded {len(graphs)} / {total} molecules; "
            f"skipped {skipped_total} ({dict(skipped)}).",
            flush=True,
        )
    return graphs


def _qm9_pyg_data_to_simple(data, atom_to_label: dict[int, int]):
    z = getattr(data, "z", None)
    if z is None:
        return None
    z_values = [int(v) for v in z.view(-1).tolist()]
    kept_old = [idx for idx, atomic_num in enumerate(z_values) if atomic_num in atom_to_label]
    if len(kept_old) < 2:
        return None
    old_to_new = {old: new for new, old in enumerate(kept_old)}
    x = torch.tensor([atom_to_label[z_values[old]] for old in kept_old], dtype=torch.long)
    edge_index_in = data.edge_index.detach().cpu().to(torch.long)
    raw_edge_attr = _qm9_edge_attr_to_bond_ids(getattr(data, "edge_attr", None), edge_index_in.size(1))
    edges = []
    edge_labels = []
    for edge_pos in range(edge_index_in.size(1)):
        u = int(edge_index_in[0, edge_pos].item())
        v = int(edge_index_in[1, edge_pos].item())
        if u == v or u not in old_to_new or v not in old_to_new:
            continue
        edges.append((old_to_new[u], old_to_new[v]))
        edge_labels.append(int(raw_edge_attr[edge_pos].item()))
    if not edges:
        return None
    return SimpleGraphData(
        x=x,
        edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_labels, dtype=torch.long),
        y=None,
    )


def _qm9_edge_attr_to_bond_ids(edge_attr, n_edges: int):
    if edge_attr is None or int(n_edges) == 0:
        return torch.ones(int(n_edges), dtype=torch.long)
    edge_attr = edge_attr.detach().cpu()
    if edge_attr.dim() == 1:
        values = edge_attr.to(torch.long)
        return torch.clamp(values, min=1, max=4)
    idx = edge_attr.argmax(dim=-1).to(torch.long) + 1
    return torch.clamp(idx, min=1, max=4)


def _load_moses_bundle(
    root: Path,
    *,
    smiles_file,
    moses_split,
    max_train_graphs,
    max_val_graphs,
    max_test_graphs,
    seed,
    dataset_label="MOSES",
    require_smiles_file=False,
):
    if require_smiles_file and not smiles_file:
        raise RuntimeError(f"{dataset_label} requires --smiles-file pointing to its training SMILES file.")
    source_smiles = _load_moses_smiles(smiles_file=smiles_file, split=moses_split)
    if not source_smiles:
        raise RuntimeError(
            f"No {dataset_label} SMILES were loaded. Pass --smiles-file"
            + (", or install the optional moses package." if not require_smiles_file else ".")
        )
    source_smiles_count = len(source_smiles)
    smiles = _precap_smiles_for_split(
        source_smiles,
        seed=seed,
        max_train=max_train_graphs,
        max_val=max_val_graphs,
        max_test=max_test_graphs,
    )
    if smiles is not source_smiles:
        source_smiles = None

    from rdkit import Chem

    atom_vocab, atom_decoder = _rdkit_atomic_number_vocabulary()
    graphs = []
    clean_smiles = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        data = graph_to_simple_data(mol, atom_vocab=atom_vocab)
        if data is None or data.num_nodes < 2:
            continue
        graphs.append(data)
        clean_smiles.append(Chem.MolToSmiles(mol, canonical=True))

    train, val, test = _split_items(
        list(zip(graphs, clean_smiles)),
        seed=seed,
        max_train=max_train_graphs,
        max_val=max_val_graphs,
        max_test=max_test_graphs,
    )
    source_path = Path(smiles_file).expanduser().resolve() if smiles_file else None
    provenance = {
        "dataset": str(dataset_label),
        "source": "smiles_file" if source_path is not None else "moses_package",
        "smiles_file": portable_path(source_path),
        "smiles_file_sha256": _sha256_file(source_path) if source_path is not None else None,
        "moses_split": str(moses_split),
        "source_smiles": source_smiles_count,
        "preselected_smiles": len(smiles),
        "usable_graphs": len(graphs),
        "split_sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "atom_label_scheme": "rdkit_atomic_number_v1",
    }
    return GraphDatasetBundle(
        train_graphs=[g for g, _s in train],
        val_graphs=[g for g, _s in val],
        test_graphs=[g for g, _s in test],
        atom_decoder=atom_decoder,
        train_smiles=[s for _g, s in train],
        mode="molecule",
        data_root=root,
        provenance=provenance,
    )


def _load_moses_smiles(*, smiles_file, split: str):
    if smiles_file:
        return _read_smiles_file(Path(smiles_file))
    try:
        import moses  # type: ignore
    except Exception:
        return []
    return [str(s) for s in moses.get_dataset(split)]


def _read_smiles_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing SMILES file: {path}")
    suffix = path.suffix.lower()
    if suffix in {".smi", ".smiles", ".txt"}:
        out = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line.split()[0])
        return out

    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        rows = [row for row in reader if row and row[0].strip() and not row[0].strip().startswith("#")]
    if not rows:
        return []
    candidates = {"smiles", "smi", "mol"}
    first = [cell.strip() for cell in rows[0]]
    lower = [cell.lower() for cell in first]
    if any(name in candidates for name in lower):
        field_idx = next(idx for idx, name in enumerate(lower) if name in candidates)
        data_rows = rows[1:]
    else:
        field_idx = 0
        data_rows = rows
    out = []
    for row in data_rows:
        if len(row) > field_idx and row[field_idx].strip():
            out.append(row[field_idx].strip())
    return out


def _rdkit_bond_to_int(bond):
    from rdkit import Chem

    btype = bond.GetBondType()
    if btype == Chem.BondType.AROMATIC:
        return 4
    if btype == Chem.BondType.DOUBLE:
        return 2
    if btype == Chem.BondType.TRIPLE:
        return 3
    return 1
