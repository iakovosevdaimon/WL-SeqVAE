"""Molecule conversion, validity, similarity, and FCD utility functions."""

from __future__ import annotations

import hashlib
import math
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import torch

from rdkit import Chem, RDLogger

from project_paths import DATASETS_ROOT

RDLogger.DisableLog("rdApp.*")

DEFAULT_ZINC_ROOT = DATASETS_ROOT / "ZINC"
RAW_ZINC_ROOT = DEFAULT_ZINC_ROOT / "raw"

INT_TO_BOND_TYPE = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
    4: Chem.BondType.AROMATIC,
}

MOL_PREPROCESS_MODES = ("repo", "catflow")
SMILES_CACHE_SCHEMA_VERSION = 1


class Dictionary:
    def __init__(self):
        pass


@dataclass
class GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: torch.Tensor | None = None

    @property
    def num_nodes(self):
        return int(self.x.size(0))


def _ensure_torch_geometric_data_shim():
    if "torch_geometric.data.data" in sys.modules:
        return

    mod_tg = types.ModuleType("torch_geometric")
    mod_data_pkg = types.ModuleType("torch_geometric.data")
    mod_data = types.ModuleType("torch_geometric.data.data")

    class Data:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def __setstate__(self, state):
            self.__dict__.update(state)

    mod_data.Data = Data
    sys.modules["torch_geometric"] = mod_tg
    sys.modules["torch_geometric.data"] = mod_data_pkg
    sys.modules["torch_geometric.data.data"] = mod_data


def _load_dictionary_pickle(path):
    sys.modules["__main__"].Dictionary = Dictionary
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_atom_decoder(root=None):
    raw_root = Path(root) / "raw" if root is not None else RAW_ZINC_ROOT
    atom_dict = _load_dictionary_pickle(raw_root / "atom_dict.pickle")
    return list(atom_dict.idx2word)


def remove_self_loops(data):
    edge_mask = data.edge_index[0] != data.edge_index[1]
    return GraphData(
        x=data.x,
        edge_index=data.edge_index[:, edge_mask],
        edge_attr=data.edge_attr[edge_mask],
        y=data.y,
    )


def _build_graph_from_serialized(data_dict, slices, graph_idx):
    node_start = int(slices["x"][graph_idx].item())
    node_end = int(slices["x"][graph_idx + 1].item())
    edge_start = int(slices["edge_index"][graph_idx].item())
    edge_end = int(slices["edge_index"][graph_idx + 1].item())
    y_start = int(slices["y"][graph_idx].item())
    y_end = int(slices["y"][graph_idx + 1].item())

    x = data_dict["x"][node_start:node_end].view(-1).to(torch.long).clone()
    edge_index = data_dict["edge_index"][:, edge_start:edge_end].to(torch.long).clone()
    edge_attr = data_dict["edge_attr"][edge_start:edge_end].view(-1).to(torch.long).clone()
    y = data_dict["y"][y_start:y_end].clone() if y_end > y_start else None
    return GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def load_processed_zinc_split(split="train", subset="small", root=None, max_graphs=None, drop_self_loops=True):
    if root is None:
        root = DEFAULT_ZINC_ROOT
    root = Path(root)

    subset = str(subset).lower()
    if subset not in {"small", "large"}:
        raise ValueError(f"Unsupported ZINC subset: {subset}")

    _ensure_torch_geometric_data_shim()
    processed_dir = "subset" if subset == "small" else "full"
    processed_path = root / processed_dir / "processed" / f"{split}.pt"
    data_dict, slices, _ = torch.load(processed_path, map_location="cpu", weights_only=False)

    total_graphs = int(slices["y"].numel() - 1)
    if max_graphs is not None:
        total_graphs = min(total_graphs, int(max_graphs))

    graphs = []
    for graph_idx in range(total_graphs):
        graph = _build_graph_from_serialized(data_dict, slices, graph_idx)
        if drop_self_loops:
            graph = remove_self_loops(graph)
        graphs.append(graph)
    return graphs


def graph_data_to_networkx(data):
    G = nx.Graph()
    for node_idx, atom_label in enumerate(data.x.view(-1).tolist()):
        G.add_node(node_idx, label=int(atom_label))

    edge_index = data.edge_index.tolist()
    edge_attr = getattr(data, "edge_attr", None)
    if edge_attr is None:
        edge_attr_values = [1] * int(data.edge_index.size(1))
    else:
        edge_attr_values = edge_attr.view(-1).tolist()
    for idx, bond_type in enumerate(edge_attr_values):
        u = int(edge_index[0][idx])
        v = int(edge_index[1][idx])
        if u < v and int(bond_type) > 0:
            G.add_edge(u, v, bond_type=int(bond_type))
    return G


def ensure_networkx_graph(graph):
    if isinstance(graph, nx.Graph):
        return graph
    if hasattr(graph, "x") and hasattr(graph, "edge_index"):
        return graph_data_to_networkx(graph)
    return None


def _parse_atom_spec(spec):
    tokens = str(spec).split()
    if not tokens:
        raise ValueError(f"Invalid atom specification: {spec!r}")

    symbol = tokens[0]
    explicit_hs = 0
    formal_charge = 0
    for token in tokens[1:]:
        if token.startswith("H"):
            explicit_hs = int(token[1:] or 1)
        elif token == "+":
            formal_charge += 1
        elif token == "-":
            formal_charge -= 1
        else:
            raise ValueError(f"Unsupported atom token {token!r} in specification {spec!r}")
    return symbol, explicit_hs, formal_charge


def build_atom_valence_limits(atom_decoder):
    """Conservative max-bond-order budget per atom-token label.

    This is a lightweight pre-RDKit filter for constructive decoding. RDKit
    sanitization remains the source of truth; these limits simply prevent the
    decoder from knowingly adding impossible high-valence edges.
    """
    base_valence = {
        "H": 1,
        "C": 4,
        "N": 4,
        "O": 2,
        "F": 1,
        "Cl": 1,
        "Br": 1,
        "I": 1,
        "S": 6,
        "P": 5,
    }
    limits = {}
    for label, spec in enumerate(atom_decoder):
        symbol, explicit_hs, formal_charge = _parse_atom_spec(spec)
        max_v = base_valence.get(symbol, 4)
        if symbol == "O":
            if formal_charge < 0:
                max_v = 1
            elif formal_charge > 0:
                max_v = 3
        elif symbol == "N":
            if formal_charge < 0:
                max_v = 2
            else:
                max_v = 4
        limits[int(label)] = max(0, int(max_v) - int(explicit_hs))
    return limits


def atom_label_to_rdkit_atom(label, atom_decoder):
    spec = atom_decoder[int(label)]
    symbol, explicit_hs, formal_charge = _parse_atom_spec(spec)
    atom = Chem.Atom(symbol)
    atom.SetFormalCharge(formal_charge)
    if explicit_hs > 0:
        atom.SetNumExplicitHs(explicit_hs)
        atom.SetNoImplicit(True)
    return atom


def nx_to_mol(G, atom_decoder):
    mol = Chem.RWMol()
    node_to_idx = {}
    try:
        for node in sorted(G.nodes()):
            label = int(G.nodes[node].get("label", 0))
            atom = atom_label_to_rdkit_atom(label, atom_decoder)
            node_to_idx[node] = mol.AddAtom(atom)

        for u, v, data in G.edges(data=True):
            bond_id = int(data.get("bond_type", 1))
            bond_type = INT_TO_BOND_TYPE.get(bond_id, Chem.BondType.SINGLE)
            if bond_id == 4:
                mol.GetAtomWithIdx(node_to_idx[u]).SetIsAromatic(True)
                mol.GetAtomWithIdx(node_to_idx[v]).SetIsAromatic(True)
            mol.AddBond(node_to_idx[u], node_to_idx[v], bond_type)

        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def _normalize_mol_preprocess(preprocess: str | None) -> str:
    mode = str(preprocess or "repo").lower()
    if mode not in MOL_PREPROCESS_MODES:
        raise ValueError(f"Unsupported molecule preprocessing mode {preprocess!r}; expected {MOL_PREPROCESS_MODES}")
    return mode


def prepare_mol_for_smiles(mol, *, preprocess: str = "repo"):
    mode = _normalize_mol_preprocess(preprocess)
    mol = Chem.Mol(mol)
    if mode == "catflow":
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol


def smiles_from_mol(mol, *, preprocess: str = "repo"):
    mode = _normalize_mol_preprocess(preprocess)
    try:
        mol = prepare_mol_for_smiles(mol, preprocess=mode)
        if mode == "catflow":
            # The released CatFlow evaluator validates/kekulizes the complete
            # output first, then canonicalizes its largest molecular fragment.
            # Selecting a fragment before sanitization would incorrectly rescue
            # outputs containing an invalid component.
            fragments = Chem.GetMolFrags(
                mol,
                asMols=True,
                sanitizeFrags=False,
            )
            if fragments:
                mol = max(
                    fragments,
                    key=lambda fragment: fragment.GetNumAtoms(),
                )
        return Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
            kekuleSmiles=(mode == "catflow"),
        )
    except Exception:
        return None


def canonical_smiles_from_graph(G, atom_decoder, *, preprocess: str = "repo"):
    G = ensure_networkx_graph(G)
    if G is None:
        return None
    mol = nx_to_mol(G, atom_decoder)
    if mol is None:
        return None
    return smiles_from_mol(mol, preprocess=preprocess)


def _smiles_graph_fingerprint(graphs):
    digest = hashlib.sha256()
    digest.update(str(len(graphs)).encode("utf-8"))
    for graph in graphs:
        nx_graph = ensure_networkx_graph(graph)
        if nx_graph is None:
            item = ("invalid", type(graph).__name__)
        else:
            labels = tuple(sorted(
                (int(node), int(data.get("label", 0)))
                for node, data in nx_graph.nodes(data=True)
            ))
            edges = tuple(sorted(
                (
                    min(int(u), int(v)),
                    max(int(u), int(v)),
                    int(data.get("bond_type", data.get("edge_attr", 1))),
                )
                for u, v, data in nx_graph.edges(data=True)
                if int(u) != int(v)
            ))
            item = (len(labels), len(edges), labels, edges)
        digest.update(repr(item).encode("utf-8"))
    return digest.hexdigest()


def _smiles_cache_provenance(graphs, atom_decoder, *, preprocess: str, deduplicate: bool):
    return {
        "schema_version": SMILES_CACHE_SCHEMA_VERSION,
        "graph_count": len(graphs),
        "graph_fingerprint": _smiles_graph_fingerprint(graphs),
        "atom_decoder": [str(value) for value in atom_decoder],
        "preprocess": str(preprocess),
        "deduplicate": bool(deduplicate),
    }


def build_smiles(graphs, atom_decoder, cache_path=None, *, preprocess: str = "repo", deduplicate: bool = False):
    preprocess = _normalize_mol_preprocess(preprocess)
    atom_decoder = list(atom_decoder)
    provenance = None
    if cache_path is not None:
        graphs = list(graphs)
        provenance = _smiles_cache_provenance(
            graphs,
            atom_decoder,
            preprocess=preprocess,
            deduplicate=deduplicate,
        )
        cache_path = Path(cache_path)
        metadata_path = cache_path.with_name(f"{cache_path.name}.meta.json")
        if cache_path.exists() and metadata_path.exists():
            try:
                cached_metadata = json.loads(metadata_path.read_text())
                cached_smiles = json.loads(cache_path.read_text())
                if cached_metadata == provenance and isinstance(cached_smiles, list):
                    return cached_smiles
            except (OSError, ValueError, TypeError):
                pass

    smiles = []
    for graph in graphs:
        smi = canonical_smiles_from_graph(graph, atom_decoder, preprocess=preprocess)
        if smi:
            smiles.append(smi)

    if deduplicate:
        smiles = sorted(set(smiles))
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(smiles))
        metadata_path.write_text(json.dumps(provenance, sort_keys=True))
    return smiles


def build_training_smiles(graphs, atom_decoder, cache_path=None, *, preprocess: str = "repo", deduplicate: bool = True):
    return build_smiles(
        graphs,
        atom_decoder,
        cache_path=cache_path,
        preprocess=preprocess,
        deduplicate=deduplicate,
    )


def compute_mol_metrics(generated_graphs, train_smiles, atom_decoder, *, preprocess: str = "repo", requested_total: int | None = None):
    valid_smiles = []
    for G in generated_graphs:
        smi = canonical_smiles_from_graph(G, atom_decoder, preprocess=preprocess)
        if smi:
            valid_smiles.append(smi)

    unique_smiles = sorted(set(valid_smiles))
    train_set = (
        train_smiles
        if isinstance(train_smiles, (set, frozenset))
        else set(train_smiles or [])
    )
    novel_smiles = sorted(set(unique_smiles) - train_set)

    n_total = len(generated_graphs)
    n_requested = int(requested_total) if requested_total is not None else n_total
    validity_given_graph = len(valid_smiles) / n_total if n_total > 0 else 0.0
    validity_per_requested_attempt = len(valid_smiles) / max(n_requested, 1)
    uniqueness = len(unique_smiles) / len(valid_smiles) if valid_smiles else 0.0
    novelty = len(novel_smiles) / len(unique_smiles) if unique_smiles else 0.0

    return {
        "n_total": n_total,
        "n_requested": n_requested,
        "n_valid": len(valid_smiles),
        "n_unique": len(unique_smiles),
        "n_novel": len(novel_smiles),
        "validity": validity_per_requested_attempt,
        "validity_per_requested_attempt": validity_per_requested_attempt,
        "validity_given_graph": validity_given_graph,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "valid_smiles": valid_smiles,
        "unique_smiles": unique_smiles,
        "novel_smiles": novel_smiles,
    }


def _parse_fcd_value(text):
    if not text:
        return None
    text = str(text).strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("fcd", "FCD"):
                if key in payload:
                    value = float(payload[key])
                    return value if math.isfinite(value) and value >= 0.0 else None
        if isinstance(payload, (int, float)):
            value = float(payload)
            return value if math.isfinite(value) and value >= 0.0 else None
    except Exception:
        pass

    number_pattern = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
    for key in ("fcd",):
        matches = re.findall(
            rf'"?{key}"?\s*[:=]\s*({number_pattern})',
            text,
            flags=re.IGNORECASE,
        )
        if matches:
            value = float(matches[-1])
            return value if math.isfinite(value) and value >= 0.0 else None

    if re.fullmatch(number_pattern, text):
        value = float(text)
        return value if math.isfinite(value) and value >= 0.0 else None
    return None


def _parse_fcd_outputs(*texts):
    for text in texts:
        value = _parse_fcd_value(text)
        if value is not None:
            return value
    return None


def _parse_fcd_metadata(*texts):
    for text in texts:
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            return {"backend": payload["backend"]} if "backend" in payload else {}
    return {}


def compute_fcd_with_command(reference_smiles, generated_smiles, command, *, work_dir=None, timeout=1200):
    """Run an external FCD implementation.

    The command may contain {ref}, {gen}, and {out} placeholders. If no
    placeholders are used, the reference and generated SMILES files are appended
    as positional arguments. The command can either print a numeric FCD value or
    write JSON/text containing the value to {out}.
    """
    if not command:
        return None, {"status": "missing_command"}
    if not reference_smiles or not generated_smiles:
        return None, {"status": "missing_smiles"}

    base_dir = (
        Path(work_dir).expanduser().resolve()
        if work_dir is not None
        else Path(tempfile.mkdtemp(prefix="ring_vae_fcd_")).resolve()
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    ref_path = base_dir / "fcd_reference_smiles.txt"
    gen_path = base_dir / "fcd_generated_smiles.txt"
    out_path = base_dir / "fcd_result.json"
    ref_path.write_text("\n".join(reference_smiles) + "\n")
    gen_path.write_text("\n".join(generated_smiles) + "\n")

    cmd = str(command)
    placeholders = {
        "{ref}": str(ref_path),
        "{gen}": str(gen_path),
        "{out}": str(out_path),
    }
    used_placeholder = False
    for placeholder, value in placeholders.items():
        if placeholder in cmd:
            used_placeholder = True
            cmd = cmd.replace(placeholder, value)
    if not used_placeholder:
        cmd = f"{cmd} {ref_path} {gen_path}"

    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(timeout),
        )
    except Exception as exc:
        return None, {"status": "failed", "error": str(exc)}

    info = {
        "returncode": int(proc.returncode),
        "command": str(command),
        "reference_smiles": len(reference_smiles),
        "generated_smiles": len(generated_smiles),
    }
    if proc.returncode != 0:
        info.update({
            "status": "failed",
            "stdout_tail": (proc.stdout or "")[-1000:],
            "stderr_tail": (proc.stderr or "")[-1000:],
        })
        return None, info

    out_text = out_path.read_text() if out_path.exists() else ""
    info.update(_parse_fcd_metadata(out_text, proc.stdout or "", proc.stderr or ""))
    value = _parse_fcd_outputs(out_text, proc.stdout or "", proc.stderr or "")
    info["status"] = "ok" if value is not None else "failed"
    if value is None:
        info["stdout_tail"] = (proc.stdout or "")[-1000:]
        info["stderr_tail"] = (proc.stderr or "")[-1000:]
    return value, info
