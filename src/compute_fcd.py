"""Compute Fréchet ChemNet Distance from reference and generated SMILES."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_smiles(path: str | Path):
    return [
        line.strip().split()[0]
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def preprocess_smiles(smiles, mode: str):
    mode = str(mode or "none").lower()
    if mode == "none":
        return list(smiles)
    if mode != "catflow":
        raise ValueError(f"Unsupported preprocessing mode: {mode}")

    from rdkit import Chem

    out = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        try:
            mol = Chem.RemoveHs(mol)
            Chem.SanitizeMol(mol)
            Chem.Kekulize(mol, clearAromaticFlags=True)
            out.append(Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True, kekuleSmiles=True))
        except Exception:
            continue
    return out


def compute_fcd(ref_smiles, gen_smiles, *, device: str, n_jobs: int, batch_size: int):
    try:
        from fcd_torch import FCD

        fcd = FCD(device=device, n_jobs=n_jobs, batch_size=batch_size)
        return float(fcd(ref=ref_smiles, gen=gen_smiles)), "fcd_torch"
    except ImportError:
        pass

    try:
        from fcd import get_fcd

        return float(get_fcd(ref_smiles, gen_smiles)), "fcd"
    except ImportError as exc:
        raise RuntimeError(
            "No FCD implementation found. Install one of: "
            "`pip install fcd_torch` or `pip install fcd`."
        ) from exc


def main():
    parser = argparse.ArgumentParser(description="Compute Frechet ChemNet Distance from two SMILES files.")
    parser.add_argument("--ref", required=True, help="Reference/test SMILES file.")
    parser.add_argument("--gen", required=True, help="Generated valid SMILES file.")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, ...")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--preprocess", default="none", choices=["none", "catflow"],
                        help="Optional RDKit canonicalization before ChemNet embedding.")
    args = parser.parse_args()

    raw_ref_smiles = read_smiles(args.ref)
    raw_gen_smiles = read_smiles(args.gen)
    ref_smiles = preprocess_smiles(raw_ref_smiles, args.preprocess)
    gen_smiles = preprocess_smiles(raw_gen_smiles, args.preprocess)
    value, backend = compute_fcd(
        ref_smiles,
        gen_smiles,
        device=args.device,
        n_jobs=args.n_jobs,
        batch_size=args.batch_size,
    )
    result = {
        "fcd": value,
        "backend": backend,
        "preprocess": args.preprocess,
        "n_reference_input": len(raw_ref_smiles),
        "n_generated_input": len(raw_gen_smiles),
        "n_reference": len(ref_smiles),
        "n_generated": len(gen_smiles),
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
