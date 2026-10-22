# Benchmark protocols

This document records the evaluation contracts implemented by the benchmark
launchers. Generated data, checkpoints, caches, and historical logs are not
distributed in the submission package.

## CatFlow abstract graphs

The main abstract-graph comparison uses the checksum-pinned GDSS collections
from revision `24cc490e0c5b39cbd265fda33ab2bc8b528d4454`.

| Dataset | Training graphs | Validation view | Test/generated graphs |
|---|---:|---:|---:|
| Ego-small | 160 | 32 | 40 |
| Community-small | 80 | 16 | 20 |
| Enzymes | 470 | 94 | 117 |
| Grid | 80 | 16 | 20 |

The first `int(0.2 * N)` graphs form the test set and the remaining graphs form
the training set. There is no outer-split shuffle or split seed. Validation is
a deterministic view of the training collection and is not removed from it.

The evaluator generates exactly the test-set size and reports raw biased
`MMD²`:

- degree: Gaussian-EMD kernel with sigma 1;
- clustering: 100 bins on `[0, 1]`, Gaussian-EMD sigma 0.1, distance scale 100;
- orbit: exact ORCA 15-orbit features through four-node graphlets, normalized
  per node, with a Gaussian kernel of sigma 30.

These values are not the train-test-normalized ratios used by the
DiGress/SPECTRE tables. `src/recompute_catflow_metrics.py` fails closed when
the dataset asset, split, sample count, checkpoint provenance, or ORCA backend
does not satisfy this contract.

## QM9 and ZINC250K

Both molecule evaluations use checksum-pinned GDSS CSV files and published
test-index files. The complement of the test indices is the full training
collection; validation remains a deterministic training view.

| Dataset | Total rows | Training rows | Test rows | Maximum heavy atoms |
|---|---:|---:|---:|---:|
| QM9 | 133,885 | 120,803 | 13,082 | 9 |
| ZINC250K | 249,455 | 224,568 | 24,887 | 38 |

Training removes hydrogen and kekulizes molecules before graph extraction.
Evaluation uses 10,000 generation attempts. Each generated molecule is
sanitized and kekulized before largest-fragment canonicalization, preventing
an invalid component from being discarded to rescue the sample. The reported
metrics are:

- validity over all attempts;
- uniqueness over valid canonical SMILES;
- FCD against the official test reference using unique valid generated
  SMILES.

The FCD implementation and environment are stored with each run because the
baseline release does not pin an exact package and weight version.

## DiGress/SPECTRE graphs

Planar and SBM use the pinned SPECTRE assets, Torch seed-0 membership,
128/32/40 train/validation/test graphs, and 40 generated graphs. The evaluator
reports DiGress/SPECTRE degree, clustering, and orbit ratios. Topology validity,
uniqueness, and novelty use the exact graph-tool SBM refinement backend when
required.

PROTEINS follows the SPECTRE DD preprocessing and split contract. The evaluator
removes isolates and self-loops, retains graphs with 100--500 nodes, and uses
the fixed paper-era split.

## Checkpoint and seed policy

Training writes three distinct checkpoint roles:

- `best_ring_vae.pt`: validation-only fixed-beta structural selection;
- `best_legacy_ring_vae.pt`: legacy warmup-weighted validation loss;
- `last_ring_vae.pt`: final epoch.

Checkpoint selection never uses test FCD. Multi-seed reports use predeclared
seeds, retain every seed, report the sample count, and aggregate with the
arithmetic mean and population standard deviation. Incomplete seed groups are
reported per seed and are not averaged.

Generation is evaluated in separate strict and permissive modes:

- strict requires every WL stub to be completed;
- permissive accepts connected partial graphs and reports completion and
  validity alongside distribution metrics.

## Reproduction commands

The launchers resolve paths relative to the repository. Set `PYBIN` only when
a specific environment interpreter is required.

Prepare CatFlow assets:

```bash
PYTHONPATH=src python src/prepare_official_catflow_data.py \
  --output-dir benchmark_data/official_catflow
```

Run the declared three-seed abstract-graph schedule:

```bash
CATFLOW_SEEDS="0 1 2" CATFLOW_WL_ITERS="1 2 3" \
  bash scripts/run_catflow_multiseed.sh
```

Run molecule schedules separately:

```bash
CATFLOW_SEEDS="0 1 2" \
  bash scripts/run_catflow_qm9_multiseed.sh

CATFLOW_SEEDS="0 1 2" \
  bash scripts/run_catflow_zinc250k_multiseed.sh
```

Run the corrected dataset-group sweeps:

```bash
bash scripts/run_corrected_graphs_wl1_wl2_wl3.sh
bash scripts/run_corrected_qm9_wl1_wl2_wl3.sh
bash scripts/run_corrected_zinc250k_wl2_wl3.sh
bash scripts/run_corrected_moses_wl2_wl3.sh
bash scripts/run_corrected_guacamol_wl2_wl3.sh
```

The CatFlow main table and appendix use different upstream split and metric
contracts. Do not combine them into one supposedly CatFlow-comparable row.
