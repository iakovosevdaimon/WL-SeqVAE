# Ring-structured WL graph generation

Anonymous source-code supplement for a graph variational autoencoder that
encodes Weisfeiler--Leman (WL) color sequences and generates graphs with a
ring-structured autoregressive decoder.

```text
WL(T) sequence -> Transformer encoder -> diagonal Gaussian latent
latent + partial ring state -> constrained ring-by-ring decoder
```

The decoder predicts a center-node color, expands required neighbors using
the previous WL stage, assigns final-stage colors to created nodes, and
predicts previous-ring and intra-ring edges. The submitted configuration uses
node/edge budget context and constrained closure.

## Repository layout

```text
src/          model, training, generation, data, and metric code
scripts/      portable benchmark and data-preparation launchers
tests/        focused correctness regression suite
docs/         benchmark protocols and aggregation rules
third_party/  external evaluation executable
```

Downloaded datasets, checkpoints, trace caches, generated graphs, logs, and
historical result snapshots are intentionally excluded. The scripts recreate
their directory structure when run.

## Installation

Python 3.11 is recommended. Create and activate an environment, then install:

```bash
python -m pip install -r requirements.txt
```

For the complete evaluation stack:

```bash
python -m pip install -r requirements-evaluation.txt
```

Exact SBM evaluation additionally requires `graph-tool`, which is normally
installed from conda-forge. After activating the target environment, the
helper below installs and validates `graph-tool` and `pyemd`:

```bash
bash scripts/setup_exact_graph_tool_env.sh
```

## Validation

Run the regression suite from the repository root:

```bash
PYTHONPATH=src python tests/correctness_regression_tests.py
```

Before packaging, run the anonymity and artifact check:

```bash
python scripts/check_submission.py
```

## Training

The following small CPU run verifies the end-to-end training path:

```bash
PYTHONPATH=src python src/train_ring_vae.py \
  --dataset COMMUNITY_SMALL \
  --n-wl-iters 1 \
  --max-train-graphs 4 \
  --max-val-graphs 1 \
  --epochs 1 \
  --batch-size 2 \
  --hidden-dim 32 \
  --latent-dim 16 \
  --n-layers 1 \
  --n-heads 2 \
  --device cpu \
  --trace-cache trace_cache/smoke.pt \
  --output-dir checkpoints/smoke
```

For the budget-conditioned model, add:

```text
--budget-context
--edge-budget-context
--no-action-head
--no-completion-head
```

Training writes three checkpoint roles:

```text
best_ring_vae.pt         fixed-beta validation/structural selection
best_legacy_ring_vae.pt  legacy warmup-weighted validation loss
last_ring_vae.pt         final epoch
```

## Generation

Generate from a selected checkpoint:

```bash
PYTHONPATH=src python src/run_ring_vae.py \
  --checkpoint checkpoints/smoke/best_ring_vae.pt \
  --dataset COMMUNITY_SMALL \
  --n-samples 2 \
  --allow-incomplete \
  --max-restarts 1 \
  --device cpu \
  --output-dir inference_runs/smoke
```

The one-epoch smoke checkpoint is intentionally evaluated in permissive mode
to keep this check short. Strict generation is the benchmark default. Use
`--allow-incomplete` only for diagnostics; completion and validity must then
be reported with the distribution metrics.

## Benchmarks

The canonical entry point is:

```bash
bash scripts/run_all_benchmarks.sh
```

The five corrected dataset-group launchers are:

```bash
bash scripts/run_corrected_graphs_wl1_wl2_wl3.sh
bash scripts/run_corrected_qm9_wl1_wl2_wl3.sh
bash scripts/run_corrected_zinc250k_wl2_wl3.sh
bash scripts/run_corrected_moses_wl2_wl3.sh
bash scripts/run_corrected_guacamol_wl2_wl3.sh
```

All launchers accept `PYBIN` as an optional interpreter override and otherwise
use `python`. Dataset paths and output roots are repository-relative and can
be overridden with the documented environment variables in each script.

Detailed split definitions, metric estimators, sample counts, seed handling,
and checkpoint aggregation rules are in
[docs/BENCHMARK_PROTOCOLS.md](docs/BENCHMARK_PROTOCOLS.md).


