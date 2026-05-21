# AItenea Docking — ML Rescoring of AutoDock Vina Poses

Machine learning rescoring of AutoDock Vina docking poses on the PDBbind 2020 general set
(~19,000 protein–ligand complexes). Two goals:

1. **Pose ranking** — select the docking pose closest to the crystal structure better than
   Vina's native ranking.
2. **Binding affinity prediction** — predict experimental ΔG (kcal/mol) from docking
   features and/or molecular fingerprints.

Test set: **CASF-2016** (285 complexes), used as a fixed benchmark.

---

## Environment

```bash
conda activate docking_aitenea   # Python 3.13, XGBoost, sklearn, RDKit, PyTorch Geometric
```

All commands below assume this environment. Replace
`python` with `/opt/anaconda3/envs/docking_aitenea/bin/python` if running outside the
activated environment.

---

## Scripts

| Script | Purpose |
|---|---|
| `process_pdbind.py` | Full pipeline: Vina docking → feature extraction → ML pose ranking + affinity prediction |
| `vina_docking.py` | Standalone Vina docking step for HPC SLURM job arrays (parallelised, restartable) |
| `gnn_affinity.py` | GINEConv-based GNN for binding affinity prediction (PyTorch Geometric) |
| `gnn_dimenet_affinity.py` | DimeNet++ GNN with protein feature augmentation for binding affinity prediction |
| `predict_affinity.py` | Standalone inference: predict ΔG for new ligands from pre-trained models |

---

## Current Results

| Task | Model | Metric | Value |
|---|---|---|---|
| Pose ranking | Vina (baseline) | Best-pose rate @ 2 Å | 28.2% |
| Pose ranking | RF ranker (optimised) | Best-pose rate @ 2 Å | **33.7%** |
| Pose ranking | Ensemble | Best-pose rate @ 2 Å | 33.0% |
| Affinity prediction | Vina (baseline) | Pearson r | 0.215 |
| Affinity prediction | GB regressor (optimised) | Pearson r | **0.649** |
| Affinity prediction | RF regressor | Pearson r | 0.634 |

Test set: CASF-2016 benchmark (285 complexes). Results above are from the previous refined-set run; will be updated after full PDBbind 2020 retraining.

---

## process_pdbind.py

### Initial run — Vina docking and CSV generation

This step runs AutoDock Vina on every complex, extracts features, and writes the three
fixed CSV files that are reused for all subsequent training runs.  It only needs to be
done once (or whenever the underlying structures change).

**Required directory layout before running:**

```
AItenea_Docking/
├── PDBind_2020/
│   ├── 1981-2000/{pdb_code}/{pdb_code}_ligand.mol2 + _protein.pdb
│   ├── 2001-2010/{pdb_code}/...
│   └── 2011-2019/{pdb_code}/...
├── PDBind_2020_index/
│   └── INDEX_general_PL.2020R1.lst
└── CASF-2016/
    └── coreset/{pdb_code}/{pdb_code}_ligand.mol2 + _protein.pdb
```

**Full run — process all PDBbind 2020 complexes + CASF-2016 test set (~many hours):**

```bash
python process_pdbind.py --num-complexes 19000
```

This will:
1. Run Vina on all 285 CASF-2016 complexes → `output/test_data.csv`
2. Run Vina on all non-CASF PDBbind 2020 complexes
3. Split non-CASF complexes 90% train / 10% val → `output/training_data.csv` + `output/val_data.csv`

**Speed up with `--no-augment` (skip mol2 + contact features, much faster but lower accuracy):**

```bash
python process_pdbind.py --num-complexes 19000 --no-augment
```

**Override the CASF directory or validation fraction:**

```bash
python process_pdbind.py \
  --num-complexes 19000 \
  --casf-dir CASF-2016/coreset \
  --val-frac 0.1
```

**Smoke-test with a small subset first:**

```bash
# Process 20 non-CASF complexes + all CASF — quick sanity check
python process_pdbind.py --num-complexes 20
```

Once the three CSVs are saved you should not need to re-run Vina. Use `--load-csv` for
all subsequent training and evaluation (see below).

---

### Typical usage (after CSVs are generated)

```bash
# Fast iteration — load pre-computed CSVs, train all models, run affinity prediction
python process_pdbind.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv

# Skip affinity prediction (pose ranking only)
python process_pdbind.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --no-affinity

# Load saved hyperparameters (no Optuna needed)
python process_pdbind.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --load-hyperparams output/best_hyperparams.json

# Full Optuna run for all models (~40–60 min)
python process_pdbind.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --optimize-hyperparams --n-trials 30 \
  --optuna-db output/optuna_studies.db

# Optimise only specific models (e.g. just the MLP affinity model)
python process_pdbind.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --load-hyperparams output/best_hyperparams.json \
  --optimize-hyperparams --optimize-models mlp_affinity

# Feature-set comparison: current features vs Morgan fingerprints
python process_pdbind.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --affinity-compare-features

# Feature-set comparison + optimise fingerprint models from scratch
python process_pdbind.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --load-hyperparams output/best_hyperparams.json \
  --affinity-compare-features \
  --optimize-hyperparams \
  --optimize-models mlp_affinity rf_affinity_fp gb_affinity_fp svr_affinity_fp \
                    xgb_affinity_fp ridge_affinity_fp mlp_affinity_fp \
  --optuna-db output/optuna_studies.db

# Reload both sets of saved hyperparameters (current features + fingerprints)
python process_pdbind.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --affinity-compare-features \
  --load-hyperparams output/best_hyperparams.json \
  --load-hyperparams-fp output/best_hyperparams_fp.json
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--load-csv TRAIN TEST [VAL]` | — | Load pre-computed pose CSVs instead of running Vina. TEST should be the CASF-2016 set. If VAL is omitted, a val set is carved from TRAIN using `--val-frac`. |
| `--val-frac F` | `0.1` | Fraction of non-CASF complexes to hold out as validation during the initial Vina run, or when VAL CSV is not provided with `--load-csv`. |
| `--num-complexes N` | `10` | Max number of non-CASF PDBbind 2020 complexes to process when running Vina (CASF-2016 is always processed in full). |
| `--casf-dir PATH` | `CASF-2016/coreset` | Path to the CASF-2016 coreset directory. |
| `--no-plots` | off | Skip all plot generation. |
| `--no-augment` | off | Skip mol2 descriptor and PDBQT geometry feature augmentation (faster but lower accuracy). |
| `--no-contact-features` | off | Skip protein–ligand contact feature computation (KDTree-based). |
| `--no-affinity` | off | Skip binding affinity prediction entirely. |
| `--optimize-hyperparams` | off | Run Optuna hyperparameter search for selected models. |
| `--n-trials N` | `30` | Optuna trials per model. |
| `--optuna-db PATH` | in-memory | SQLite file for persisting/resuming Optuna trials (e.g. `output/optuna_studies.db`). |
| `--optimize-models M [M ...]` | all models | Restrict Optuna to specific models. See list below. |
| `--load-hyperparams PATH` | — | Load current-feature hyperparameters from JSON (e.g. `output/best_hyperparams.json`). |
| `--load-hyperparams-fp PATH` | — | Load fingerprint-model hyperparameters from JSON (e.g. `output/best_hyperparams_fp.json`). Used with `--affinity-compare-features`. |
| `--affinity-compare-features` | off | Train all affinity models twice — once on current features, once on Morgan fingerprints only — and print a side-by-side comparison table. |

#### `--optimize-models` choices

| Key | Description |
|---|---|
| `rf_ranker` | Random Forest pose ranker |
| `gb_ranker` | Gradient Boosting pose ranker |
| `xgb_ranker` | XGBoost pose ranker |
| `rf_affinity` | RF affinity regressor (current features) |
| `gb_affinity` | GB affinity regressor (current features) |
| `svr_affinity` | SVR affinity regressor (current features) |
| `xgb_affinity` | XGBoost affinity regressor (current features) |
| `ridge_affinity` | Ridge regression affinity (current features) |
| `mlp_affinity` | MLP affinity regressor (current features) |
| `rf_affinity_fp` | RF affinity regressor (fingerprints only) |
| `gb_affinity_fp` | GB affinity regressor (fingerprints only) |
| `svr_affinity_fp` | SVR affinity regressor (fingerprints only) |
| `xgb_affinity_fp` | XGBoost affinity regressor (fingerprints only) |
| `ridge_affinity_fp` | Ridge regression affinity (fingerprints only) |
| `mlp_affinity_fp` | MLP affinity regressor (fingerprints only) |

### Features used by process_pdbind.py

| Group | Count | Features |
|---|---|---|
| Vina score features | 8 | score, rank, diffs, z-score, num_poses, range, log_score, score/atom |
| PDBQT geometry | 4 | radius of gyration, COM drift, max extent, volume |
| RDKit mol2 descriptors | 13 | MW, logP, TPSA, HBD/A, rotatable bonds, rings, etc. |
| Engineered features | 5 | Boltzmann probability, score gaps, Boltzmann weight |
| Protein–ligand contacts (basic) | 7 | contact counts at 3/4/5 Å, min dist, LJ score, buried frac, contacts/atom |
| Protein–ligand contacts (typed) | 5 | hydrophobic, H-bond, aromatic, Gaussian score, normalised H-bond |

**Total: 42 features** (used for both pose ranking and affinity prediction)

### Output files

| File | Description |
|---|---|
| `output/training_data.csv` | Per-pose training features |
| `output/val_data.csv` | Per-pose validation features |
| `output/test_data.csv` | Per-pose test features |
| `output/best_hyperparams.json` | Best hyperparameters (current features) |
| `output/best_hyperparams_fp.json` | Best hyperparameters (fingerprints; with `--affinity-compare-features`) |
| `output/svr_affinity_fp_model.joblib` | Trained SVR+fingerprints pipeline (with `--affinity-compare-features`) |
| `output/svr_affinity_fp_metadata.json` | Fingerprint column names and bit count for the SVR model |
| `output/svr_affinity_fp_pose_model.joblib` | Pose-aware SVR (with `--affinity-compare-features --pose-aware-affinity`) |
| `output/svr_affinity_fp_pose_metadata.json` | Fingerprint + pose feature metadata for the pose-aware SVR |
| `output/pose_selection_metrics.txt` | Pose ranking metrics for all models |
| `output/score_comparison.csv` | Per-pose scores from all rankers |
| `output/affinity_predictions.csv` | Per-complex predicted vs experimental ΔG |
| `output/affinity_predictions_*.png` | Affinity scatter plots (test/val, each model) |
| `output/rmsd_distribution.png` | RMSD histogram + boxplot |
| `output/success_vs_threshold.png` | Success rate vs RMSD threshold (all models) |
| `output/rmsd_cdf.png` | RMSD CDF (all models) |
| `output/spearman_distribution.png` | Per-complex Spearman correlation distribution |
| `output/feature_importance.png` | RF/XGB feature importance |

---

## gnn_affinity.py

Graph Neural Network for binding affinity prediction using **PyTorch Geometric**.
Each ligand is represented as a molecular graph (atoms = nodes, bonds = edges).

### Typical usage

```bash
# Basic run — default hyperparameters, auto-selects GPU/MPS/CPU
python gnn_affinity.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv

# With Optuna hyperparameter search (3-fold CV, 50 epochs/fold, ~1–2 h on CPU)
python gnn_affinity.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --optimize-hyperparams --n-trials 30 \
  --optuna-db output/gnn_optuna.db

# Load saved hyperparameters from a previous run
python gnn_affinity.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --load-hyperparams output/gnn_best_hyperparams.json

# Force a specific device
python gnn_affinity.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --device cuda

# Longer training with more patience
python gnn_affinity.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --epochs 300 --patience 30
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--load-csv TRAIN TEST [VAL]` | required | CSVs produced by `process_pdbind.py`. VAL is optional; carved from TRAIN if omitted. |
| `--val-frac F` | `0.2` | Val fraction when VAL CSV is not provided. |
| `--optimize-hyperparams` | off | Run Optuna hyperparameter search (3-fold CV, max 50 epochs/fold). |
| `--n-trials N` | `30` | Optuna trials. |
| `--optuna-db PATH` | in-memory | SQLite file for Optuna persistence (e.g. `output/gnn_optuna.db`). |
| `--load-hyperparams PATH` | — | Load hyperparameters JSON from a previous run. |
| `--epochs N` | `200` | Maximum epochs for final model training. |
| `--patience N` | `20` | Early stopping patience for final training. |
| `--no-plots` | off | Skip all plot generation. |
| `--device` | `auto` | Device: `auto` (CUDA → MPS → CPU), `cuda`, `mps`, or `cpu`. |

### GNN architecture

```
Ligand mol2 file
    └─► RDKit molecular graph
            ├─ Nodes (atoms):  28-dim feature vector
            │       atom type (10) · degree (8) · formal charge (1)
            │       implicit Hs (1) · aromatic (1) · in ring (1)
            │       hybridisation (6)
            └─ Edges (bonds):  7-dim feature vector
                    bond type (5) · conjugated (1) · in ring (1)
                        ↓
            Input projection (Linear → hidden_dim)
                        ↓
            n_layers × GINEConv + BatchNorm + ReLU + residual
                        ↓
            Global mean pool + Global max pool → concat
                        ↓
            MLP head → predicted ΔG (kcal/mol)
```

**Optuna search space:**

| Hyperparameter | Choices |
|---|---|
| `hidden_dim` | 64, 128, 256 |
| `n_layers` | 2, 3, 4 |
| `dropout` | 0.0, 0.1, 0.2, 0.3 |
| `lr` | 1e-4, 5e-4, 1e-3, 5e-3 |
| `batch_size` | 32, 64, 128 |

### Output files

| File | Description |
|---|---|
| `output/gnn_model.pt` | GINEConv GNN weights + hyperparameters |
| `output/gnn_pose_model.pt` | Pose-aware GINEConv GNN weights (with `--pose-aware`) |
| `output/gnn_best_hyperparams.json` | Best Optuna hyperparameters for GINEConv GNN |
| `output/gnn_pose_best_hyperparams.json` | Best Optuna hyperparameters for pose-aware GNN |
| `output/gnn_affinity_predictions.csv` | Per-complex predicted vs experimental ΔG (test set) |
| `output/gnn_affinity_predictions_test.png` | Scatter plot — test set |
| `output/gnn_affinity_predictions_val.png` | Scatter plot — validation set |
| `output/gnn_optuna_trials.png` | Optuna CV MSE vs trial number |
| `output/dimenet_model.pt` | DimeNet++ weights + hyperparameters + protein feature metadata |
| `output/dimenet_best_hyperparams.json` | Best Optuna hyperparameters for DimeNet++ |
| `output/dimenet_affinity_predictions.csv` | Per-complex DimeNet++ predicted vs experimental ΔG |
| `output/dimenet_affinity_predictions_test.png` | DimeNet++ scatter plot — test set |
| `output/dimenet_affinity_predictions_val.png` | DimeNet++ scatter plot — validation set |
| `output/dimenet_optuna_trials.png` | DimeNet++ Optuna CV MSE vs trial number |

---

## predict_affinity.py

Standalone inference script for integrating trained models into a drug discovery
workflow. Given Vina-docked poses for one or more novel ligands, predicts binding
affinity (ΔG, kcal/mol). No retraining or PDBbind data required at inference time.

Five models are supported:
- **`svr_fp`** / **`gnn`** — per-ligand: all poses of the same ligand receive the
  same predicted ΔG (models use molecular structure only).
- **`svr_fp_pose`** / **`gnn_pose`** — pose-aware: trained on all Vina poses with
  the Vina score as an additional feature; predictions differ per pose.
- **`dimenet`** — DimeNet++ GNN with protein augmentation: uses 3D atom coordinates
  and 12 protein–ligand contact scalar features; inherently pose-aware.

> See **[PREDICT_AFFINITY_GUIDE.md](PREDICT_AFFINITY_GUIDE.md)** for full
> instructions: prerequisites, input formats, and troubleshooting.

### Quick start

```bash
# 1. Train and save models (only needed once)
python process_pdbind.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --affinity-compare-features              # saves svr_affinity_fp_model.joblib

python process_pdbind.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --affinity-compare-features --pose-aware-affinity   # saves svr_affinity_fp_pose_model.joblib

python gnn_affinity.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv
  # saves gnn_model.pt

python gnn_affinity.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --pose-aware                              # saves gnn_pose_model.pt

python gnn_dimenet_affinity.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv
  # saves dimenet_model.pt (requires contact features from --augment)

# 2. Create a manifest CSV (one row per ligand)
# ligand_id,ligand_mol2,protein_pdb,vina_pdbqt
# lig1,inputs/lig1.mol2,inputs/protein.pdb,inputs/lig1_poses.pdbqt

# 3. Per-ligand prediction (same ΔG for all poses of a ligand)
python predict_affinity.py --manifest inputs/manifest.csv --model svr_fp --output predictions.csv
python predict_affinity.py --manifest inputs/manifest.csv --model gnn    --output predictions.csv

# 4. Pose-aware prediction (different ΔG per pose)
python predict_affinity.py --manifest inputs/manifest.csv --model svr_fp_pose --output predictions.csv
python predict_affinity.py --manifest inputs/manifest.csv --model gnn_pose    --output predictions.csv
python predict_affinity.py --manifest inputs/manifest.csv --model dimenet     --output predictions.csv
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--manifest PATH` | required | Manifest CSV: `ligand_id`, `ligand_mol2`, `protein_pdb`, `vina_pdbqt` |
| `--model` | required | `svr_fp`, `gnn` (per-ligand) or `svr_fp_pose`, `gnn_pose`, `dimenet` (pose-aware) |
| `--svr-model PATH` | auto | SVR joblib file (auto-resolved by model type) |
| `--svr-meta PATH` | auto | SVR metadata JSON (auto-resolved by model type) |
| `--gnn-model PATH` | auto | GNN/DimeNet checkpoint (auto-resolved by model type) |
| `--device` | `auto` | GNN device: `auto`, `cuda`, `mps`, or `cpu` |
| `--output PATH` | `predictions.csv` | Output CSV |

### Output

One row per (ligand, pose):

| Column | Description |
|---|---|
| `ligand_id` | From the manifest |
| `pose_idx` | Vina pose number (1-based) |
| `vina_score` | Vina score for this pose (kcal/mol) |
| `predicted_affinity_kcal_mol` | ML-predicted ΔG (kcal/mol) |
| `model` | Model identifier (e.g. `svr_fp`, `gnn`, `dimenet`) |

---

## vina_docking.py

Standalone AutoDock Vina docking script designed for HPC SLURM job arrays.
Separates the compute-intensive docking step from feature extraction and ML
training, so large datasets can be parallelised across many nodes.

**Key features:**
- Each SLURM task processes a non-overlapping slice of complexes (`--start` / `--count`)
- Runs `--cpus` Vina instances simultaneously; each Vina call uses exactly 1 CPU
  (`--cpu 1` passed to Vina), so CPU allocation is predictable
- Protein and ligand PDBQT files are written to the shared `output/` directory and
  skipped on re-run if they already exist (safe restarts after node failures)
- Each task writes `output/vina_batch_{start:06d}.csv`; batches are concatenated
  after all tasks finish and fed to `process_pdbind.py --load-csv`

### SLURM job array example

```bash
#!/bin/bash
#SBATCH --job-name=vina_array
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/vina_%A_%a.out

module purge
module load cesga/2020 python/3.10.8 autodock_vina/1.1.2_linux_x86
source ~/vina_env/bin/activate

cd $LUSTRE/vina_project
mkdir -p logs output

# Task 0 → complexes 0–999, Task 1 → 1000–1999, …
STEP=1000
START_IDX=$((SLURM_ARRAY_TASK_ID * STEP))

python vina_docking.py \
    --start $START_IDX \
    --count $STEP \
    --cpus $SLURM_CPUS_PER_TASK
```
to run: sbatch --array=0-19%10 submit_vina_array.sh (or something similar)

### Post-processing (after all array tasks complete)

```bash
# Combine all batch CSVs into a single training file
python - <<'EOF'
import pandas as pd, glob
dfs = [pd.read_csv(f) for f in sorted(glob.glob('output/vina_batch_*.csv'))]
pd.concat(dfs, ignore_index=True).to_csv('output/training_data.csv', index=False)
print(f"Combined {len(dfs)} batches → output/training_data.csv")
EOF

# Run CASF-2016 test set (optional, single job)
python vina_docking.py --casf --cpus 8

# Feature extraction + ML on the combined data
python process_pdbind.py \
    --load-csv output/training_data.csv output/casf_results.csv
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--start N` | 0 | Start index into the sorted complex list |
| `--count N` | 1000 | Number of complexes this task processes |
| `--cpus N` | 8 | Parallel Vina workers; should match `--cpus-per-task` |
| `--exhaustiveness N` | 8 | Vina exhaustiveness parameter |
| `--vina-executable PATH` | VINA_EXECUTABLE | Override Vina binary location |
| `--casf` | (flag) | Run CASF-2016 test set; output → `output/casf_results.csv` |
| `--output-dir PATH` | `output/` | Shared output directory for PDBQT and result CSVs |

---

## Data / Directory Structure

```
AItenea_Docking/
├── process_pdbind.py          # Main ML pipeline
├── gnn_affinity.py            # GNN affinity model
├── plan.md                    # Development plan and progress
├── PDBind_2020/               # PDBbind 2020 full general set (~19,000 complexes)
│   ├── 1981-2000/
│   │   └── {pdb_code}/
│   │       ├── {pdb_code}_ligand.mol2
│   │       └── {pdb_code}_protein.pdb
│   ├── 2001-2010/{pdb_code}/...
│   └── 2011-2019/{pdb_code}/...
├── PDBind_2020_index/
│   └── INDEX_general_PL.2020R1.lst   # PDBbind affinity labels
├── CASF-2016/
│   └── coreset/               # Fixed benchmark test set (285 complexes)
│       └── {pdb_code}/
│           ├── {pdb_code}_ligand.mol2
│           └── {pdb_code}_protein.pdb
└── output/
    ├── training_data.csv      # Generated by initial Vina run; reused with --load-csv
    ├── val_data.csv
    ├── test_data.csv          # CASF-2016 complexes
    ├── best_hyperparams.json
    ├── best_hyperparams_fp.json
    ├── gnn_best_hyperparams.json
    ├── gnn_model.pt
    └── *.png
```

---

## Recommended workflow

```bash
# 1. Run Vina on all complexes and generate fixed CSVs (first time only — slow)
#    CASF-2016 → test_data.csv; PDBind_2020 non-CASF → training_data.csv + val_data.csv
python process_pdbind.py --num-complexes 19000

# 2. Fast iteration on saved CSVs (no Vina, no augmentation re-run)
python process_pdbind.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv

# 3. Optimise hyperparameters for all models, save results
python process_pdbind.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --optimize-hyperparams --n-trials 30 --optuna-db output/optuna_studies.db

# 4. Re-run with saved hyperparameters (no Optuna overhead)
python process_pdbind.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --load-hyperparams output/best_hyperparams.json

# 5. Run GNN affinity model
python gnn_affinity.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv

# 6. Optimise GNN hyperparameters
python gnn_affinity.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --optimize-hyperparams --n-trials 30 --optuna-db output/gnn_optuna.db
```
