# Binding Affinity Prediction Guide

`predict_affinity.py` is a standalone inference script for predicting binding
affinity (ΔG, kcal/mol) of novel protein–ligand pairs using pre-trained models
from the AItenea Docking pipeline.

---

## Overview

Five models are available, split into two categories:

### Per-ligand models (same ΔG for every pose of a ligand)

| Model | Key | Input |
|---|---|---|
| SVR + Morgan fingerprints | `svr_fp` | ECFP4 fingerprint (2048 bits) from ligand mol2 |
| Graph Neural Network | `gnn` | Molecular graph (atoms + bonds) from ligand mol2 |

These models use only molecular structure — not pose coordinates — so all
docking poses of the same ligand receive the same predicted ΔG. The output
includes one row per (ligand, pose) so predictions sit alongside Vina scores.

### Pose-aware models (different ΔG per pose)

| Model | Key | Input |
|---|---|---|
| Pose-aware SVR + fingerprints | `svr_fp_pose` | Fingerprint + Vina score per pose |
| Pose-aware GNN | `gnn_pose` | Molecular graph + Vina score per pose |
| DimeNet++ with protein features | `dimenet` | 3D ligand graph + protein contact scalars per pose |

The `svr_fp_pose` and `gnn_pose` models are trained on **all** Vina poses (not
just rank-1) and append the per-pose Vina score to the molecular representation.
Predictions therefore differ between poses of the same ligand.

The `dimenet` model uses a different mechanism: it passes the 3D ligand structure
through a **DimeNet++** graph neural network (direction-aware message passing on
3D atom coordinates), then concatenates 12 protein–ligand contact scalar features
with the resulting embedding before the MLP head. Because contact features are
computed per docking pose from the protein PDB + pose PDBQT coordinates,
predictions are inherently pose-specific without adding an explicit Vina-score
feature.

---

## Prerequisites

Activate the project conda environment before running any command:

```bash
conda activate docking_aitenea
```

Required packages (all present in `docking_aitenea`):

| Package | Used by |
|---|---|
| `rdkit` | fingerprint + graph construction |
| `scikit-learn` | SVR pipeline |
| `joblib` | loading the saved SVR model (ships with scikit-learn) |
| `numpy`, `pandas` | data handling |
| `torch`, `torch-geometric` | GNN inference only |

---

## Step 1: Train and Save Models

The prediction script loads pre-trained weights, so you must train the models
first using the main pipeline.

### SVR + fingerprints (`svr_fp`)

```bash
python process_pdbind.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --affinity-compare-features
```

Saves:
- `output/svr_affinity_fp_model.joblib` — trained SVR pipeline (StandardScaler → SVR)
- `output/svr_affinity_fp_metadata.json` — fingerprint column names and bit count

With optimised hyperparameters:

```bash
python process_pdbind.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --affinity-compare-features \
  --load-hyperparams-fp output/best_hyperparams_fp.json
```

### Pose-aware SVR (`svr_fp_pose`)

```bash
python process_pdbind.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --affinity-compare-features --pose-aware-affinity
```

Saves:
- `output/svr_affinity_fp_pose_model.joblib` — pose-aware SVR pipeline
- `output/svr_affinity_fp_pose_metadata.json` — fingerprint info + pose feature names

### GNN (`gnn`)

```bash
python gnn_affinity.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv
```

Saves `output/gnn_model.pt` (weights + hyperparameters).

With optimised hyperparameters:

```bash
python gnn_affinity.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --load-hyperparams output/gnn_best_hyperparams.json
```

### Pose-aware GNN (`gnn_pose`)

```bash
python gnn_affinity.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --pose-aware
```

Saves `output/gnn_pose_model.pt` (includes `pose_feature_cols` metadata).

With optimised hyperparameters:

```bash
python gnn_affinity.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --pose-aware --load-hyperparams output/gnn_pose_best_hyperparams.json
```

### DimeNet++ with protein features (`dimenet`)

**Prerequisite:** the training CSV must contain protein–ligand contact feature
columns. These are added by running `process_pdbind.py` with `--augment` before
training. The 12 contact columns are:
`contact_n_3A`, `contact_n_4A`, `contact_n_5A`, `contact_min_dist`,
`contact_score_lj`, `contact_buried_frac`, `contact_n_per_atom`,
`contact_n_hydrophobic`, `contact_n_hbond`, `contact_n_aromatic`,
`contact_score_gaussian`, `contact_hbond_normalized`.

**Also requires:** `torch-cluster`, `torch-sparse`, and `torch-scatter`. Install
them from the PyG wheels matching your PyTorch version (e.g. torch 2.10.0):

```bash
pip install torch-cluster torch-sparse torch-scatter \
    --find-links https://data.pyg.org/whl/torch-2.10.0+cpu.html
```

```bash
python gnn_dimenet_affinity.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv
```

Saves:
- `output/dimenet_model.pt` — trained DimeNet++ weights + metadata

With optimised hyperparameters:

```bash
python gnn_dimenet_affinity.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --load-hyperparams output/dimenet_best_hyperparams.json
```

Without protein features (ligand-only DimeNet++):

```bash
python gnn_dimenet_affinity.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --no-protein-features
```

---

## Step 2: Prepare Input Files

### Ligand mol2 files

Each ligand must have a mol2 file (same format used in PDBbind). The mol2 file
provides explicit bond information needed for fingerprint computation and graph
construction. Vina PDBQT output files do not contain bond types and cannot be
used alone.

If you have an SDF or SMILES file, convert with OpenBabel:

```bash
obabel ligand.sdf -O ligand.mol2
obabel -:CCO -omol2 -O ethanol.mol2   # from SMILES
```

### Vina PDBQT output files

Run AutoDock Vina normally for each ligand. The PDBQT output file must contain
standard `REMARK VINA RESULT:` lines, which Vina writes automatically:

```
REMARK VINA RESULT:    -8.500      0.000      0.000
```

### Manifest CSV

Create a CSV file with one row per ligand:

```csv
ligand_id,ligand_mol2,protein_pdb,vina_pdbqt
lig1,inputs/lig1.mol2,inputs/proteinA.pdb,inputs/lig1_poses.pdbqt
lig2,inputs/lig2.mol2,inputs/proteinA.pdb,inputs/lig2_poses.pdbqt
lig3,inputs/lig3.mol2,inputs/proteinB.pdb,inputs/lig3_poses.pdbqt
```

| Column | Description |
|---|---|
| `ligand_id` | Unique name for this ligand (used in the output) |
| `ligand_mol2` | Path to the ligand mol2 file (absolute or relative to the manifest) |
| `protein_pdb` | Path to the protein PDB file (stored in output for reference) |
| `vina_pdbqt` | Path to the Vina multi-pose output PDBQT file |

Paths can be absolute or relative to the manifest file's directory.

Multiple ligands can share the same protein (docking against the same target).

---

## Step 3: Run Predictions

### Per-ligand models (same ΔG for all poses of one ligand)

```bash
# SVR + fingerprints (fast, CPU, no GPU needed)
python predict_affinity.py \
    --manifest inputs/manifest.csv \
    --model svr_fp \
    --output predictions_svr.csv

# GNN
python predict_affinity.py \
    --manifest inputs/manifest.csv \
    --model gnn \
    --output predictions_gnn.csv
```

### Pose-aware models (different ΔG per pose)

```bash
# Pose-aware SVR (fast, CPU)
python predict_affinity.py \
    --manifest inputs/manifest.csv \
    --model svr_fp_pose \
    --output predictions_svr_pose.csv

# Pose-aware GNN
python predict_affinity.py \
    --manifest inputs/manifest.csv \
    --model gnn_pose \
    --output predictions_gnn_pose.csv

# DimeNet++ with protein features (pose-aware by nature)
python predict_affinity.py \
    --manifest inputs/manifest.csv \
    --model dimenet \
    --output predictions_dimenet.csv
```

Model file paths default to the standard output directory names. Override with
`--svr-model`, `--svr-meta`, or `--gnn-model` when using non-default paths.

Force a specific device for GNN inference:

```bash
python predict_affinity.py --manifest inputs/manifest.csv --model gnn_pose \
    --device cpu --output predictions_gnn_pose.csv
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--manifest PATH` | required | Manifest CSV file |
| `--model` | required | `svr_fp`, `gnn`, `svr_fp_pose`, `gnn_pose`, or `dimenet` |
| `--svr-model PATH` | auto | SVR joblib file (auto-selected by model type) |
| `--svr-meta PATH` | auto | SVR metadata JSON (auto-selected by model type) |
| `--gnn-model PATH` | auto | GNN/DimeNet checkpoint (auto-selected by model type) |
| `--device` | `auto` | GNN device: `auto`, `cuda`, `mps`, or `cpu` |
| `--output PATH` | `predictions.csv` | Output CSV path |

---

## Output

The output CSV has one row per (ligand, pose):

| Column | Description |
|---|---|
| `ligand_id` | From the manifest |
| `pose_idx` | Vina pose number (1-based) |
| `vina_score` | Vina binding energy (kcal/mol) for this pose |
| `predicted_affinity_kcal_mol` | ML-predicted ΔG (kcal/mol) |
| `model` | Model used (`svr_fp` or `gnn`) |

Example output:

```csv
ligand_id,pose_idx,vina_score,predicted_affinity_kcal_mol,model
lig1,1,-8.5,-7.24,svr_fp
lig1,2,-8.2,-7.24,svr_fp
lig1,3,-7.9,-7.24,svr_fp
lig2,1,-9.1,-8.41,svr_fp
lig2,2,-8.8,-8.41,svr_fp
```

**For per-ligand models** (`svr_fp`, `gnn`): `predicted_affinity_kcal_mol` is the
same for all poses of the same ligand. Use the Vina score to distinguish poses;
use the ML prediction to compare ligands against each other.

**For pose-aware models** (`svr_fp_pose`, `gnn_pose`): `predicted_affinity_kcal_mol`
differs between poses of the same ligand because the Vina score is included as an
input feature. Better-docked poses (lower Vina score) typically receive lower
(more favorable) predicted ΔG values.

**Typical range for drug-like binders:** −4 to −12 kcal/mol. More negative
values indicate stronger predicted binding.

---

## Choosing a Model

### Should I use per-ligand or pose-aware?

Use **per-ligand** (`svr_fp`, `gnn`) when you want to compare ligands against each
other and are not concerned with which pose of each ligand is most favorable.
All poses of a ligand receive the same score, so you are effectively ranking
ligands.

Use **pose-aware** (`svr_fp_pose`, `gnn_pose`, `dimenet`) when you want to rank
poses of the same ligand (e.g., to select the best docking pose for downstream
analysis) or when you want a prediction that accounts for how well the ligand
fits the binding site.

### SVR + fingerprints (`svr_fp`)

- No GPU required; runs in seconds even for thousands of ligands
- Good choice for lead optimisation where analogues share a scaffold

### Pose-aware SVR (`svr_fp_pose`)

- Same speed as `svr_fp`; trained on all Vina poses with Vina score appended
- Predictions differ per pose — the Vina score has a strong influence
- Best when you want to both rank ligands and distinguish poses

### GNN (`gnn`)

- Operates on the full molecular graph (all atoms + bond types)
- May generalise better to novel scaffolds not seen during training
- Benefits from GPU (MPS on Apple Silicon, CUDA on NVIDIA)

### Pose-aware GNN (`gnn_pose`)

- Molecular graph + Vina score concatenated at the readout layer
- Useful when scaffold diversity is high
- Requires GPU for practical speed

### DimeNet++ with protein features (`dimenet`)

- Uses 3D atom coordinates from mol2 (direction-aware message passing)
- Captures molecular geometry that 2D graph models cannot
- Protein–ligand contact scalars provide binding-site context
- Pose-specific by nature: contact features differ for each docked pose
- Requires `torch-cluster`, `torch-sparse`, `torch-scatter` (see Step 1)
- Best when 3D geometry and protein context are important
- Requires training data with `--augment` contact features; falls back to
  ligand-only mode if contact columns are absent

When in doubt, run both `svr_fp_pose` and `dimenet` and compare. Consistent
predictions between models increase confidence in the ranking.

---

## Troubleshooting

**`Error: model file not found: output/svr_affinity_fp_model.joblib`**
Run `process_pdbind.py` with `--affinity-compare-features` first (see Step 1).

**`Warning: could not parse mol2 for <lig> — skipping`**
The mol2 file is missing, empty, or RDKit could not parse it. Check the file
path in the manifest and verify the mol2 is valid. Try opening it in RDKit:
```python
from rdkit import Chem
mol = Chem.MolFromMol2File('ligand.mol2', removeHs=True)
print(mol)  # None means parse failed
```

**`Warning: no poses found in ... for <lig>`**
The PDBQT file exists but contains no `REMARK VINA RESULT:` lines. Ensure the
file was produced by AutoDock Vina (not a receptor or input PDBQT). The
`predicted_affinity_kcal_mol` will still be written for the ligand.

**GNN import errors (`torch`, `torch_geometric` not found)**
These are only needed for `--model gnn`. Install via:
```bash
conda activate docking_aitenea
pip install torch torch-geometric
```
Or use `--model svr_fp` if PyTorch is unavailable.

**`ImportError: 'radius_graph' requires 'torch-cluster'`** (DimeNet++ only)
DimeNet++ requires three extra PyG extensions. Install the wheels matching your
PyTorch version:
```bash
pip install torch-cluster torch-sparse torch-scatter \
    --find-links https://data.pyg.org/whl/torch-2.10.0+cpu.html
```
Replace `2.10.0` with your actual PyTorch version (`python -c "import torch; print(torch.__version__)"`).

**`Warning: protein feature columns not in CSV` during DimeNet training**
The training CSV does not contain contact feature columns. Re-run
`process_pdbind.py` with `--augment` to compute them, or use
`--no-protein-features` to train in ligand-only mode.
