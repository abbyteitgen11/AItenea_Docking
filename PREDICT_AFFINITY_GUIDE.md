# Binding Affinity Prediction Guide

`predict_affinity.py` is a standalone inference script for predicting binding
affinity (ΔG, kcal/mol) of novel protein–ligand pairs using pre-trained models
from the AItenea Docking pipeline.

---

## Overview

Two models are available:

| Model | Key | Input | Speed |
|---|---|---|---|
| SVR + Morgan fingerprints | `svr_fp` | ECFP4 fingerprint (2048 bits) from ligand mol2 | Fast (CPU) |
| Graph Neural Network | `gnn` | Molecular graph (atoms + bonds) from ligand mol2 | Moderate (GPU optional) |

**Both models are per-ligand.** They use only molecular structure — not
pose coordinates — so all docking poses of the same ligand receive the same
predicted ΔG. The output table includes one row per (ligand, pose) so the
predictions sit alongside Vina scores for easy comparison.

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

### SVR + fingerprints

```bash
python process_pdbind.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --affinity-compare-features
```

This trains affinity models on both feature sets and saves:
- `output/svr_affinity_fp_model.joblib` — trained SVR pipeline (StandardScaler → SVR)
- `output/svr_affinity_fp_metadata.json` — fingerprint column names and bit count

To use optimised hyperparameters (recommended):

```bash
python process_pdbind.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --affinity-compare-features \
  --load-hyperparams-fp output/best_hyperparams_fp.json
```

### GNN

```bash
python gnn_affinity.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv
```

This saves `output/gnn_model.pt` (weights + hyperparameters).

With optimised hyperparameters:

```bash
python gnn_affinity.py \
  --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
  --load-hyperparams output/gnn_best_hyperparams.json
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

### SVR + fingerprints (recommended for large sets, no GPU needed)

```bash
python predict_affinity.py \
    --manifest inputs/manifest.csv \
    --model svr_fp \
    --svr-model output/svr_affinity_fp_model.joblib \
    --svr-meta  output/svr_affinity_fp_metadata.json \
    --output predictions_svr.csv
```

### GNN

```bash
python predict_affinity.py \
    --manifest inputs/manifest.csv \
    --model gnn \
    --gnn-model output/gnn_model.pt \
    --output predictions_gnn.csv
```

Force a specific device for GNN:

```bash
python predict_affinity.py \
    --manifest inputs/manifest.csv \
    --model gnn \
    --gnn-model output/gnn_model.pt \
    --device cpu \
    --output predictions_gnn.csv
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--manifest PATH` | required | Manifest CSV file |
| `--model` | required | `svr_fp` or `gnn` |
| `--svr-model PATH` | `output/svr_affinity_fp_model.joblib` | SVR joblib file (svr_fp only) |
| `--svr-meta PATH` | `output/svr_affinity_fp_metadata.json` | SVR metadata JSON (svr_fp only) |
| `--gnn-model PATH` | `output/gnn_model.pt` | GNN checkpoint (gnn only) |
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

**Note:** `predicted_affinity_kcal_mol` is the same for all poses of the same
ligand — this is expected. The ML models predict affinity from molecular
structure, not from pose geometry. Use the Vina score to distinguish poses
within a ligand; use the ML prediction to compare ligands against each other.

**Typical range for drug-like binders:** −4 to −12 kcal/mol. More negative
values indicate stronger predicted binding.

---

## Choosing a Model

### SVR + fingerprints (`svr_fp`)

- No GPU required; runs in seconds even for thousands of ligands
- Uses 2048-bit Morgan ECFP4 fingerprints — captures local atomic environments
- Good choice for lead optimisation where analogues share a scaffold
- Requires `output/svr_affinity_fp_model.joblib` and `output/svr_affinity_fp_metadata.json`

### GNN (`gnn`)

- Operates on the full molecular graph (all atoms + bond types)
- May generalise better to novel scaffolds not seen during training
- Benefits from GPU (MPS on Apple Silicon, CUDA on NVIDIA)
- Requires `output/gnn_model.pt`

When in doubt, run both and compare. Consistent predictions between models
increase confidence.

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
