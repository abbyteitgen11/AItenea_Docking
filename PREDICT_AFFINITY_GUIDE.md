# Binding Affinity Prediction Guide

`predict_affinity.py` is a standalone **inference** script for predicting binding
affinity (ΔG, kcal/mol) of novel protein–ligand pairs. It loads a pre-trained,
final model checkpoint and runs predictions — it does **not** train anything.
You supply the checkpoint(s); by default the script looks in `checkpoints/`.

---

## Models

Two **pose-aware** models are supported (each pose of a ligand gets its own ΔG):

| Model | Key | Input | Checkpoint files |
|---|---|---|---|
| Pose-aware SVR + fingerprints | `svr_fp_pose` | ECFP4 fingerprint (2048 bits) + per-pose Vina score | `*_pose_model.joblib` **and** `*_pose_metadata.json` |
| DimeNet++ with protein features | `dimenet` | 3D ligand graph + 12 protein–ligand contact scalars per pose | `dimenet_model.pt` |

- **`svr_fp_pose`** appends the per-pose Vina score to the molecular fingerprint, so
  predictions differ between poses (the Vina score has a strong influence). It needs
  **both** files: the joblib pipeline and the metadata JSON (the JSON holds the
  fingerprint bit count and pose feature names).
- **`dimenet`** runs the 3D ligand structure through a DimeNet++ network and
  concatenates 12 protein–ligand contact scalars (computed per pose from the protein
  PDB + that pose's PDBQT coordinates) before the MLP head. Pose-specific by nature;
  the single `.pt` file contains weights + hyperparameters + protein-feature metadata.

---

## Prerequisites

**Uses Python version 3.13.5**

**For `svr_fp_pose` (no PyTorch needed):**
```bash
pip install numpy pandas scipy scikit-learn joblib rdkit
```

**Additional packages for `dimenet` (local CPU):**
```bash
pip install torch torch-geometric
# torch-cluster must match your installed torch version. Find it with:
#   python -c "import torch; print(torch.__version__)"
# then use the matching CPU wheel index, e.g. for torch 2.12.1:
pip install pyg-lib -f https://data.pyg.org/whl/torch-2.12.1+cpu.html
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.12.1+cpu.html
```
On a Mac, run
`--model dimenet` with `--device cpu` — DimeNet++ does **not** support Apple MPS. The
`svr_fp_pose` model needs none of the torch packages.

---

## Step 1: Prepare input files

### Ligand mol2 files
Each ligand needs a mol2 file (explicit bonds + 3D coordinates). Vina PDBQT output
alone cannot be used. Convert from SDF/SMILES with OpenBabel if needed:
```bash
obabel ligand.sdf -O ligand.mol2
obabel -:"CCO" -omol2 -O ethanol.mol2   # from SMILES
```

### Vina PDBQT output files
The multi-pose Vina output must contain standard `REMARK VINA RESULT:` lines:
```
REMARK VINA RESULT:    -8.500      0.000      0.000
```

### Protein PDB files (`dimenet` only)
DimeNet computes per-pose contact features from the protein structure, so a
`protein_pdb` column is required for `--model dimenet`. It is optional/ignored for
`svr_fp_pose`.

### Manifest CSV
One row per (ligand, protein, Vina output):
```csv
ligand_id,ligand_mol2,protein_pdb,vina_pdbqt
1a30,example_output/1a30_ligand.mol2,example_output/1a30_protein.pdb,example_output/1a30_vina_output.pdbqt
1bcu,example_output/1bcu_ligand.mol2,example_output/1bcu_protein.pdb,example_output/1bcu_vina_output.pdbqt
4f3c,example_output/4f3c_ligand.mol2,example_output/4f3c_protein.pdb,example_output/4f3c_vina_output.pdbqt
```

| Column | Description | Required by |
|---|---|---|
| `ligand_id` | Unique name for this ligand (appears in the output) | both |
| `ligand_mol2` | Path to the ligand mol2 file | both |
| `vina_pdbqt` | Path to the Vina multi-pose output PDBQT | both |
| `protein_pdb` | Path to the protein PDB file | `dimenet` only |

Paths may be absolute or **relative to the manifest file's directory**. The same
manifest (with all four columns) works for both models — `svr_fp_pose` simply ignores
`protein_pdb`.

Note the number of **poses per ligand can vary**: Vina returns up to `--num_modes` poses but sometimes
fewer (here `1a30` and `1bcu` have 5, `4f3c` has 1). The output simply has one row per
available pose, so both models work the same regardless of pose count.

---

## Step 2: Run predictions

```bash
# Pose-aware SVR + fingerprints (fast, CPU)
python predict_affinity.py \
    --manifest example_manifest.csv \
    --model svr_fp_pose \
    --svr-model checkpoints/svr_affinity_fp_pose_model.joblib \
    --svr-meta  checkpoints/svr_affinity_fp_pose_metadata.json \
    --output predictions_svr_pose.csv

# DimeNet++ with protein features (use cpu or cuda)
python predict_affinity.py \
    --manifest example_manifest.csv \
    --model dimenet \
    --dimenet-model checkpoints/dimenet_model.pt \
    --device cpu \
    --output predictions_dimenet.csv
```

The `--svr-model`/`--svr-meta`/`--dimenet-model` paths default to the `checkpoints/`
names shown above, so you can omit them if your files are in `checkpoints/` with the
standard names.

### All flags

| Flag | Default | Description |
|---|---|---|
| `--manifest PATH` | required | Manifest CSV file |
| `--model` | required | `svr_fp_pose` or `dimenet` |
| `--svr-model PATH` | `checkpoints/svr_affinity_fp_pose_model.joblib` | Pose-aware SVR joblib checkpoint |
| `--svr-meta PATH` | `checkpoints/svr_affinity_fp_pose_metadata.json` | Pose-aware SVR metadata JSON |
| `--dimenet-model PATH` | `checkpoints/dimenet_model.pt` | DimeNet++ `.pt` checkpoint |
| `--device` | `auto` | DimeNet device: `auto`, `cuda`, `mps`*, or `cpu` (*MPS unsupported by DimeNet++) |
| `--output PATH` | `predictions.csv` | Output CSV path |

---

## Output

One row per (ligand, pose):

| Column | Description |
|---|---|
| `ligand_id` | From the manifest |
| `pose_idx` | Vina pose number (1-based) |
| `vina_score` | Vina binding energy (kcal/mol) for this pose |
| `predicted_affinity_kcal_mol` | ML-predicted ΔG (kcal/mol) |
| `model` | `svr_fp_pose` or `dimenet` |

```csv
ligand_id,pose_idx,vina_score,predicted_affinity_kcal_mol,model
lig1,1,-8.5,-7.94,svr_fp_pose
lig1,2,-8.2,-7.71,svr_fp_pose
lig2,1,-9.1,-8.63,svr_fp_pose
```

Both models give a **different ΔG per pose** (more negative = stronger predicted binding). The script
also prints the best (most negative) predicted ΔG per ligand.

---

## Note: HPC / GPU installation (`dimenet` with CUDA)

The local instructions above install CPU builds. To run `--model dimenet` on a GPU
cluster, install CUDA-matched wheels instead — the versions of `torch`, `torch-cluster`,
and the wheel index's `cuXXX` tag must all agree:

```bash
# Example for torch 2.3.0 + CUDA 12.1 (match to the CUDA module available on your cluster):
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

Then run with `--device cuda`
