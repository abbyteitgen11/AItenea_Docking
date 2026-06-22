"""
predict_affinity.py — Standalone binding affinity prediction for new ligands.

Loads a pre-trained, final model checkpoint and predicts binding affinity
(ΔG, kcal/mol) for Vina-docked poses of novel ligands. This script performs
inference only — it does not train anything. Supply the checkpoint(s) produced by
the training pipeline (by default it looks in checkpoints/).

Input:  a manifest CSV listing ligand mol2 files, Vina PDBQT outputs, and
        (for DimeNet) protein PDB files.
Output: a CSV with one row per (ligand, pose) containing the Vina score and the
        ML-predicted ΔG. Both supported models are pose-aware, so each pose of a
        ligand gets its own prediction.

Models
──────
  svr_fp_pose  Pose-aware SVR + Morgan fingerprints. Trained on all Vina poses with
               the Vina score as an extra feature; produces a different ΔG per pose.
               Checkpoint: a joblib pipeline + a metadata JSON.
  dimenet      DimeNet++ GNN with protein-feature augmentation. Uses 3D ligand
               coordinates and 12 protein–ligand contact scalars computed per pose;
               produces a different ΔG per pose. Checkpoint: a single .pt file.

Usage:
    # Pose-aware SVR + fingerprints
    python predict_affinity.py --manifest inputs/manifest.csv --model svr_fp_pose \\
        --svr-model checkpoints/svr_affinity_fp_pose_model.joblib \\
        --svr-meta  checkpoints/svr_affinity_fp_pose_metadata.json \\
        --output predictions_svr_pose.csv

    # DimeNet++ with protein features
    python predict_affinity.py --manifest inputs/manifest.csv --model dimenet \\
        --dimenet-model checkpoints/dimenet_model.pt \\
        --output predictions_dimenet.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial import KDTree


# ---------------------------------------------------------------------------
# Inlined structure / feature helpers (self-contained — no project imports)
# ---------------------------------------------------------------------------

def compute_morgan_fingerprint(mol2_path: Path, n_bits: int = 2048) -> Optional[np.ndarray]:
    """
    Compute Morgan ECFP4 fingerprint (radius=2) from a mol2 file.

    Returns:
        np.ndarray of shape (n_bits,) with dtype float32, or None if parsing fails.
    """
    mol = Chem.MolFromMol2File(str(mol2_path), removeHs=True)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def extract_pose_atoms(pdbqt_file: Path) -> List[Tuple[np.ndarray, List[str]]]:
    """
    Extract atom coordinates and AutoDock atom types for each pose from a Vina PDBQT file.

    Returns:
        List of (coords_array, atom_types_list) per pose; HD (donor hydrogen) atoms excluded.
    """
    poses: List[Tuple[np.ndarray, List[str]]] = []
    current_coords: List[List[float]] = []
    current_types: List[str] = []

    try:
        with open(pdbqt_file, 'r') as f:
            for line in f:
                line = line.rstrip()
                if line.startswith('MODEL'):
                    current_coords = []
                    current_types = []
                elif line.startswith('ENDMDL'):
                    if current_coords:
                        poses.append((np.array(current_coords, dtype=np.float32), current_types))
                    current_coords = []
                    current_types = []
                elif line.startswith('ATOM') or line.startswith('HETATM'):
                    # AutoDock atom type is the last whitespace-delimited token on the line
                    parts = line.split()
                    if not parts:
                        continue
                    adt_type = parts[-1]
                    # Skip donor hydrogens
                    if adt_type == 'HD':
                        continue
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        current_coords.append([x, y, z])
                        current_types.append(adt_type)
                    except (ValueError, IndexError):
                        continue
    except Exception:
        return poses

    if current_coords:
        poses.append((np.array(current_coords, dtype=np.float32), current_types))

    return poses


def load_protein_atoms(
    protein_pdb: Path,
) -> Optional[Tuple[np.ndarray, List[str], List[str]]]:
    """
    Parse a protein PDB file and return heavy-atom coordinates, elements, and residue names.

    Returns:
        Tuple of (coords array (N,3), elements list, resnames list), or None if parsing fails
    """
    coords, elements, resnames = [], [], []
    try:
        with open(protein_pdb, 'r') as f:
            for line in f:
                if not (line.startswith('ATOM') or line.startswith('HETATM')):
                    continue
                atom_name = line[12:16].strip()
                # Determine element: prefer element column (76-78), fall back to atom name
                element = ''
                if len(line) >= 78:
                    element = line[76:78].strip().upper()
                if not element:
                    # Infer from atom name: strip leading digits, take first letter
                    stripped = atom_name.lstrip('0123456789')
                    element = stripped[0].upper() if stripped else ''
                # Skip hydrogens
                if element == 'H':
                    continue
                if not element and (atom_name.startswith('H') or
                                    (len(atom_name) > 1 and atom_name[1] == 'H')):
                    continue
                resname = line[17:20].strip()
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                    elements.append(element)
                    resnames.append(resname)
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"    Warning: Could not parse protein PDB {protein_pdb}: {e}")
        return None

    if not coords:
        return None
    return np.array(coords, dtype=np.float32), elements, resnames


_POLAR_LIG_TYPES = {'OA', 'NA', 'N', 'O'}
_HYDROPHOBIC_LIG_TYPES = {'C', 'A'}


def compute_pose_contact_features(
    ligand_coords: np.ndarray,
    protein_tree: KDTree,
    ligand_atom_types: Optional[List[str]] = None,
    protein_hydrophobic_tree: Optional[KDTree] = None,
    protein_polar_tree: Optional[KDTree] = None,
    protein_aromatic_tree: Optional[KDTree] = None,
) -> Dict:
    """
    Compute protein–ligand contact features using a pre-built KDTree of protein atoms.

    Args:
        ligand_coords: (N_ligand, 3) array of ligand heavy-atom coordinates
        protein_tree:  KDTree built from all protein heavy-atom coordinates
        ligand_atom_types: AutoDock atom types per ligand atom (enables chemical features)
        protein_hydrophobic_tree: KDTree of protein C/S atoms
        protein_polar_tree: KDTree of protein N/O atoms
        protein_aromatic_tree: KDTree of protein atoms in aromatic residues

    Returns:
        Dictionary of contact feature values
    """
    all_base_keys = ['contact_n_3A', 'contact_n_4A', 'contact_n_5A',
                     'contact_min_dist', 'contact_score_lj',
                     'contact_buried_frac', 'contact_n_per_atom']
    all_rich_keys = ['contact_n_hydrophobic', 'contact_n_hbond', 'contact_n_aromatic',
                     'contact_score_gaussian', 'contact_hbond_normalized']
    n_lig = len(ligand_coords)
    if n_lig == 0:
        return {k: np.nan for k in all_base_keys + all_rich_keys}

    # Count contacts at different radii
    counts_5A = protein_tree.query_ball_point(ligand_coords, r=5.0, return_length=True)
    counts_4A = protein_tree.query_ball_point(ligand_coords, r=4.0, return_length=True)
    counts_3A = protein_tree.query_ball_point(ligand_coords, r=3.0, return_length=True)

    n_5A = int(counts_5A.sum())
    n_4A = int(counts_4A.sum())
    n_3A = int(counts_3A.sum())

    # Minimum protein–ligand distance per ligand atom
    min_dists, _ = protein_tree.query(ligand_coords, k=1)
    min_dist = float(min_dists.min())

    # Approximate attractive LJ term: Σ (1/r^6), capped at r > 0.5 Å to avoid singularity
    lj_score = float(np.sum(1.0 / np.maximum(min_dists, 0.5) ** 6))

    # Fraction of ligand atoms "buried" (at least one protein atom within 4.5 Å)
    counts_45A = protein_tree.query_ball_point(ligand_coords, r=4.5, return_length=True)
    buried_frac = float((counts_45A > 0).sum() / n_lig)

    result = {
        'contact_n_3A': n_3A,
        'contact_n_4A': n_4A,
        'contact_n_5A': n_5A,
        'contact_min_dist': min_dist,
        'contact_score_lj': lj_score,
        'contact_buried_frac': buried_frac,
        'contact_n_per_atom': n_4A / n_lig,
    }

    # --- Atom-type-aware features (only when type info is available) ---
    if ligand_atom_types is not None:
        lig_hphob_mask = np.array([t in _HYDROPHOBIC_LIG_TYPES for t in ligand_atom_types])
        lig_polar_mask  = np.array([t in _POLAR_LIG_TYPES       for t in ligand_atom_types])
        n_polar_lig = int(lig_polar_mask.sum())

        # Hydrophobic contacts
        contact_n_hydrophobic = np.nan
        if protein_hydrophobic_tree is not None and lig_hphob_mask.any():
            hphob_counts = protein_hydrophobic_tree.query_ball_point(
                ligand_coords[lig_hphob_mask], r=4.5, return_length=True)
            contact_n_hydrophobic = int(hphob_counts.sum())

        # H-bond contacts
        contact_n_hbond = np.nan
        if protein_polar_tree is not None and lig_polar_mask.any():
            hbond_counts = protein_polar_tree.query_ball_point(
                ligand_coords[lig_polar_mask], r=3.5, return_length=True)
            contact_n_hbond = int(hbond_counts.sum())
        else:
            contact_n_hbond = 0

        # Aromatic contacts
        contact_n_aromatic = np.nan
        if protein_aromatic_tree is not None:
            arom_counts = protein_aromatic_tree.query_ball_point(
                ligand_coords, r=5.0, return_length=True)
            contact_n_aromatic = int(arom_counts.sum())

        # Gaussian-weighted contact score (smooth version of shell counts)
        contact_score_gaussian = float(np.sum(np.exp(-min_dists ** 2 / 8.0)))

        # H-bonds normalised by polar ligand atom count
        contact_hbond_normalized = (float(contact_n_hbond) / max(n_polar_lig, 1)
                                    if not np.isnan(contact_n_hbond) else np.nan)

        result.update({
            'contact_n_hydrophobic':   contact_n_hydrophobic,
            'contact_n_hbond':         contact_n_hbond,
            'contact_n_aromatic':      contact_n_aromatic,
            'contact_score_gaussian':  contact_score_gaussian,
            'contact_hbond_normalized': contact_hbond_normalized,
        })
    else:
        result.update({k: np.nan for k in all_rich_keys})

    return result


# ---------------------------------------------------------------------------
# PDBQT parsing
# ---------------------------------------------------------------------------

def parse_pdbqt_scores(pdbqt_path: Path) -> List[Tuple[int, float]]:
    """
    Extract (pose_idx, vina_score) pairs from a Vina output PDBQT file.

    pose_idx is 1-based (matches Vina's MODEL numbering).
    Returns an empty list if the file is missing or contains no REMARK VINA lines.
    """
    if not pdbqt_path.exists():
        return []
    poses: List[Tuple[int, float]] = []
    pose_idx = 0
    with open(pdbqt_path, 'r') as fh:
        for line in fh:
            if line.startswith('MODEL'):
                pose_idx += 1
            elif line.startswith('REMARK VINA RESULT:'):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        score = float(parts[3])
                        poses.append((pose_idx, score))
                    except ValueError:
                        pass
    return poses


def _resolve_path(p: str, base_dir: Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return base_dir / path


def _pose_scalar_vec(pose_feature_cols: List[str], vina_score: float) -> np.ndarray:
    """Build a pose scalar vector from a feature column list and the current Vina score.

    Currently only 'vina_score' is supported as a pose feature col name; other columns
    are set to 0 (they require protein structure computation not available at inference
    without additional inputs).
    """
    vals = []
    for col in pose_feature_cols:
        if col == 'vina_score':
            vals.append(vina_score)
        else:
            vals.append(0.0)
    return np.array(vals, dtype=np.float32)


# ---------------------------------------------------------------------------
# Pose-aware SVR + fingerprints
# ---------------------------------------------------------------------------

def load_svr_fp(model_path: Path, meta_path: Path):
    """Load the joblib SVR pipeline and its fingerprint/pose metadata.

    Pose-aware metadata stores 'combined_feature_cols' and 'fp_n_bits'; the older
    per-ligand format used 'feature_cols'/'n_bits'. Accept either for robustness.
    """
    pipeline = joblib.load(model_path)
    meta = json.loads(meta_path.read_text())
    feature_cols: List[str] = meta.get('feature_cols') or meta.get('combined_feature_cols')
    n_bits: int = meta.get('n_bits') or meta.get('fp_n_bits', 2048)
    return pipeline, feature_cols, n_bits


def predict_svr_fp_pose(
    manifest_df: pd.DataFrame,
    pipeline,
    fp_n_bits: int,
    pose_feature_cols: List[str],
    manifest_dir: Path,
) -> pd.DataFrame:
    """
    Predict ΔG per (ligand, pose) using the pose-aware SVR+fingerprints model.

    For each ligand: fingerprint is computed once from mol2, then concatenated with
    per-pose Vina scalars to produce a distinct prediction for each pose.

    Returns a DataFrame with columns: ligand_id, pose_idx, vina_score,
    predicted_affinity_kcal_mol, model.
    """
    # Cache fingerprints per ligand
    fp_cache: Dict[str, Optional[np.ndarray]] = {}
    for row in manifest_df.itertuples(index=False):
        lid = row.ligand_id
        if lid not in fp_cache:
            mol2_path = _resolve_path(row.ligand_mol2, manifest_dir)
            fp_cache[lid] = compute_morgan_fingerprint(mol2_path, n_bits=fp_n_bits)
            if fp_cache[lid] is None:
                print(f"  Warning: could not parse mol2 for {lid} — skipping")

    rows = []
    fp_rows: List[np.ndarray] = []
    meta: List[dict] = []

    for row in manifest_df.itertuples(index=False):
        lid = row.ligand_id
        fp = fp_cache.get(lid)
        if fp is None:
            continue
        pdbqt_path = _resolve_path(row.vina_pdbqt, manifest_dir)
        poses = parse_pdbqt_scores(pdbqt_path)
        if not poses:
            print(f"  Warning: no poses in {pdbqt_path} for {lid}")
            continue
        for pose_idx, vina_score in poses:
            pose_vals = _pose_scalar_vec(pose_feature_cols, vina_score)
            fp_rows.append(np.concatenate([fp, pose_vals]))
            meta.append({'ligand_id': lid, 'pose_idx': pose_idx, 'vina_score': vina_score})

    if not fp_rows:
        return pd.DataFrame(columns=['ligand_id', 'pose_idx', 'vina_score',
                                     'predicted_affinity_kcal_mol', 'model'])

    X = np.nan_to_num(np.array(fp_rows, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    preds = pipeline.predict(X)
    for m, pred in zip(meta, preds):
        rows.append({**m, 'predicted_affinity_kcal_mol': float(pred), 'model': 'svr_fp_pose'})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DimeNet++ with protein feature augmentation
# ---------------------------------------------------------------------------

_HYDROPHOBIC_ELEMENTS_INF = {'C', 'S'}
_POLAR_ELEMENTS_INF = {'N', 'O'}
_AROMATIC_RESNAMES_INF = {'PHE', 'TRP', 'TYR', 'HIS', 'ARG'}


def compute_contact_feats_for_pose(
    protein_pdb: Path,
    pdbqt_path: Path,
    pose_idx: int,
) -> Optional[np.ndarray]:
    """
    Compute the 12 protein–ligand contact scalars for a single Vina pose.

    pose_idx is 1-based (matches Vina's MODEL numbering).
    Returns np.ndarray of shape (12,), or None if computation fails.
    """
    protein_result = load_protein_atoms(protein_pdb)
    if protein_result is None:
        return None
    protein_coords, prot_elements, prot_resnames = protein_result
    if len(protein_coords) == 0:
        return None

    pose_atoms_list = extract_pose_atoms(pdbqt_path)
    # pose_atoms_list is 0-based internally; pose_idx is 1-based
    idx0 = pose_idx - 1
    if idx0 >= len(pose_atoms_list) or idx0 < 0:
        return None
    lig_coords, lig_types = pose_atoms_list[idx0]
    if len(lig_coords) == 0:
        return None

    protein_tree = KDTree(protein_coords)

    hphob_mask = np.array([e in _HYDROPHOBIC_ELEMENTS_INF for e in prot_elements])
    polar_mask = np.array([e in _POLAR_ELEMENTS_INF for e in prot_elements])
    aromatic_mask = np.array([r in _AROMATIC_RESNAMES_INF for r in prot_resnames])

    prot_hphob_tree = KDTree(protein_coords[hphob_mask]) if hphob_mask.any() else None
    prot_polar_tree = KDTree(protein_coords[polar_mask]) if polar_mask.any() else None
    prot_arom_tree = KDTree(protein_coords[aromatic_mask]) if aromatic_mask.any() else None

    feats_dict = compute_pose_contact_features(
        lig_coords,
        protein_tree,
        ligand_atom_types=lig_types,
        protein_hydrophobic_tree=prot_hphob_tree,
        protein_polar_tree=prot_polar_tree,
        protein_aromatic_tree=prot_arom_tree,
    )

    ordered_keys = [
        'contact_n_3A', 'contact_n_4A', 'contact_n_5A',
        'contact_min_dist', 'contact_score_lj', 'contact_buried_frac',
        'contact_n_per_atom', 'contact_n_hydrophobic', 'contact_n_hbond',
        'contact_n_aromatic', 'contact_score_gaussian', 'contact_hbond_normalized',
    ]
    vals = [float(feats_dict.get(k, 0.0)) for k in ordered_keys]
    vals = [0.0 if (np.isnan(v) or np.isinf(v)) else v for v in vals]
    return np.array(vals, dtype=np.float32)


def _build_dimenet_affinity(**kwargs):
    """Construct the DimeNet++ affinity model. Defined lazily so torch/torch_geometric
    are only required for `--model dimenet` (svr_fp_pose stays torch-free)."""
    import torch.nn as nn
    from torch_geometric.data import Data
    from torch_geometric.nn import DimeNetPlusPlus

    class DimeNetAffinity(nn.Module):
        """
        DimeNet++ GNN for binding affinity regression with optional protein augmentation.

        Architecture:
            DimeNetPlusPlus (ligand 3D graph) → graph embedding (out_channels,)
            → cat([embedding, protein_feats]) if n_protein_features > 0
            → MLP: Linear → ReLU → Dropout → Linear → scalar ΔG
        """

        def __init__(
            self,
            hidden_channels: int = 128,
            out_channels: int = 128,
            num_blocks: int = 4,
            int_emb_size: int = 64,
            basis_emb_size: int = 8,
            out_emb_channels: int = 256,
            n_protein_features: int = 0,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.n_protein_features = n_protein_features
            self.gnn = DimeNetPlusPlus(
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                int_emb_size=int_emb_size,
                basis_emb_size=basis_emb_size,
                out_emb_channels=out_emb_channels,
                num_spherical=7,
                num_radial=6,
                cutoff=5.0,
            )
            head_in = out_channels + n_protein_features
            self.head = nn.Sequential(
                nn.Linear(head_in, hidden_channels),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_channels, 1),
            )

        def forward(self, data: Data) -> "object":
            import torch
            # DimeNetPlusPlus returns (batch_size, out_channels) — global pooling is internal
            emb = self.gnn(data.z, data.pos, data.batch)
            if self.n_protein_features > 0 and hasattr(data, 'protein_feats') and data.protein_feats is not None:
                emb = torch.cat([emb, data.protein_feats], dim=-1)
            return self.head(emb).squeeze(-1)

    return DimeNetAffinity(**kwargs)


def load_dimenet(model_path: Path, device: str):
    """Load the saved DimeNet++ model and return (model, torch_device, protein_feature_cols)."""
    try:
        import torch
    except ImportError as e:
        sys.exit(f"Error: could not import DimeNet dependencies ({e}). "
                 "Ensure torch and torch_geometric (with torch-cluster) are installed.")

    torch_device = torch.device(device)
    checkpoint = torch.load(model_path, map_location=torch_device)
    hp = checkpoint['hyperparams']
    n_protein_features = checkpoint.get('n_protein_features', 0)
    protein_feature_cols = checkpoint.get('protein_feature_cols', [])

    model = _build_dimenet_affinity(
        hidden_channels=hp.get('hidden_channels', 128),
        out_channels=hp.get('out_channels', 128),
        num_blocks=hp.get('num_blocks', 4),
        int_emb_size=hp.get('int_emb_size', 64),
        basis_emb_size=hp.get('basis_emb_size', 8),
        out_emb_channels=hp.get('out_emb_channels', 256),
        n_protein_features=n_protein_features,
        dropout=hp.get('dropout', 0.1),
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(torch_device)
    model.eval()
    return model, torch_device, protein_feature_cols


def predict_dimenet(
    manifest_df: pd.DataFrame,
    model,
    device,
    protein_feature_cols: List[str],
    manifest_dir: Path,
) -> pd.DataFrame:
    """
    Predict ΔG per (ligand, pose) using the DimeNet++ model with protein augmentation.

    For each pose:
      1. Builds a 3D ligand graph from mol2 (same structure for all poses of one ligand).
      2. Computes protein contact features from the protein PDB + this pose's PDBQT coordinates.
      3. Attaches protein_feats to the graph and runs a batched forward pass.

    Predictions differ per pose because protein contact features reflect each pose's
    3D position in the binding site.

    Returns a DataFrame with columns:
      ligand_id, pose_idx, vina_score, predicted_affinity_kcal_mol, model
    """
    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
    except ImportError as e:
        sys.exit(f"Error: could not import DimeNet dependencies ({e}).")

    def mol_to_graph_3d(mol, y: float = 0.0, protein_feats: Optional[List[float]] = None):
        """Convert an RDKit molecule with a 3D conformer to a PyG Data object for DimeNet++.

        Requires z (atomic numbers) and pos (3D coordinates). Returns None if the molecule
        is missing, has no atoms, or has no conformer.
        """
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        try:
            conf = mol.GetConformer()
        except ValueError:
            return None
        try:
            pos = torch.tensor(
                [[conf.GetAtomPosition(i).x,
                  conf.GetAtomPosition(i).y,
                  conf.GetAtomPosition(i).z]
                 for i in range(mol.GetNumAtoms())],
                dtype=torch.float,
            )
            z = torch.tensor(
                [atom.GetAtomicNum() for atom in mol.GetAtoms()],
                dtype=torch.long,
            )
            data = Data(z=z, pos=pos, y=torch.tensor([y], dtype=torch.float))
            if protein_feats is not None:
                data.protein_feats = torch.tensor([protein_feats], dtype=torch.float)
            return data
        except Exception:
            return None

    n_prot = model.n_protein_features

    # Cache parsed molecules per ligand
    mol_cache: Dict[str, object] = {}
    for row in manifest_df.itertuples(index=False):
        lid = row.ligand_id
        if lid not in mol_cache:
            mol2_path = _resolve_path(row.ligand_mol2, manifest_dir)
            mol = None
            if mol2_path.exists():
                mol = Chem.MolFromMol2File(str(mol2_path), removeHs=True)
            mol_cache[lid] = mol
            if mol is None:
                print(f"  Warning: could not parse mol2 for {lid} — skipping")

    graphs = []
    meta: List[dict] = []

    for row in manifest_df.itertuples(index=False):
        lid = row.ligand_id
        mol = mol_cache.get(lid)
        if mol is None:
            continue

        pdbqt_path = _resolve_path(row.vina_pdbqt, manifest_dir)
        poses = parse_pdbqt_scores(pdbqt_path)
        if not poses:
            print(f"  Warning: no poses in {pdbqt_path} for {lid}")
            continue

        protein_pdb = _resolve_path(row.protein_pdb, manifest_dir) if hasattr(row, 'protein_pdb') else None

        for pose_idx, vina_score in poses:
            prot_feats: Optional[List[float]] = None
            if n_prot > 0 and protein_pdb is not None and protein_pdb.exists():
                feats_arr = compute_contact_feats_for_pose(protein_pdb, pdbqt_path, pose_idx)
                if feats_arr is not None and len(feats_arr) >= n_prot:
                    prot_feats = feats_arr[:n_prot].tolist()
                else:
                    prot_feats = [0.0] * n_prot
            elif n_prot > 0:
                prot_feats = [0.0] * n_prot

            graph = mol_to_graph_3d(mol, y=0.0, protein_feats=prot_feats)
            if graph is None:
                continue
            graphs.append(graph)
            meta.append({'ligand_id': lid, 'pose_idx': pose_idx, 'vina_score': vina_score})

    if not graphs:
        return pd.DataFrame(columns=['ligand_id', 'pose_idx', 'vina_score',
                                     'predicted_affinity_kcal_mol', 'model'])

    loader = DataLoader(graphs, batch_size=32, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).squeeze(-1)
            preds.extend(out.cpu().numpy().tolist())

    rows = []
    for m, pred in zip(meta, preds):
        rows.append({**m, 'predicted_affinity_kcal_mol': float(pred), 'model': 'dimenet'})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _select_device(requested: str) -> str:
    if requested != 'auto':
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
    except ImportError:
        pass
    return 'cpu'


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Predict binding affinity for Vina-docked poses using a pre-trained, '
                    'final model checkpoint (inference only — no training).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--manifest', required=True,
                        help='CSV with columns: ligand_id, ligand_mol2, vina_pdbqt '
                             '(plus protein_pdb when --model dimenet).')
    parser.add_argument('--model', required=True,
                        choices=['svr_fp_pose', 'dimenet'],
                        help='Pose-aware model to run (different ΔG per pose): '
                             'svr_fp_pose (SVR+fingerprints) or dimenet (DimeNet++ + protein features).')
    parser.add_argument('--svr-model',
                        default='checkpoints/svr_affinity_fp_pose_model.joblib',
                        help='Path to the pose-aware SVR joblib checkpoint '
                             '(default: checkpoints/svr_affinity_fp_pose_model.joblib).')
    parser.add_argument('--svr-meta',
                        default='checkpoints/svr_affinity_fp_pose_metadata.json',
                        help='Path to the pose-aware SVR metadata JSON '
                             '(default: checkpoints/svr_affinity_fp_pose_metadata.json).')
    parser.add_argument('--dimenet-model',
                        default='checkpoints/dimenet_model.pt',
                        help='Path to the DimeNet++ .pt checkpoint '
                             '(default: checkpoints/dimenet_model.pt).')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device for DimeNet inference (default: auto). '
                             'Note: DimeNet++ does not support Apple MPS — use cpu or cuda.')
    parser.add_argument('--output', default='predictions.csv',
                        help='Output CSV path (default: predictions.csv)')
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        sys.exit(f"Error: manifest file not found: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    required_cols = {'ligand_id', 'ligand_mol2', 'vina_pdbqt'}
    if args.model == 'dimenet':
        required_cols.add('protein_pdb')  # needed for per-pose contact features
    missing = required_cols - set(manifest_df.columns)
    if missing:
        sys.exit(f"Error: manifest is missing columns: {', '.join(sorted(missing))}")

    manifest_dir = manifest_path.parent
    n_ligands = manifest_df['ligand_id'].nunique()
    print(f"Loaded manifest: {len(manifest_df)} rows, {n_ligands} unique ligands")

    if args.model == 'dimenet':
        model_path = Path(args.dimenet_model)
        if not model_path.exists():
            sys.exit(f"Error: DimeNet checkpoint not found: {model_path}\n"
                     "Provide it with --dimenet-model /path/to/dimenet_model.pt")
        device = _select_device(args.device)
        print(f"Loading DimeNet++ model from {model_path} (device: {device}) ...")
        model, torch_device, protein_feature_cols = load_dimenet(model_path, device)
        print(f"  Protein features ({model.n_protein_features}): {protein_feature_cols}")
        print("Computing per-pose predictions (ligand 3D graph + protein contact features) ...")
        results_df = predict_dimenet(
            manifest_df, model, torch_device, protein_feature_cols, manifest_dir)

    else:  # svr_fp_pose
        svr_model_path = Path(args.svr_model)
        svr_meta_path  = Path(args.svr_meta)
        for label, p in (('--svr-model', svr_model_path), ('--svr-meta', svr_meta_path)):
            if not p.exists():
                sys.exit(f"Error: SVR checkpoint not found: {p}\n"
                         f"Provide it with {label} /path/to/file. The pose-aware SVR needs "
                         "BOTH the .joblib model and its metadata .json.")
        print(f"Loading SVR model from {svr_model_path} ...")
        pipeline, feature_cols, n_bits = load_svr_fp(svr_model_path, svr_meta_path)
        meta_json = json.loads(svr_meta_path.read_text())
        pose_feature_cols = meta_json.get('pose_feature_cols', ['vina_score'])
        print(f"  Fingerprint bits: {n_bits}, pose features: {pose_feature_cols}")
        print("Computing per-pose predictions ...")
        results_df = predict_svr_fp_pose(
            manifest_df, pipeline, n_bits, pose_feature_cols, manifest_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(results_df)} rows to {output_path}")

    # Brief summary: best predicted ΔG per ligand
    if not results_df.empty and 'predicted_affinity_kcal_mol' in results_df.columns:
        summary = (results_df
                   .dropna(subset=['predicted_affinity_kcal_mol'])
                   .groupby('ligand_id')['predicted_affinity_kcal_mol'].min()
                   .reset_index()
                   .sort_values('predicted_affinity_kcal_mol'))
        if not summary.empty:
            print("\nBest predicted ΔG per ligand (kcal/mol) — sorted best to worst:")
            print(f"  {'Ligand':<20}  {'Best ΔG':>10}")
            print(f"  {'-'*20}  {'-'*10}")
            for r in summary.itertuples(index=False):
                print(f"  {r.ligand_id:<20}  {r.predicted_affinity_kcal_mol:>10.3f}")


if __name__ == '__main__':
    main()
