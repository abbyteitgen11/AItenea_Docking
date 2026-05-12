"""
predict_affinity.py — Standalone binding affinity prediction for new ligands.

Loads a pre-trained SVR+fingerprints or GNN model and predicts binding affinity
(ΔG, kcal/mol) for Vina-docked poses of novel ligands.

Input: a manifest CSV listing ligand mol2 files, protein PDB files, and Vina
       PDBQT outputs.
Output: a CSV with one row per (ligand, pose) containing the Vina score and the
        ML-predicted ΔG.

Model options
─────────────
  svr_fp       Per-ligand SVR+fingerprints. All poses of the same ligand get the
               same predicted ΔG (model uses only molecular structure).
  gnn          Per-ligand GNN. Same caveat as svr_fp.
  svr_fp_pose  Pose-aware SVR+fingerprints. Trained on all Vina poses with Vina
               score as an additional feature; produces a different ΔG per pose.
  gnn_pose     Pose-aware GNN. Graph embedding + Vina score concatenated before the
               MLP head; produces a different ΔG per pose.
  dimenet      DimeNet++ GNN with protein feature augmentation. Uses 3D ligand
               coordinates and protein–ligand contact scalars; produces different
               ΔG per pose because contact features differ between docked poses.

Usage:
    # Per-ligand models (same ΔG for all poses of one ligand)
    python predict_affinity.py --manifest inputs/manifest.csv --model svr_fp \\
        --svr-model output/svr_affinity_fp_model.joblib \\
        --svr-meta  output/svr_affinity_fp_metadata.json --output predictions.csv

    python predict_affinity.py --manifest inputs/manifest.csv --model gnn \\
        --gnn-model output/gnn_model.pt --output predictions.csv

    # Pose-aware models (different ΔG per pose)
    python predict_affinity.py --manifest inputs/manifest.csv --model svr_fp_pose \\
        --svr-model output/svr_affinity_fp_pose_model.joblib \\
        --svr-meta  output/svr_affinity_fp_pose_metadata.json --output predictions.csv

    python predict_affinity.py --manifest inputs/manifest.csv --model gnn_pose \\
        --gnn-model output/gnn_pose_model.pt --output predictions.csv

    # DimeNet++ with protein features (pose-aware by nature)
    python predict_affinity.py --manifest inputs/manifest.csv --model dimenet \\
        --gnn-model output/dimenet_model.pt --output predictions_dimenet.csv
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

sys.path.insert(0, str(Path(__file__).parent))
from process_pdbind import (  # noqa: E402
    compute_morgan_fingerprint,
    load_protein_atoms,
    extract_pose_atoms,
    compute_pose_contact_features,
)
from scipy.spatial import KDTree


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


# ---------------------------------------------------------------------------
# SVR + fingerprints
# ---------------------------------------------------------------------------

def load_svr_fp(model_path: Path, meta_path: Path):
    """Load the joblib SVR pipeline and its feature column list."""
    pipeline = joblib.load(model_path)
    meta = json.loads(meta_path.read_text())
    feature_cols: List[str] = meta['feature_cols']
    n_bits: int = meta.get('n_bits', 2048)
    return pipeline, feature_cols, n_bits


def predict_svr_fp(
    manifest_df: pd.DataFrame,
    pipeline,
    feature_cols: List[str],
    n_bits: int,
    manifest_dir: Path,
) -> Dict[str, Optional[float]]:
    """
    Predict ΔG for each unique ligand using the SVR+fingerprints pipeline.

    Returns a dict mapping ligand_id → predicted ΔG (or None if mol2 fails).
    """
    predictions: Dict[str, Optional[float]] = {}
    seen_ids = set()
    for row in manifest_df.itertuples(index=False):
        lid = row.ligand_id
        if lid in seen_ids:
            continue
        seen_ids.add(lid)

        mol2_path = _resolve_path(row.ligand_mol2, manifest_dir)
        fp = compute_morgan_fingerprint(mol2_path, n_bits=n_bits)
        if fp is None:
            print(f"  Warning: could not parse mol2 for {lid} — skipping")
            predictions[lid] = None
            continue

        # Build feature vector matching the column order the model was trained on
        fp_dict = {f'morgan_fp_{i}': fp[i] for i in range(len(fp))}
        X = np.array([[fp_dict.get(c, 0.0) for c in feature_cols]], dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0)
        predictions[lid] = float(pipeline.predict(X)[0])

    return predictions


# ---------------------------------------------------------------------------
# GNN
# ---------------------------------------------------------------------------

def load_gnn(model_path: Path, device: str):
    """Load the saved GNN model and return (model, torch_device, pose_feature_cols)."""
    try:
        import torch
        from gnn_affinity import AffinityGNN
    except ImportError as e:
        sys.exit(f"Error: could not import GNN dependencies ({e}). "
                 "Ensure torch and torch_geometric are installed.")

    torch_device = torch.device(device)
    checkpoint = torch.load(model_path, map_location=torch_device)
    hp = checkpoint['hyperparams']
    node_dim           = checkpoint.get('node_dim', 28)
    edge_dim           = checkpoint.get('edge_dim', 7)
    n_pose_features    = checkpoint.get('n_pose_features', 0)
    pose_feature_cols  = checkpoint.get('pose_feature_cols', [])

    model = AffinityGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hp.get('hidden_dim', 128),
        n_layers=hp.get('n_layers', 3),
        dropout=hp.get('dropout', 0.1),
        n_pose_features=n_pose_features,
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(torch_device)
    model.eval()
    return model, torch_device, pose_feature_cols


def predict_gnn(
    manifest_df: pd.DataFrame,
    model,
    device,
    manifest_dir: Path,
) -> Dict[str, Optional[float]]:
    """
    Predict ΔG for each unique ligand using the GNN model.

    Returns a dict mapping ligand_id → predicted ΔG (or None if mol2 fails).
    """
    try:
        import torch
        from gnn_affinity import mol_to_graph
        from torch_geometric.loader import DataLoader
    except ImportError as e:
        sys.exit(f"Error: could not import GNN dependencies ({e}).")

    # Build one graph per unique ligand
    graphs = []
    ligand_ids = []
    seen_ids: set = set()

    for row in manifest_df.itertuples(index=False):
        lid = row.ligand_id
        if lid in seen_ids:
            continue
        seen_ids.add(lid)

        mol2_path = _resolve_path(row.ligand_mol2, manifest_dir)
        mol = None
        if mol2_path.exists():
            mol = Chem.MolFromMol2File(str(mol2_path), removeHs=True)
        graph = mol_to_graph(mol, y=0.0) if mol is not None else None

        if graph is None:
            print(f"  Warning: could not parse mol2 for {lid} — skipping")
            ligand_ids.append(lid)
            graphs.append(None)
        else:
            ligand_ids.append(lid)
            graphs.append(graph)

    valid_graphs = [g for g in graphs if g is not None]
    valid_ids = [lid for lid, g in zip(ligand_ids, graphs) if g is not None]
    failed_ids = {lid for lid, g in zip(ligand_ids, graphs) if g is None}

    predictions: Dict[str, Optional[float]] = {lid: None for lid in failed_ids}

    if not valid_graphs:
        return predictions

    loader = DataLoader(valid_graphs, batch_size=64, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).squeeze(-1)
            preds.extend(out.cpu().numpy().tolist())

    for lid, pred in zip(valid_ids, preds):
        predictions[lid] = float(pred)

    return predictions


# ---------------------------------------------------------------------------
# Pose-aware prediction (different ΔG per pose)
# ---------------------------------------------------------------------------

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


def predict_gnn_pose(
    manifest_df: pd.DataFrame,
    model,
    device,
    pose_feature_cols: List[str],
    manifest_dir: Path,
) -> pd.DataFrame:
    """
    Predict ΔG per (ligand, pose) using the pose-aware GNN.

    For each pose: builds a copy of the ligand graph with pose_feats attached,
    then runs a batched forward pass.

    Returns a DataFrame with columns: ligand_id, pose_idx, vina_score,
    predicted_affinity_kcal_mol, model.
    """
    try:
        import torch
        from gnn_affinity import mol_to_graph
        from torch_geometric.loader import DataLoader
    except ImportError as e:
        sys.exit(f"Error: could not import GNN dependencies ({e}).")

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
        for pose_idx, vina_score in poses:
            graph = mol_to_graph(mol, y=0.0)
            if graph is None:
                continue
            pose_vals = _pose_scalar_vec(pose_feature_cols, vina_score)
            graph.pose_feats = torch.tensor([pose_vals.tolist()], dtype=torch.float)
            graphs.append(graph)
            meta.append({'ligand_id': lid, 'pose_idx': pose_idx, 'vina_score': vina_score})

    if not graphs:
        return pd.DataFrame(columns=['ligand_id', 'pose_idx', 'vina_score',
                                     'predicted_affinity_kcal_mol', 'model'])

    loader = DataLoader(graphs, batch_size=64, shuffle=False)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).squeeze(-1)
            preds.extend(out.cpu().numpy().tolist())

    rows = []
    for m, pred in zip(meta, preds):
        rows.append({**m, 'predicted_affinity_kcal_mol': float(pred), 'model': 'gnn_pose'})
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

    prot_elem_arr = np.array(prot_elements)
    prot_res_arr = np.array(prot_resnames)
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


def load_dimenet(model_path: Path, device: str):
    """Load the saved DimeNetAffinity model and return (model, torch_device, protein_feature_cols)."""
    try:
        import torch
        from gnn_dimenet_affinity import DimeNetAffinity
    except ImportError as e:
        sys.exit(f"Error: could not import DimeNet dependencies ({e}). "
                 "Ensure torch and torch_geometric are installed.")

    torch_device = torch.device(device)
    checkpoint = torch.load(model_path, map_location=torch_device)
    hp = checkpoint['hyperparams']
    n_protein_features = checkpoint.get('n_protein_features', 0)
    protein_feature_cols = checkpoint.get('protein_feature_cols', [])

    model = DimeNetAffinity(
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
        from gnn_dimenet_affinity import mol_to_graph_3d
        from torch_geometric.loader import DataLoader
    except ImportError as e:
        sys.exit(f"Error: could not import DimeNet dependencies ({e}).")

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
# Result assembly
# ---------------------------------------------------------------------------

def _resolve_path(p: str, base_dir: Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return base_dir / path


def build_results_df(
    manifest_df: pd.DataFrame,
    predictions: Dict[str, Optional[float]],
    model_name: str,
    manifest_dir: Path,
) -> pd.DataFrame:
    """
    Combine Vina pose scores with per-ligand ML predictions into a results table.

    Each row corresponds to one (ligand_id, pose_idx) pair.
    """
    rows = []
    for row in manifest_df.itertuples(index=False):
        lid = row.ligand_id
        pred = predictions.get(lid)
        pdbqt_path = _resolve_path(row.vina_pdbqt, manifest_dir)
        poses = parse_pdbqt_scores(pdbqt_path)
        if not poses:
            print(f"  Warning: no poses found in {pdbqt_path} for {lid}")
            rows.append({
                'ligand_id': lid,
                'pose_idx': None,
                'vina_score': None,
                'predicted_affinity_kcal_mol': pred,
                'model': model_name,
            })
            continue
        for pose_idx, vina_score in poses:
            rows.append({
                'ligand_id': lid,
                'pose_idx': pose_idx,
                'vina_score': vina_score,
                'predicted_affinity_kcal_mol': pred,
                'model': model_name,
            })
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
        description='Predict binding affinity for Vina-docked poses using a pre-trained ML model.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--manifest', required=True,
                        help='CSV with columns: ligand_id, ligand_mol2, protein_pdb, vina_pdbqt')
    parser.add_argument('--model', required=True,
                        choices=['svr_fp', 'gnn', 'svr_fp_pose', 'gnn_pose', 'dimenet'],
                        help='Model to use. Per-ligand (same ΔG for all poses): svr_fp, gnn. '
                             'Pose-aware (different ΔG per pose): svr_fp_pose, gnn_pose, dimenet.')
    parser.add_argument('--svr-model',
                        default=None,
                        help='Path to the SVR joblib file. Defaults to '
                             'output/svr_affinity_fp_model.joblib (svr_fp) or '
                             'output/svr_affinity_fp_pose_model.joblib (svr_fp_pose).')
    parser.add_argument('--svr-meta',
                        default=None,
                        help='Path to the SVR metadata JSON. Defaults to '
                             'output/svr_affinity_fp_metadata.json (svr_fp) or '
                             'output/svr_affinity_fp_pose_metadata.json (svr_fp_pose).')
    parser.add_argument('--gnn-model',
                        default=None,
                        help='Path to the GNN .pt checkpoint. Defaults to '
                             'output/gnn_model.pt (gnn), output/gnn_pose_model.pt (gnn_pose), '
                             'or output/dimenet_model.pt (dimenet).')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device for GNN inference (default: auto)')
    parser.add_argument('--output', default='predictions.csv',
                        help='Output CSV path (default: predictions.csv)')
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        sys.exit(f"Error: manifest file not found: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path)
    required_cols = {'ligand_id', 'ligand_mol2', 'protein_pdb', 'vina_pdbqt'}
    missing = required_cols - set(manifest_df.columns)
    if missing:
        sys.exit(f"Error: manifest is missing columns: {', '.join(sorted(missing))}")

    manifest_dir = manifest_path.parent
    n_ligands = manifest_df['ligand_id'].nunique()
    print(f"Loaded manifest: {len(manifest_df)} rows, {n_ligands} unique ligands")

    pose_aware = args.model in ('svr_fp_pose', 'gnn_pose')
    results_df: pd.DataFrame

    if args.model == 'dimenet':
        default_gnn = 'output/dimenet_model.pt'
        gnn_model_path = Path(args.gnn_model or default_gnn)
        if not gnn_model_path.exists():
            sys.exit(f"Error: DimeNet model file not found: {gnn_model_path}\n"
                     "Run gnn_dimenet_affinity.py to generate it.")
        device = _select_device(args.device)
        print(f"Loading DimeNet++ model from {gnn_model_path} (device: {device}) ...")
        model, torch_device, protein_feature_cols = load_dimenet(gnn_model_path, device)
        print(f"  Protein features ({model.n_protein_features}): {protein_feature_cols}")
        print("Computing per-pose predictions (ligand 3D graph + protein contact features) ...")
        results_df = predict_dimenet(
            manifest_df, model, torch_device, protein_feature_cols, manifest_dir)

    elif args.model in ('svr_fp', 'svr_fp_pose'):
        default_model = ('output/svr_affinity_fp_pose_model.joblib' if pose_aware
                         else 'output/svr_affinity_fp_model.joblib')
        default_meta  = ('output/svr_affinity_fp_pose_metadata.json' if pose_aware
                         else 'output/svr_affinity_fp_metadata.json')
        svr_model_path = Path(args.svr_model or default_model)
        svr_meta_path  = Path(args.svr_meta  or default_meta)
        train_flag = ('--affinity-compare-features --pose-aware-affinity' if pose_aware
                      else '--affinity-compare-features')
        for p in (svr_model_path, svr_meta_path):
            if not p.exists():
                sys.exit(f"Error: model file not found: {p}\n"
                         f"Run process_pdbind.py with {train_flag} to generate it.")
        print(f"Loading SVR model from {svr_model_path} ...")
        pipeline, feature_cols, n_bits = load_svr_fp(svr_model_path, svr_meta_path)
        meta_json = json.loads(svr_meta_path.read_text())

        if pose_aware:
            pose_feature_cols = meta_json.get('pose_feature_cols', ['vina_score'])
            print(f"  Fingerprint bits: {n_bits}, pose features: {pose_feature_cols}")
            print("Computing per-pose predictions ...")
            results_df = predict_svr_fp_pose(
                manifest_df, pipeline, n_bits, pose_feature_cols, manifest_dir)
        else:
            print(f"  Fingerprint bits: {n_bits}, feature columns: {len(feature_cols)}")
            print("Computing fingerprints and predicting ...")
            predictions = predict_svr_fp(
                manifest_df, pipeline, feature_cols, n_bits, manifest_dir)
            n_predicted = sum(1 for v in predictions.values() if v is not None)
            n_failed = n_ligands - n_predicted
            print(f"Predictions: {n_predicted}/{n_ligands} ligands succeeded "
                  f"({n_failed} failed mol2 parsing)")
            results_df = build_results_df(manifest_df, predictions, 'svr_fp', manifest_dir)

    else:  # gnn or gnn_pose
        default_gnn = ('output/gnn_pose_model.pt' if pose_aware else 'output/gnn_model.pt')
        gnn_model_path = Path(args.gnn_model or default_gnn)
        train_flag = '--pose-aware' if pose_aware else ''
        if not gnn_model_path.exists():
            sys.exit(f"Error: GNN model file not found: {gnn_model_path}\n"
                     f"Run gnn_affinity.py {train_flag} to generate it.")
        device = _select_device(args.device)
        print(f"Loading GNN model from {gnn_model_path} (device: {device}) ...")
        model, torch_device, pose_feature_cols = load_gnn(gnn_model_path, device)

        if pose_aware:
            print(f"  Pose features: {pose_feature_cols}")
            print("Building per-pose graphs and predicting ...")
            results_df = predict_gnn_pose(
                manifest_df, model, torch_device, pose_feature_cols, manifest_dir)
        else:
            print("Building molecular graphs and predicting ...")
            predictions = predict_gnn(manifest_df, model, torch_device, manifest_dir)
            n_predicted = sum(1 for v in predictions.values() if v is not None)
            n_failed = n_ligands - n_predicted
            print(f"Predictions: {n_predicted}/{n_ligands} ligands succeeded "
                  f"({n_failed} failed mol2 parsing)")
            results_df = build_results_df(manifest_df, predictions, 'gnn', manifest_dir)

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
