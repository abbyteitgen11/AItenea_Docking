"""
predict_affinity.py — Standalone binding affinity prediction for new ligands.

Loads a pre-trained SVR+fingerprints or GNN model and predicts binding affinity
(ΔG, kcal/mol) for Vina-docked poses of novel ligands.

Input: a manifest CSV listing ligand mol2 files, protein PDB files, and Vina
       PDBQT outputs.
Output: a CSV with one row per (ligand, pose) containing the Vina score and the
        ML-predicted ΔG.

Both models are per-ligand: they use only molecular structure (not pose
coordinates), so all poses of the same ligand receive the same predicted ΔG.
The Vina score varies per pose and is included for reference.

Usage:
    # SVR + fingerprints
    python predict_affinity.py \\
        --manifest inputs/manifest.csv \\
        --model svr_fp \\
        --svr-model output/svr_affinity_fp_model.joblib \\
        --svr-meta  output/svr_affinity_fp_metadata.json \\
        --output predictions.csv

    # GNN
    python predict_affinity.py \\
        --manifest inputs/manifest.csv \\
        --model gnn \\
        --gnn-model output/gnn_model.pt \\
        --output predictions.csv
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
from process_pdbind import compute_morgan_fingerprint  # noqa: E402


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
    """Load the saved GNN model and return (model, device)."""
    try:
        import torch
        from gnn_affinity import AffinityGNN
    except ImportError as e:
        sys.exit(f"Error: could not import GNN dependencies ({e}). "
                 "Ensure torch and torch_geometric are installed.")

    torch_device = torch.device(device)
    checkpoint = torch.load(model_path, map_location=torch_device)
    hp = checkpoint['hyperparams']
    node_dim = checkpoint.get('node_dim', 28)
    edge_dim = checkpoint.get('edge_dim', 7)

    model = AffinityGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hp.get('hidden_dim', 128),
        n_layers=hp.get('n_layers', 3),
        dropout=hp.get('dropout', 0.1),
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(torch_device)
    model.eval()
    return model, torch_device


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
    parser.add_argument('--model', required=True, choices=['svr_fp', 'gnn'],
                        help='Model to use: svr_fp (SVR+fingerprints) or gnn (Graph Neural Network)')
    parser.add_argument('--svr-model', default='output/svr_affinity_fp_model.joblib',
                        help='Path to the saved SVR joblib file (required for --model svr_fp)')
    parser.add_argument('--svr-meta', default='output/svr_affinity_fp_metadata.json',
                        help='Path to the SVR metadata JSON file (required for --model svr_fp)')
    parser.add_argument('--gnn-model', default='output/gnn_model.pt',
                        help='Path to the saved GNN .pt checkpoint (required for --model gnn)')
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

    if args.model == 'svr_fp':
        svr_model_path = Path(args.svr_model)
        svr_meta_path = Path(args.svr_meta)
        for p in (svr_model_path, svr_meta_path):
            if not p.exists():
                sys.exit(f"Error: model file not found: {p}\n"
                         "Run process_pdbind.py with --affinity-compare-features to generate it.")
        print(f"Loading SVR+fingerprints model from {svr_model_path} ...")
        pipeline, feature_cols, n_bits = load_svr_fp(svr_model_path, svr_meta_path)
        print(f"  Fingerprint bits: {n_bits}, feature columns: {len(feature_cols)}")
        print("Computing fingerprints and predicting ...")
        predictions = predict_svr_fp(manifest_df, pipeline, feature_cols, n_bits, manifest_dir)
        model_label = 'svr_fp'

    else:  # gnn
        gnn_model_path = Path(args.gnn_model)
        if not gnn_model_path.exists():
            sys.exit(f"Error: GNN model file not found: {gnn_model_path}\n"
                     "Run gnn_affinity.py to generate it.")
        device = _select_device(args.device)
        print(f"Loading GNN model from {gnn_model_path} (device: {device}) ...")
        model, torch_device = load_gnn(gnn_model_path, device)
        print("Building molecular graphs and predicting ...")
        predictions = predict_gnn(manifest_df, model, torch_device, manifest_dir)
        model_label = 'gnn'

    n_predicted = sum(1 for v in predictions.values() if v is not None)
    n_failed = len(predictions) - n_predicted
    print(f"Predictions: {n_predicted}/{n_ligands} ligands succeeded "
          f"({n_failed} failed mol2 parsing)")

    results_df = build_results_df(manifest_df, predictions, model_label, manifest_dir)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(results_df)} rows to {output_path}")

    # Print a brief summary table
    summary = (results_df
               .dropna(subset=['predicted_affinity_kcal_mol'])
               .drop_duplicates(subset='ligand_id')
               .sort_values('predicted_affinity_kcal_mol'))
    if not summary.empty:
        print("\nPredicted affinities (ΔG, kcal/mol) — sorted best to worst:")
        print(f"  {'Ligand':<20}  {'Predicted ΔG':>14}")
        print(f"  {'-'*20}  {'-'*14}")
        for r in summary.itertuples(index=False):
            print(f"  {r.ligand_id:<20}  {r.predicted_affinity_kcal_mol:>14.3f}")


if __name__ == '__main__':
    main()
