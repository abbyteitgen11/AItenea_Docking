"""
gnn_affinity.py — Graph Neural Network for binding affinity prediction.

Trains a GINEConv-based GNN on ligand molecular graphs to predict experimental
binding affinity (ΔG, kcal/mol). Uses the same train/val/test CSV splits as
process_pdbind.py and follows the same Optuna hyperparameter optimisation protocol.

Usage:
    # Basic run (default hyperparams):
    python gnn_affinity.py \
        --load-csv output/training_data.csv output/test_data.csv output/val_data.csv

    # With Optuna:
    python gnn_affinity.py \
        --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
        --optimize-hyperparams --n-trials 30 --optuna-db output/gnn_optuna.db

    # Reload saved hyperparams:
    python gnn_affinity.py \
        --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \
        --load-hyperparams output/gnn_best_hyperparams.json
"""

import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import optuna
from rdkit import Chem
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool, BatchNorm

# Import shared utilities from process_pdbind.py
sys.path.insert(0, str(Path(__file__).parent))
from process_pdbind import (  # noqa: E402
    load_data_from_csv,
    join_affinity_labels,
    prepare_affinity_data,
    split_val_from_train,
    STRUCTURES_DIR,
    OUTPUT_DIR,
    RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# Atom and bond feature constants
# ---------------------------------------------------------------------------

_ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']  # 9 + 1 "other"
_DEGREES = [0, 1, 2, 3, 4, 5, 6]
_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]  # 5 + 1 "other"
_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

# Node: 10 (atom type) + 8 (degree) + 1 (charge) + 1 (nH) + 1 (aromatic) + 1 (ring) + 6 (hybrid) = 28
NODE_DIM = (len(_ATOM_TYPES) + 1) + (len(_DEGREES) + 1) + 1 + 1 + 1 + 1 + (len(_HYBRIDIZATIONS) + 1)
# Edge: 5 (bond type + "other") + 1 (conjugated) + 1 (ring) = 7
EDGE_DIM = (len(_BOND_TYPES) + 1) + 2


def _one_hot(value, choices: list) -> List[float]:
    """One-hot encode value against choices; last bit is the 'other' category."""
    enc = [0.0] * (len(choices) + 1)
    if value in choices:
        enc[choices.index(value)] = 1.0
    else:
        enc[-1] = 1.0
    return enc


def atom_features(atom) -> np.ndarray:
    """Return a float32 feature vector for an RDKit atom."""
    feats = (
        _one_hot(atom.GetSymbol(), _ATOM_TYPES)
        + _one_hot(atom.GetDegree(), _DEGREES)
        + [float(np.clip(atom.GetFormalCharge(), -2, 2))]
        + [float(np.clip(atom.GetTotalNumHs(), 0, 4))]
        + [float(atom.GetIsAromatic())]
        + [float(atom.IsInRing())]
        + _one_hot(atom.GetHybridization(), _HYBRIDIZATIONS)
    )
    return np.array(feats, dtype=np.float32)


def bond_features(bond) -> np.ndarray:
    """Return a float32 feature vector for an RDKit bond."""
    feats = (
        _one_hot(bond.GetBondType(), _BOND_TYPES)
        + [float(bond.GetIsConjugated())]
        + [float(bond.IsInRing())]
    )
    return np.array(feats, dtype=np.float32)


def mol_to_graph(mol, y: float) -> Optional[Data]:
    """
    Convert an RDKit molecule and an affinity label to a PyG Data object.

    Returns None if the molecule has no atoms or feature extraction fails.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    try:
        # Node features
        x = torch.tensor(
            np.array([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32),
            dtype=torch.float,
        )

        # Edge index and edge features (bidirectional)
        edge_src, edge_dst, edge_feats = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            ef = bond_features(bond)
            edge_src += [i, j]
            edge_dst += [j, i]
            edge_feats += [ef, ef]

        if not edge_src:
            # Molecule with no bonds (e.g. single atom) — add self-loop
            edge_src = list(range(mol.GetNumAtoms()))
            edge_dst = list(range(mol.GetNumAtoms()))
            edge_feats = [np.zeros(EDGE_DIM, dtype=np.float32)] * mol.GetNumAtoms()

        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feats, dtype=np.float32), dtype=torch.float)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([y], dtype=torch.float),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LigandDataset(InMemoryDataset):
    """
    In-memory PyG dataset built from a per-complex affinity DataFrame.

    Each entry corresponds to one complex; the graph is the ligand molecular graph
    (from the mol2 crystal structure). Complexes whose mol2 files are missing or
    unparseable are silently dropped.

    Attributes:
        pdb_codes (List[str]): PDB codes of successfully parsed complexes.
        y_true    (np.ndarray): Experimental ΔG values for successfully parsed complexes.
    """

    def __init__(self, aff_df: pd.DataFrame):
        super().__init__(root=None, transform=None, pre_transform=None)
        data_list = []
        pdb_codes = []
        y_values = []

        for row in aff_df.itertuples(index=False):
            mol2_path = STRUCTURES_DIR / row.pdb_code / f"{row.pdb_code}_ligand.mol2"
            mol = None
            if mol2_path.exists():
                mol = Chem.MolFromMol2File(str(mol2_path), removeHs=True)
            graph = mol_to_graph(mol, float(row.exp_affinity_kcal_mol))
            if graph is not None:
                data_list.append(graph)
                pdb_codes.append(row.pdb_code)
                y_values.append(float(row.exp_affinity_kcal_mol))

        self.pdb_codes: List[str] = pdb_codes
        self.y_true: np.ndarray = np.array(y_values, dtype=np.float32)
        self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass


def build_datasets(
    train_aff: pd.DataFrame,
    val_aff: pd.DataFrame,
    test_aff: pd.DataFrame,
) -> Tuple['LigandDataset', 'LigandDataset', 'LigandDataset']:
    """Build train/val/test LigandDatasets from per-complex affinity DataFrames."""
    print("Building graph datasets...")
    train_ds = LigandDataset(train_aff)
    val_ds   = LigandDataset(val_aff)
    test_ds  = LigandDataset(test_aff)
    print(f"  Train: {len(train_ds)} / Val: {len(val_ds)} / Test: {len(test_ds)} complexes parsed")
    dropped_train = len(train_aff) - len(train_ds)
    dropped_val   = len(val_aff)   - len(val_ds)
    dropped_test  = len(test_aff)  - len(test_ds)
    if dropped_train + dropped_val + dropped_test > 0:
        print(f"  Dropped (missing/unparseable mol2): "
              f"{dropped_train} train, {dropped_val} val, {dropped_test} test")
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class AffinityGNN(nn.Module):
    """
    GINEConv-based GNN for binding affinity regression.

    Architecture:
        - Linear input projection: node_dim → hidden_dim
        - Linear edge projection:  edge_dim → hidden_dim  (required by GINEConv)
        - n_layers × GINEConv(MLP: hidden_dim → hidden_dim) + BatchNorm + ReLU + residual
        - Global mean + max pooling → concat (2 * hidden_dim)
        - MLP head: FC → ReLU → Dropout → FC → scalar output
    """

    def __init__(
        self,
        node_dim: int = NODE_DIM,
        edge_dim: int = EDGE_DIM,
        hidden_dim: int = 128,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.bns.append(BatchNorm(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x        = self.node_proj(data.x)
        edge_attr = self.edge_proj(data.edge_attr)

        for conv, bn in zip(self.convs, self.bns):
            x_new = conv(x, data.edge_index, edge_attr)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x = x + x_new  # residual connection

        x_mean = global_mean_pool(x, data.batch)
        x_max  = global_max_pool(x, data.batch)
        x_pool = torch.cat([x_mean, x_max], dim=-1)

        return self.head(x_pool).squeeze(-1)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_epoch(
    model: AffinityGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns mean MSE loss over the dataset."""
    model.train()
    total_loss = 0.0
    n_samples  = 0
    criterion  = nn.MSELoss(reduction='sum')

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_samples  += batch.num_graphs

    return total_loss / max(n_samples, 1)


def evaluate(
    model: AffinityGNN,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) arrays for all batches in loader."""
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred  = model(batch)
            y_true_list.append(batch.y.squeeze().cpu().numpy())
            y_pred_list.append(pred.cpu().numpy())
    y_true = np.concatenate(y_true_list).ravel()
    y_pred = np.concatenate(y_pred_list).ravel()
    return y_true, y_pred


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Return Pearson r, Spearman r, RMSE, MAE."""
    r_p, _ = pearsonr(y_pred, y_true)
    r_s, _ = spearmanr(y_pred, y_true)
    return {
        'pearson_r':  float(r_p),
        'spearman_r': float(r_s),
        'rmse':       float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae':        float(np.mean(np.abs(y_true - y_pred))),
    }


def train_gnn(
    train_ds: LigandDataset,
    val_ds: LigandDataset,
    hyperparams: Dict,
    epochs: int,
    patience: int,
    device: torch.device,
) -> Tuple[AffinityGNN, Dict]:
    """
    Train the GNN with early stopping on val MSE.

    Args:
        train_ds:    Training LigandDataset.
        val_ds:      Validation LigandDataset.
        hyperparams: Dict with keys hidden_dim, n_layers, dropout, lr, batch_size.
        epochs:      Maximum number of training epochs.
        patience:    Early stopping patience (epochs without val improvement).
        device:      Torch device.

    Returns:
        (best_model, val_metrics_dict) where val_metrics_dict has pearson_r, spearman_r,
        rmse, mae, y_true, y_pred keys.
    """
    hidden_dim = hyperparams.get('hidden_dim', 128)
    n_layers   = hyperparams.get('n_layers', 3)
    dropout    = hyperparams.get('dropout', 0.1)
    lr         = hyperparams.get('lr', 1e-3)
    batch_size = hyperparams.get('batch_size', 64)

    model = AffinityGNN(
        node_dim=NODE_DIM, edge_dim=EDGE_DIM,
        hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout,
    ).to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6,
    )

    best_val_mse   = float('inf')
    best_state     = None
    patience_count = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        y_true_val, y_pred_val = evaluate(model, val_loader, device)
        val_mse = float(mean_squared_error(y_true_val, y_pred_val))
        scheduler.step(val_mse)

        if val_mse < best_val_mse:
            best_val_mse   = val_mse
            best_state     = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1

        if epoch % 10 == 0 or epoch == 1:
            val_r, _ = pearsonr(y_pred_val, y_true_val)
            print(f"  Epoch {epoch:4d}  train_loss={train_loss:.4f}  "
                  f"val_mse={val_mse:.4f}  val_pearson_r={val_r:+.3f}  "
                  f"patience={patience_count}/{patience}")

        if patience_count >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    y_true_val, y_pred_val = evaluate(model, val_loader, device)
    val_metrics = compute_metrics(y_true_val, y_pred_val)
    val_metrics['y_true'] = y_true_val
    val_metrics['y_pred'] = y_pred_val

    return model, val_metrics


# ---------------------------------------------------------------------------
# Optuna hyperparameter optimisation
# ---------------------------------------------------------------------------

def optimize_gnn_hyperparams(
    train_aff_df: pd.DataFrame,
    val_aff_df: pd.DataFrame,
    n_trials: int = 30,
    storage: Optional[str] = None,
    study_name: str = 'gnn_affinity',
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Optuna search for GNN hyperparameters using 3-fold KFold CV on train+val combined.

    Uses max 50 epochs per fold with patience 10 to keep runtime manageable.
    Objective: minimise mean val MSE across 3 folds.
    """
    if device is None:
        device = torch.device('cpu')

    combined_df = pd.concat([train_aff_df, val_aff_df], ignore_index=True)
    # Build full dataset once (expensive mol2 parsing)
    print(f"  Building combined dataset for Optuna ({len(combined_df)} complexes)...")
    combined_ds = LigandDataset(combined_df)
    n = len(combined_ds)
    print(f"  {n} complexes parsed. Running {n_trials} Optuna trials × 3-fold CV...")

    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    indices = np.arange(n)

    def objective(trial):
        hp = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'n_layers':   trial.suggest_categorical('n_layers',   [2, 3, 4]),
            'dropout':    trial.suggest_categorical('dropout',    [0.0, 0.1, 0.2, 0.3]),
            'lr':         trial.suggest_categorical('lr',         [1e-4, 5e-4, 1e-3, 5e-3]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        }
        fold_mses = []
        for train_idx, val_idx in kf.split(indices):
            train_sub = torch.utils.data.Subset(combined_ds, train_idx.tolist())
            val_sub   = torch.utils.data.Subset(combined_ds, val_idx.tolist())
            t_loader  = DataLoader(train_sub, batch_size=hp['batch_size'], shuffle=True)
            v_loader  = DataLoader(val_sub,   batch_size=hp['batch_size'], shuffle=False)

            fold_model = AffinityGNN(
                node_dim=NODE_DIM, edge_dim=EDGE_DIM,
                hidden_dim=hp['hidden_dim'], n_layers=hp['n_layers'], dropout=hp['dropout'],
            ).to(device)
            opt = torch.optim.Adam(fold_model.parameters(), lr=hp['lr'], weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', patience=5, factor=0.5,
            )

            best_fold_mse = float('inf')
            p_count = 0
            for _ in range(50):
                train_epoch(fold_model, t_loader, opt, device)
                y_t, y_p = evaluate(fold_model, v_loader, device)
                mse = float(mean_squared_error(y_t, y_p))
                sched.step(mse)
                if mse < best_fold_mse:
                    best_fold_mse = mse
                    p_count = 0
                else:
                    p_count += 1
                if p_count >= 10:
                    break
            fold_mses.append(best_fold_mse)

        return float(np.mean(fold_mses))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='minimize',
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    n_existing = len(study.trials)
    if n_existing:
        print(f"  Loaded {n_existing} existing GNN Optuna trials from storage")
    study.optimize(objective, n_trials=n_trials)
    print(f"  GNN best CV MSE: {study.best_value:.6f} (over {len(study.trials)} total trials)")
    print(f"  Best params: {study.best_params}")
    return study.best_params


# ---------------------------------------------------------------------------
# Output functions
# ---------------------------------------------------------------------------

def export_gnn_predictions_csv(
    pdb_codes: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> None:
    """Save per-complex GNN predictions to CSV."""
    out = pd.DataFrame({
        'pdb_code':                    pdb_codes,
        'exp_affinity_kcal_mol':       y_true,
        'predicted_affinity_kcal_mol': y_pred,
    })
    path = output_dir / 'gnn_affinity_predictions.csv'
    out.to_csv(path, index=False)
    print(f"Saved GNN predictions to {path}  ({len(out)} complexes)")


def plot_gnn_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict,
    output_dir: Path,
    split: str = 'test',
) -> None:
    """Scatter plot of predicted vs experimental ΔG."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20, color='steelblue')

    lims = [min(y_true.min(), y_pred.min()) - 0.5,
            max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.6)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Experimental ΔG (kcal/mol)')
    ax.set_ylabel('Predicted ΔG (kcal/mol)')
    ax.set_title(f'GNN Affinity Predictions ({split.capitalize()} set)')
    ax.text(
        0.05, 0.93,
        f"Pearson r = {metrics['pearson_r']:+.3f}\n"
        f"Spearman r = {metrics['spearman_r']:+.3f}\n"
        f"RMSE = {metrics['rmse']:.3f} kcal/mol",
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )

    path = output_dir / f'gnn_affinity_predictions_{split}.png'
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved GNN plot to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='GNN binding affinity prediction using PyTorch Geometric.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--load-csv',
        nargs='+',
        required=True,
        metavar='CSV',
        help='TRAIN_CSV TEST_CSV [VAL_CSV] — CSV files produced by process_pdbind.py',
    )
    parser.add_argument(
        '--val-frac',
        type=float,
        default=0.2,
        help='Fraction of training complexes for validation when VAL_CSV is not provided',
    )
    parser.add_argument(
        '--optimize-hyperparams',
        action='store_true',
        help='Run Optuna hyperparameter search (3-fold CV, 50 epochs/fold)',
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=30,
        help='Number of Optuna trials',
    )
    parser.add_argument(
        '--optuna-db',
        type=str,
        default=None,
        help='SQLite path for persisting/resuming Optuna trials '
             '(e.g. output/gnn_optuna.db)',
    )
    parser.add_argument(
        '--load-hyperparams',
        type=str,
        default=None,
        metavar='JSON_PATH',
        help='Load previously optimised hyperparameters from JSON '
             '(e.g. output/gnn_best_hyperparams.json)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Maximum epochs for final model training',
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience (epochs) for final model training',
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip scatter plot generation',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Torch device: auto detects CUDA → MPS → CPU',
    )
    args = parser.parse_args()

    # --- Device ---
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    optuna_storage = f"sqlite:///{args.optuna_db}" if args.optuna_db else None

    # --- Load CSVs ---
    print("\nLoading data from CSV files...")
    csvs    = args.load_csv
    val_csv = Path(csvs[2]) if len(csvs) >= 3 else None
    train_df, val_df, test_df = load_data_from_csv(
        Path(csvs[0]), Path(csvs[1]), val_csv,
    )
    if val_df is None:
        train_df, val_df = split_val_from_train(train_df, args.val_frac)
        print(f"  Val set carved from training: "
              f"{val_df['pdb_code'].nunique()} complexes")

    # --- Join affinity labels ---
    print("\nJoining affinity labels...")
    train_df_aff = join_affinity_labels(train_df)
    val_df_aff   = join_affinity_labels(val_df)
    test_df_aff  = join_affinity_labels(test_df)

    _aff_exclude = {'pdb_code', 'pose_idx', 'is_best_pose', 'rmsd',
                    'exp_affinity_pKd', 'exp_affinity_kcal_mol'}
    feature_cols_aff = [c for c in train_df.columns if c not in _aff_exclude]

    print("\nPreparing per-complex affinity datasets (rank-1 pose)...")
    train_aff = prepare_affinity_data(train_df_aff, feature_cols_aff)
    val_aff   = prepare_affinity_data(val_df_aff,   feature_cols_aff)
    test_aff  = prepare_affinity_data(test_df_aff,  feature_cols_aff)

    if len(train_aff) < 10 or len(test_aff) < 2:
        print("Error: too few complexes with affinity labels. Exiting.")
        sys.exit(1)

    # --- Build graph datasets ---
    print()
    train_ds, val_ds, test_ds = build_datasets(train_aff, val_aff, test_aff)

    if len(train_ds) < 10:
        print("Error: too few training graphs parsed. Check mol2 file paths.")
        sys.exit(1)

    # --- Hyperparameters ---
    hyperparams: Dict = {}
    if args.load_hyperparams:
        hp_path = Path(args.load_hyperparams)
        if not hp_path.exists():
            print(f"Warning: --load-hyperparams file not found: {hp_path}. Using defaults.")
        else:
            hyperparams = json.loads(hp_path.read_text())
            print(f"Loaded hyperparameters from {hp_path}: {hyperparams}")

    if args.optimize_hyperparams:
        print(f"\nOptimising GNN hyperparameters "
              f"(Optuna, {args.n_trials} trials × 3-fold CV, max 50 epochs/fold)...")
        hyperparams = optimize_gnn_hyperparams(
            train_aff, val_aff,
            n_trials=args.n_trials,
            storage=optuna_storage,
            study_name='gnn_affinity',
            device=device,
        )
        hp_save = OUTPUT_DIR / 'gnn_best_hyperparams.json'
        hp_save.write_text(json.dumps(hyperparams, indent=2))
        print(f"Saved GNN hyperparameters to {hp_save}")

    # --- Final training ---
    print(f"\nTraining final GNN model "
          f"(max {args.epochs} epochs, patience {args.patience})...")
    print(f"  Hyperparams: {hyperparams or 'defaults'}")
    model, val_metrics = train_gnn(
        train_ds, val_ds, hyperparams,
        epochs=args.epochs, patience=args.patience, device=device,
    )

    # --- Evaluate on test set ---
    test_loader = DataLoader(test_ds, batch_size=hyperparams.get('batch_size', 64), shuffle=False)
    y_true_test, y_pred_test = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(y_true_test, y_pred_test)

    # --- Print results ---
    n_val  = len(val_ds)
    n_test = len(test_ds)
    print(f"\n{'='*60}")
    print("GNN BINDING AFFINITY PERFORMANCE")
    print(f"{'='*60}")
    print(f"Val complexes: {n_val}   Test complexes: {n_test}")
    print(f"\nValidation set:")
    print(f"  Pearson r  : {val_metrics['pearson_r']:+.3f}")
    print(f"  Spearman r : {val_metrics['spearman_r']:+.3f}")
    print(f"  RMSE       : {val_metrics['rmse']:.3f} kcal/mol")
    print(f"  MAE        : {val_metrics['mae']:.3f} kcal/mol")
    print(f"\nTest set:")
    print(f"  Pearson r  : {test_metrics['pearson_r']:+.3f}")
    print(f"  Spearman r : {test_metrics['spearman_r']:+.3f}")
    print(f"  RMSE       : {test_metrics['rmse']:.3f} kcal/mol")
    print(f"  MAE        : {test_metrics['mae']:.3f} kcal/mol")

    # --- Save model ---
    model_path = OUTPUT_DIR / 'gnn_model.pt'
    torch.save({
        'state_dict':  model.state_dict(),
        'hyperparams': hyperparams,
        'node_dim':    NODE_DIM,
        'edge_dim':    EDGE_DIM,
    }, model_path)
    print(f"\nSaved model to {model_path}")

    # --- Export predictions CSV ---
    export_gnn_predictions_csv(test_ds.pdb_codes, y_true_test, y_pred_test, OUTPUT_DIR)

    # --- Plots ---
    if not args.no_plots:
        try:
            plot_gnn_predictions(
                val_metrics['y_true'], val_metrics['y_pred'],
                val_metrics, OUTPUT_DIR, split='val',
            )
            plot_gnn_predictions(
                y_true_test, y_pred_test,
                test_metrics, OUTPUT_DIR, split='test',
            )
        except Exception as e:
            print(f"Warning: could not generate GNN plots: {e}")

    print("\nDone!")


if __name__ == '__main__':
    main()
