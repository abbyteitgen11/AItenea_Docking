"""
gnn_dimenet_affinity.py — DimeNet++ GNN with protein feature augmentation for binding
affinity prediction.

Uses 3D atom coordinates from mol2 crystal structures as input to DimeNet++, then
concatenates protein–ligand contact scalar features with the graph embedding before the
MLP head. This gives the model both 3D ligand geometry (via DimeNet++) and protein
context (via contact scalars) without requiring a full protein graph.

Architecture:
    ligand 3D graph (z, pos) → DimeNetPlusPlus → graph embedding (out_channels,)
    → cat([graph_embedding, protein_contact_features])
    → MLP head → scalar ΔG (kcal/mol)

Usage:
    # Basic run with protein contact features (requires --augment data):
    python gnn_dimenet_affinity.py \\
        --load-csv output/training_data.csv output/test_data.csv output/val_data.csv

    # Without protein features (ligand-only):
    python gnn_dimenet_affinity.py \\
        --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \\
        --no-protein-features

    # With Optuna hyperparameter search:
    python gnn_dimenet_affinity.py \\
        --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \\
        --optimize-hyperparams --n-trials 30

    # Load previously saved hyperparameters:
    python gnn_dimenet_affinity.py \\
        --load-csv output/training_data.csv output/test_data.csv output/val_data.csv \\
        --load-hyperparams output/dimenet_best_hyperparams.json
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
from torch_geometric.nn import DimeNetPlusPlus

sys.path.insert(0, str(Path(__file__).parent))
from process_pdbind import (  # noqa: E402
    load_data_from_csv,
    join_affinity_labels,
    prepare_affinity_data,
    split_val_from_train,
    get_complex_path,
    STRUCTURES_DIR,
    OUTPUT_DIR,
    RANDOM_SEED,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PROTEIN_FEATURE_COLS = [
    'contact_n_3A', 'contact_n_4A', 'contact_n_5A',
    'contact_min_dist', 'contact_score_lj', 'contact_buried_frac',
    'contact_n_per_atom', 'contact_n_hydrophobic', 'contact_n_hbond',
    'contact_n_aromatic', 'contact_score_gaussian', 'contact_hbond_normalized',
]


# ---------------------------------------------------------------------------
# 3D molecular graph construction
# ---------------------------------------------------------------------------

def mol_to_graph_3d(
    mol,
    y: float = 0.0,
    protein_feats: Optional[List[float]] = None,
) -> Optional[Data]:
    """
    Convert an RDKit molecule with a 3D conformer to a PyG Data object for DimeNet++.

    DimeNet++ requires:
      z   — atomic numbers, shape (N,), dtype Long
      pos — 3D atomic positions, shape (N, 3), dtype Float

    Returns None if the molecule is missing, has no atoms, or has no conformer.
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LigandDataset3D(InMemoryDataset):
    """
    In-memory PyG dataset of ligand 3D molecular graphs for DimeNet++.

    Each entry is built from a mol2 crystal structure file (which contains 3D
    coordinates). Entries whose mol2 files are missing, unparseable, or lack a
    3D conformer are silently dropped.

    When protein_feature_cols is provided, each Data object carries a
    protein_feats tensor of shape (1, n_protein_features) populated from the
    corresponding DataFrame columns. NaN values are replaced with 0.

    Attributes:
        pdb_codes (List[str]): PDB codes of successfully parsed entries.
        y_true    (np.ndarray): Experimental ΔG values for parsed entries.
    """

    def __init__(
        self,
        aff_df: pd.DataFrame,
        protein_feature_cols: Optional[List[str]] = None,
    ):
        super().__init__(root=None, transform=None, pre_transform=None)
        data_list = []
        pdb_codes = []
        y_values = []

        _prot_cols = [c for c in (protein_feature_cols or []) if c in aff_df.columns]
        if protein_feature_cols and len(_prot_cols) < len(protein_feature_cols):
            missing = set(protein_feature_cols) - set(_prot_cols)
            print(f"  Warning: protein feature columns not in CSV: {sorted(missing)}")
            print("  Run process_pdbind.py with --augment to compute contact features.")

        for row in aff_df.itertuples(index=False):
            try:
                mol2_path = get_complex_path(row.pdb_code) / f"{row.pdb_code}_ligand.mol2"
            except FileNotFoundError:
                mol2_path = STRUCTURES_DIR / row.pdb_code / f"{row.pdb_code}_ligand.mol2"
            mol = None
            if mol2_path.exists():
                mol = Chem.MolFromMol2File(str(mol2_path), removeHs=True)

            prot_feats = None
            if _prot_cols:
                raw = [float(getattr(row, c, np.nan)) for c in _prot_cols]
                prot_feats = [0.0 if np.isnan(v) else v for v in raw]

            graph = mol_to_graph_3d(mol, float(row.exp_affinity_kcal_mol), prot_feats)
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


def build_datasets_3d(
    train_aff: pd.DataFrame,
    val_aff: pd.DataFrame,
    test_aff: pd.DataFrame,
    protein_feature_cols: Optional[List[str]] = None,
) -> Tuple[LigandDataset3D, LigandDataset3D, LigandDataset3D]:
    """Build train/val/test LigandDataset3D instances from affinity DataFrames."""
    print("Building 3D graph datasets...")
    train_ds = LigandDataset3D(train_aff, protein_feature_cols=protein_feature_cols)
    val_ds   = LigandDataset3D(val_aff,   protein_feature_cols=protein_feature_cols)
    test_ds  = LigandDataset3D(test_aff,  protein_feature_cols=protein_feature_cols)
    print(f"  Train: {len(train_ds)} / Val: {len(val_ds)} / Test: {len(test_ds)} complexes parsed")
    dropped = (len(train_aff) - len(train_ds),
               len(val_aff) - len(val_ds),
               len(test_aff) - len(test_ds))
    if sum(dropped) > 0:
        print(f"  Dropped (missing/no-3D mol2): "
              f"{dropped[0]} train, {dropped[1]} val, {dropped[2]} test")
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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

    def forward(self, data: Data) -> torch.Tensor:
        # DimeNetPlusPlus returns (batch_size, out_channels) — global pooling is internal
        emb = self.gnn(data.z, data.pos, data.batch)
        if self.n_protein_features > 0 and hasattr(data, 'protein_feats') and data.protein_feats is not None:
            emb = torch.cat([emb, data.protein_feats], dim=-1)
        return self.head(emb).squeeze(-1)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def make_loader(dataset, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    use_pin = device.type == 'cuda'
    workers = 4 if device.type == 'cuda' else 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=use_pin,
        num_workers=workers,
        persistent_workers=(workers > 0),
    )


def train_epoch(
    model: DimeNetAffinity,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    criterion = nn.MSELoss(reduction='sum')
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_samples += batch.num_graphs
    return total_loss / max(n_samples, 1)


def evaluate(
    model: DimeNetAffinity,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            y_true_list.append(batch.y.squeeze().cpu().numpy())
            y_pred_list.append(pred.cpu().numpy())
    y_true = np.concatenate(y_true_list).ravel()
    y_pred = np.concatenate(y_pred_list).ravel()
    return y_true, y_pred


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    r_p, _ = pearsonr(y_pred, y_true)
    r_s, _ = spearmanr(y_pred, y_true)
    return {
        'pearson_r':  float(r_p),
        'spearman_r': float(r_s),
        'rmse':       float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae':        float(np.mean(np.abs(y_true - y_pred))),
    }


def train_dimenet(
    train_ds: LigandDataset3D,
    val_ds: LigandDataset3D,
    hyperparams: Dict,
    epochs: int,
    patience: int,
    device: torch.device,
    n_protein_features: int = 0,
) -> Tuple[DimeNetAffinity, Dict]:
    """
    Train DimeNetAffinity with early stopping on validation MSE.

    Returns (best_model, val_metrics_dict).
    """
    hp = {
        'hidden_channels': hyperparams.get('hidden_channels', 128),
        'out_channels':     hyperparams.get('out_channels', 128),
        'num_blocks':       hyperparams.get('num_blocks', 4),
        'int_emb_size':     hyperparams.get('int_emb_size', 64),
        'basis_emb_size':   hyperparams.get('basis_emb_size', 8),
        'out_emb_channels': hyperparams.get('out_emb_channels', 256),
        'dropout':          hyperparams.get('dropout', 0.1),
        'lr':               hyperparams.get('lr', 1e-3),
        'batch_size':       hyperparams.get('batch_size', 32),
    }

    model = DimeNetAffinity(
        hidden_channels=hp['hidden_channels'],
        out_channels=hp['out_channels'],
        num_blocks=hp['num_blocks'],
        int_emb_size=hp['int_emb_size'],
        basis_emb_size=hp['basis_emb_size'],
        out_emb_channels=hp['out_emb_channels'],
        n_protein_features=n_protein_features,
        dropout=hp['dropout'],
    ).to(device)

    train_loader = make_loader(train_ds, batch_size=hp['batch_size'], shuffle=True,  device=device)
    val_loader   = make_loader(val_ds,   batch_size=hp['batch_size'], shuffle=False, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=hp['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-6,
    )

    best_val_mse = float('inf')
    best_state = None
    patience_count = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        y_true_val, y_pred_val = evaluate(model, val_loader, device)
        val_mse = float(mean_squared_error(y_true_val, y_pred_val))
        scheduler.step(val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = copy.deepcopy(model.state_dict())
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

def optimize_dimenet_hyperparams(
    train_aff_df: pd.DataFrame,
    val_aff_df: pd.DataFrame,
    n_trials: int = 30,
    storage: Optional[str] = None,
    study_name: str = 'dimenet_affinity',
    device: Optional[torch.device] = None,
    output_dir: Optional[Path] = None,
    protein_feature_cols: Optional[List[str]] = None,
    n_protein_features: int = 0,
) -> Dict:
    """
    Optuna search for DimeNet++ hyperparameters using 3-fold KFold CV on train+val combined.

    Uses max 30 epochs per fold with patience 8 to keep runtime manageable.
    Objective: minimise mean val MSE across 3 folds.
    """
    if device is None:
        device = torch.device('cpu')

    combined_df = pd.concat([train_aff_df, val_aff_df], ignore_index=True)
    print(f"  Building combined dataset for Optuna ({len(combined_df)} complexes)...")
    combined_ds = LigandDataset3D(combined_df, protein_feature_cols=protein_feature_cols)
    n = len(combined_ds)
    print(f"  {n} complexes parsed. Running {n_trials} Optuna trials × 3-fold CV...")

    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    indices = np.arange(n)

    def objective(trial):
        hp = {
            'hidden_channels': trial.suggest_categorical('hidden_channels', [64, 128, 256]),
            'out_channels':    trial.suggest_categorical('out_channels',    [64, 128, 256]),
            'num_blocks':      trial.suggest_categorical('num_blocks',      [2, 3, 4, 6]),
            'int_emb_size':    trial.suggest_categorical('int_emb_size',    [32, 64]),
            'basis_emb_size':  trial.suggest_categorical('basis_emb_size',  [8, 16]),
            'out_emb_channels':trial.suggest_categorical('out_emb_channels',[64, 128, 256]),
            'dropout':         trial.suggest_categorical('dropout',         [0.0, 0.1, 0.2, 0.3]),
            'lr':              trial.suggest_categorical('lr',              [1e-4, 5e-4, 1e-3]),
            'batch_size':      trial.suggest_categorical('batch_size',      [16, 32, 64]),
        }
        fold_mses = []
        for train_idx, val_idx in kf.split(indices):
            train_sub = torch.utils.data.Subset(combined_ds, train_idx.tolist())
            val_sub   = torch.utils.data.Subset(combined_ds, val_idx.tolist())
            t_loader  = make_loader(train_sub, batch_size=hp['batch_size'], shuffle=True,  device=device)
            v_loader  = make_loader(val_sub,   batch_size=hp['batch_size'], shuffle=False, device=device)

            fold_model = DimeNetAffinity(
                hidden_channels=hp['hidden_channels'],
                out_channels=hp['out_channels'],
                num_blocks=hp['num_blocks'],
                int_emb_size=hp['int_emb_size'],
                basis_emb_size=hp['basis_emb_size'],
                out_emb_channels=hp['out_emb_channels'],
                n_protein_features=n_protein_features,
                dropout=hp['dropout'],
            ).to(device)
            opt = torch.optim.Adam(fold_model.parameters(), lr=hp['lr'], weight_decay=1e-5)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=5, factor=0.5)

            best_fold_mse = float('inf')
            p_count = 0
            for _ in range(30):
                train_epoch(fold_model, t_loader, opt, device)
                y_t, y_p = evaluate(fold_model, v_loader, device)
                mse = float(mean_squared_error(y_t, y_p))
                sched.step(mse)
                if mse < best_fold_mse:
                    best_fold_mse = mse
                    p_count = 0
                else:
                    p_count += 1
                if p_count >= 8:
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
        print(f"  Loaded {n_existing} existing Optuna trials from storage")
    total_trials = n_existing + n_trials

    def _trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        print(f"  Trial {trial.number + 1:3d}/{total_trials}  "
              f"MSE={trial.value:.6f}  best={study.best_value:.6f}  "
              f"{trial.params}")

    study.optimize(objective, n_trials=n_trials, callbacks=[_trial_callback])
    print(f"  DimeNet++ best CV MSE: {study.best_value:.6f} (over {len(study.trials)} total trials)")
    print(f"  Best params: {study.best_params}")
    if output_dir is not None:
        _plot_optuna_trials(study, output_dir)
    return study.best_params


def _plot_optuna_trials(study: optuna.Study, output_dir: Path) -> None:
    values = [t.value for t in study.trials if t.value is not None]
    if not values:
        return
    trials_x = list(range(1, len(values) + 1))
    best_so_far = np.minimum.accumulate(values)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(trials_x, values, s=25, alpha=0.6, label='Trial CV MSE')
    ax.plot(trials_x, best_so_far, color='crimson', lw=2, label='Best so far')
    ax.set_xlabel('Trial number')
    ax.set_ylabel('CV MSE (kcal/mol)²')
    ax.set_title('Optuna DimeNet++ hyperparameter search')
    ax.legend()
    fig.tight_layout()
    path = output_dir / 'dimenet_optuna_trials.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved Optuna trial plot to {path}")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def plot_dimenet_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict,
    output_dir: Path,
    split: str = 'test',
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=20, color='teal')
    lims = [min(y_true.min(), y_pred.min()) - 0.5,
            max(y_true.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.6)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('Experimental ΔG (kcal/mol)')
    ax.set_ylabel('Predicted ΔG (kcal/mol)')
    ax.set_title(f'DimeNet++ Affinity Predictions ({split.capitalize()} set)')
    ax.text(
        0.05, 0.93,
        f"Pearson r = {metrics['pearson_r']:+.3f}\n"
        f"Spearman r = {metrics['spearman_r']:+.3f}\n"
        f"RMSE = {metrics['rmse']:.3f} kcal/mol",
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )
    path = output_dir / f'dimenet_affinity_predictions_{split}.png'
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved DimeNet++ plot to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='DimeNet++ GNN binding affinity prediction with protein feature augmentation.',
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
        help='Run Optuna hyperparameter search (3-fold CV, 30 epochs/fold)',
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
        help='SQLite path for persisting/resuming Optuna trials',
    )
    parser.add_argument(
        '--load-hyperparams',
        type=str,
        default=None,
        metavar='JSON_PATH',
        help='Load previously optimised hyperparameters from JSON',
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
        '--protein-feature-cols',
        nargs='+',
        default=None,
        metavar='COL',
        help='Protein scalar feature column names from the CSV. '
             'Defaults to the 12 contact columns (contact_n_3A, ...). '
             'These are present when process_pdbind.py was run with --augment.',
    )
    parser.add_argument(
        '--no-protein-features',
        action='store_true',
        help='Train in ligand-only mode (no protein features concatenated to the embedding)',
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

    # --- Protein feature columns ---
    if args.no_protein_features:
        protein_feature_cols = []
    else:
        protein_feature_cols = args.protein_feature_cols or _DEFAULT_PROTEIN_FEATURE_COLS

    # --- Load CSVs ---
    print("\nLoading data from CSV files...")
    csvs = args.load_csv
    val_csv = Path(csvs[2]) if len(csvs) >= 3 else None
    train_df, val_df, test_df = load_data_from_csv(
        Path(csvs[0]), Path(csvs[1]), val_csv,
    )
    if val_df is None:
        train_df, val_df = split_val_from_train(train_df, args.val_frac)
        print(f"  Val set carved from training: {val_df['pdb_code'].nunique()} complexes")

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

    # Determine which protein feature cols are actually present in the data
    present_prot_cols = [c for c in protein_feature_cols if c in train_aff.columns]
    n_protein_features = len(present_prot_cols)
    if protein_feature_cols and not present_prot_cols:
        print("  Warning: no protein feature columns found in CSV. "
              "Running in ligand-only mode. Re-run process_pdbind.py with --augment "
              "to include contact features.")
    elif present_prot_cols:
        print(f"\nProtein features ({n_protein_features}): {present_prot_cols}")

    # --- Build 3D graph datasets ---
    print()
    train_ds, val_ds, test_ds = build_datasets_3d(
        train_aff, val_aff, test_aff,
        protein_feature_cols=present_prot_cols or None,
    )

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
        print(f"\nOptimising DimeNet++ hyperparameters "
              f"(Optuna, {args.n_trials} trials × 3-fold CV, max 30 epochs/fold)...")
        hyperparams = optimize_dimenet_hyperparams(
            train_aff, val_aff,
            n_trials=args.n_trials,
            storage=optuna_storage,
            study_name='dimenet_affinity',
            device=device,
            output_dir=OUTPUT_DIR if not args.no_plots else None,
            protein_feature_cols=present_prot_cols or None,
            n_protein_features=n_protein_features,
        )
        hp_save = OUTPUT_DIR / 'dimenet_best_hyperparams.json'
        hp_save.write_text(json.dumps(hyperparams, indent=2))
        print(f"Saved DimeNet++ hyperparameters to {hp_save}")

    # --- Final training ---
    print(f"\nTraining final DimeNet++ model "
          f"(max {args.epochs} epochs, patience {args.patience})...")
    print(f"  n_protein_features: {n_protein_features}")
    print(f"  Hyperparams: {hyperparams or 'defaults'}")
    model, val_metrics = train_dimenet(
        train_ds, val_ds, hyperparams,
        epochs=args.epochs, patience=args.patience, device=device,
        n_protein_features=n_protein_features,
    )

    # --- Evaluate on test set ---
    batch_size = hyperparams.get('batch_size', 32)
    test_loader = make_loader(test_ds, batch_size=batch_size, shuffle=False, device=device)
    y_true_test, y_pred_test = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(y_true_test, y_pred_test)

    # --- Print results ---
    print(f"\n{'='*60}")
    print("DIMENET++ BINDING AFFINITY PERFORMANCE")
    print(f"{'='*60}")
    print(f"Val: {len(val_ds)} complexes   Test: {len(test_ds)} complexes")
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
    model_path = OUTPUT_DIR / 'dimenet_model.pt'
    torch.save({
        'state_dict':           model.state_dict(),
        'hyperparams':          hyperparams,
        'n_protein_features':   n_protein_features,
        'protein_feature_cols': present_prot_cols,
    }, model_path)
    print(f"\nSaved model to {model_path}")

    # --- Predictions CSV ---
    pred_df = pd.DataFrame({
        'pdb_code':                    test_ds.pdb_codes,
        'exp_affinity_kcal_mol':       y_true_test,
        'predicted_affinity_kcal_mol': y_pred_test,
    })
    pred_csv = OUTPUT_DIR / 'dimenet_affinity_predictions.csv'
    pred_df.to_csv(pred_csv, index=False)
    print(f"Saved predictions to {pred_csv}  ({len(pred_df)} complexes)")

    # --- Plots ---
    if not args.no_plots:
        try:
            plot_dimenet_predictions(
                val_metrics['y_true'], val_metrics['y_pred'],
                val_metrics, OUTPUT_DIR, split='val',
            )
            plot_dimenet_predictions(
                y_true_test, y_pred_test,
                test_metrics, OUTPUT_DIR, split='test',
            )
        except Exception as e:
            print(f"Warning: could not generate DimeNet++ plots: {e}")

    print("\nDone!")


if __name__ == '__main__':
    main()
