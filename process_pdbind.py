"""
process_pdbind.py

Process PDBbind dataset for molecular docking rescoring with XGBoost.

Workflow:
1. Load binding affinity data from PDBbind index
2. Run AutoDock Vina on protein-ligand complexes
3. Extract RDKit 2D molecular features
4. Train XGBoost model to rescore Vina poses
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


# Configuration
INDEX_DIR = Path("index")
STRUCTURES_DIR = Path("P-L")
OUTPUT_DIR = Path("output")
VINA_EXECUTABLE = Path("/Applications/autodock_vina_1_1_2_mac_catalina_64bit/bin/vina")  # Adjust path for your system

# Constants
NUM_TOP_POSES = 5
RANDOM_SEED = 42


def parse_binding_affinity(binding_str: str) -> Optional[float]:
    """
    Parse binding affinity string to get numeric value in nM.
    
    Args:
        binding_str: String like "Kd=49uM", "Ki<0.43uM", "IC50>100nM"
    
    Returns:
        Affinity in nM, or None if parsing fails
    """
    try:
        # Match patterns like Kd=49uM, Ki<0.43uM, IC50>100nM, Ki=0.43uM
        match = re.search(r'(Kd|Ki|IC50)([<>=])([0-9.]+)([munpM]+)', binding_str)
        if not match:
            return None
        
        value = float(match.group(3))
        # Extract just the first character (prefix) from unit (e.g., "uM" -> "u")
        unit = match.group(4).lower()[0]
        
        # Convert to nM
        if unit == 'm':
            return value * 1e6
        elif unit == 'u':
            return value * 1e3
        elif unit == 'n':
            return value
        elif unit == 'p':
            return value * 1e-3
        else:
            return None
    except Exception:
        return None


def load_binding_data() -> pd.DataFrame:
    """
    Load binding affinity data from PDBbind index file.
    
    Returns:
        DataFrame with columns: pdb_code, resolution, year, affinity_nM, reference
    """
    index_file = INDEX_DIR / "INDEX_general_PL.2020R1.lst"
    
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")
    
    records = []
    with open(index_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse line format: PDB_CODE  RESOLUTION  YEAR  BINDING_DATA  // REFERENCE
            parts = line.split()
            if len(parts) < 4:
                continue
            
            pdb_code = parts[0]
            resolution = parts[1]
            year = int(parts[2])
            
            # Binding data is everything between year and //
            binding_data = ''
            for part in parts[3:]:
                if part == '//':
                    break
                binding_data += part + ' '
            binding_data = binding_data.strip()
            
            affinity_nM = parse_binding_affinity(binding_data)
            
            if affinity_nM is not None:
                records.append({
                    'pdb_code': pdb_code,
                    'resolution': resolution,
                    'year': year,
                    'affinity_nM': affinity_nM,
                    'binding_data': binding_data
                })
    
    return pd.DataFrame(records)


def get_complexes() -> List[str]:
    """
    Get list of PDB codes that have both ligand and protein files.
    
    Returns:
        List of valid PDB codes
    """
    if not STRUCTURES_DIR.exists():
        raise FileNotFoundError(f"Structures directory not found: {STRUCTURES_DIR}")
    
    valid_complexes = []
    
    for pdb_dir in STRUCTURES_DIR.rglob("*"):
        if not pdb_dir.is_dir():
            continue
        
        pdb_code = pdb_dir.name
        ligand_file = pdb_dir / f"{pdb_code}_ligand.mol2"
        protein_file = pdb_dir / f"{pdb_code}_protein.pdb"
        
        if ligand_file.exists() and protein_file.exists():
            valid_complexes.append(pdb_code)
    
    # Sort by year order (oldest to newest) to process in chronological order
    valid_complexes.sort()
    
    return valid_complexes


def get_complex_path(pdb_code: str) -> Path:
    """
    Find the full path to a complex directory given its PDB code.
    
    Args:
        pdb_code: The PDB code to find
        
    Returns:
        Path to the complex directory
        
    Raises:
        FileNotFoundError: If the complex directory is not found
    """
    for pdb_dir in STRUCTURES_DIR.rglob(pdb_code):
        if pdb_dir.is_dir():
            return pdb_dir
    
    raise FileNotFoundError(f"Complex directory not found for {pdb_code}")


def define_binding_site(protein_pdb: str, ligand_mol2: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Define binding site box centered on ligand.
    
    Args:
        protein_pdb: Path to protein PDB file
        ligand_mol2: Path to ligand mol2 file
    
    Returns:
        Tuple of (center, size) for Vina box
    """
    # Load ligand to get coordinates
    mol = Chem.MolFromMol2File(ligand_mol2)
    if mol is None:
        raise ValueError(f"Could not parse ligand: {ligand_mol2}")
    
    # Get ligand coordinates
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    
    # Calculate center and size
    center = coords.mean(axis=0)
    size = coords.max(axis=0) - coords.min(axis=0) + 10  # Add padding
    
    # Ensure minimum size
    size = np.maximum(size, 20)
    
    return (float(center[0]), float(center[1]), float(center[2])), \
           (float(size[0]), float(size[1]), float(size[2]))


def extract_pose_coordinates(pdbqt_file: Path) -> List[np.ndarray]:
    """
    Extract atom coordinates for each pose from Vina PDBQT output.
    
    Args:
        pdbqt_file: Path to Vina output PDBQT file
    
    Returns:
        List of numpy arrays, one per pose, containing heavy atom coordinates
    """
    poses = []
    current_pose_atoms = []
    
    with open(pdbqt_file, 'r') as f:
        for line in f:
            line = line.rstrip()
            
            # New pose starts with MODEL
            if line.startswith('MODEL'):
                current_pose_atoms = []
            
            # End of pose
            elif line.startswith('ENDMDL'):
                if current_pose_atoms:
                    poses.append(np.array(current_pose_atoms))
                    current_pose_atoms = []
            
            # Atom line - extract coordinates for heavy atoms only
            elif line.startswith('ATOM') or line.startswith('HETATM'):
                # Check if it's a heavy atom (not hydrogen)
                atom_name = line[12:16].strip()
                if not atom_name.startswith('H'):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        current_pose_atoms.append([x, y, z])
                    except (ValueError, IndexError):
                        continue
    
    # Add last pose if file doesn't end with ENDMDL
    if current_pose_atoms:
        poses.append(np.array(current_pose_atoms))
    
    return poses


def extract_crystal_ligand_coords(pdb_code: str) -> Optional[np.ndarray]:
    """
    Extract heavy atom coordinates from crystal structure ligand.
    
    Args:
        pdb_code: PDB code of the complex
    
    Returns:
        Numpy array of heavy atom coordinates, or None if failed
    """
    try:
        pdb_dir = get_complex_path(pdb_code)
        ligand_mol2 = pdb_dir / f"{pdb_code}_ligand.mol2"
        
        if not ligand_mol2.exists():
            return None
        
        mol = Chem.MolFromMol2File(str(ligand_mol2))
        if mol is None:
            return None
        
        conf = mol.GetConformer()
        coords = []
        
        for atom in mol.GetAtoms():
            # Skip hydrogens
            if atom.GetAtomicNum() == 1:
                continue
            pos = conf.GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])
        
        return np.array(coords)
    except Exception as e:
        print(f"Warning: Could not extract crystal coordinates for {pdb_code}: {e}")
        return None


def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """
    Calculate RMSD between two sets of coordinates using Kabsch algorithm.
    
    Args:
        coords1: First set of coordinates (reference)
        coords2: Second set of coordinates (mobile)
    
    Returns:
        RMSD value
    """
    if coords1.shape != coords2.shape:
        print(f"Warning: Coordinate shape mismatch: {coords1.shape} vs {coords2.shape}")
        return float('inf')
    
    # Center the coordinates
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    
    centered1 = coords1 - centroid1
    centered2 = coords2 - centroid2
    
    # Calculate covariance matrix
    covariance_matrix = np.dot(centered2.T, centered1)
    
    # SVD
    V, S, Wt = np.linalg.svd(covariance_matrix)
    
    # Calculate rotation matrix
    d = np.linalg.det(np.dot(V, Wt))
    if d < 0:
        V[:, -1] = -V[:, -1]
    
    rotation_matrix = np.dot(V, Wt)
    
    # Rotate coords2
    rotated_coords2 = np.dot(centered2, rotation_matrix)
    
    # Calculate RMSD
    rmsd = np.sqrt(np.mean(np.sum((centered1 - rotated_coords2) ** 2, axis=1)))
    
    return rmsd


def calculate_all_pose_rmsds(vina_result: Dict, pdb_code: str) -> Optional[List[float]]:
    """
    Calculate RMSD between each Vina pose and crystal structure.
    
    Args:
        vina_result: Dictionary from run_vina()
        pdb_code: PDB code of the complex
    
    Returns:
        List of RMSD values for each pose, or None if failed
    """
    # Get crystal structure coordinates
    crystal_coords = extract_crystal_ligand_coords(pdb_code)
    if crystal_coords is None:
        return None
    
    # Get pose coordinates
    pose_coords = vina_result.get('pose_coordinates', [])
    if not pose_coords:
        return None
    
    # Check if atom counts match
    if crystal_coords.shape[0] != pose_coords[0].shape[0]:
        print(f"Warning: Atom count mismatch for {pdb_code}: crystal={crystal_coords.shape[0]}, pose={pose_coords[0].shape[0]}")
        return None
    
    # Calculate RMSD for each pose
    rmsds = []
    for pose in pose_coords:
        rmsd = calculate_rmsd(crystal_coords, pose)
        rmsds.append(rmsd)
    
    return rmsds


def run_vina(pdb_code: str) -> Optional[Dict]:
    """
    Run AutoDock Vina on a single complex.
    
    Args:
        pdb_code: PDB code of the complex
    
    Returns:
        Dictionary with docking results including pose coordinates, or None if failed
    """
    try:
        pdb_dir = get_complex_path(pdb_code)
    except FileNotFoundError:
        print(f"Warning: Could not find directory for {pdb_code}")
        return None
    
    protein_pdb = pdb_dir / f"{pdb_code}_protein.pdb"
    ligand_mol2 = pdb_dir / f"{pdb_code}_ligand.mol2"
    
    if not protein_pdb.exists() or not ligand_mol2.exists():
        print(f"Warning: Missing files for {pdb_code}")
        return None
    
    # Define binding site
    try:
        center, size = define_binding_site(str(protein_pdb), str(ligand_mol2))
    except Exception as e:
        print(f"Warning: Could not define binding site for {pdb_code}: {e}")
        return None
    
    # Create temporary files
    vina_input = OUTPUT_DIR / f"{pdb_code}_vina_input.pdbqt"
    vina_ligand = OUTPUT_DIR / f"{pdb_code}_vina_ligand.pdbqt"
    vina_output = OUTPUT_DIR / f"{pdb_code}_vina_output.pdbqt"
    
    # Prepare receptor (convert PDB to PDBQT using Open Babel)
    prepare_receptor = subprocess.run(
        ['obabel', '-ipdb', str(protein_pdb), '-opdbqt', '-O', str(vina_input), 
         '-xr', '--partialcharge', 'gasteiger'],
        capture_output=True,
        text=True
    )
    
    if prepare_receptor.returncode != 0:
        print(f"Warning: Failed to prepare receptor for {pdb_code}: {prepare_receptor.stderr}")
        return None
    
    # Prepare ligand (convert MOL2 to PDBQT using Open Babel)
    prepare_ligand = subprocess.run(
        ['obabel', '-imol2', str(ligand_mol2), '-opdbqt', '-O', str(vina_ligand),
         '--partialcharge', 'gasteiger'],
        capture_output=True,
        text=True
    )
    
    if prepare_ligand.returncode != 0:
        print(f"Warning: Failed to prepare ligand for {pdb_code}: {prepare_ligand.stderr}")
        return None
    
    # Run Vina
    vina_cmd = [
        str(VINA_EXECUTABLE),
        '--receptor', str(vina_input),
        '--ligand', str(vina_ligand),
        '--center_x', str(center[0]),
        '--center_y', str(center[1]),
        '--center_z', str(center[2]),
        '--size_x', str(size[0]),
        '--size_y', str(size[1]),
        '--size_z', str(size[2]),
        '--out', str(vina_output),
        '--num_modes', str(NUM_TOP_POSES),
        '--exhaustiveness', '8'
    ]
    
    result = subprocess.run(vina_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Warning: Vina failed for {pdb_code}: {result.stderr}")
        return None
    
    # Parse Vina output for scores
    scores = []
    with open(vina_output, 'r') as f:
        for line in f:
            if line.startswith('REMARK VINA RESULT:'):
                parts = line.split()
                if len(parts) >= 5:
                    score = float(parts[3])
                    scores.append(score)
    
    if len(scores) == 0:
        print(f"Warning: No poses generated for {pdb_code}")
        return None
    
    # Extract pose coordinates
    pose_coords = extract_pose_coordinates(vina_output)
    
    if len(pose_coords) != len(scores):
        print(f"Warning: Mismatch between number of scores ({len(scores)}) and poses ({len(pose_coords)}) for {pdb_code}")
        return None
    
    return {
        'pdb_code': pdb_code,
        'scores': scores,
        'pose_coordinates': pose_coords,
        'top_score': scores[0],
        'center': center,
        'size': size
    }


def extract_rdkit_features(ligand_mol2: str) -> Optional[Dict]:
    """
    Extract RDKit 2D molecular descriptors.
    
    Args:
        ligand_mol2: Path to ligand mol2 file
    
    Returns:
        Dictionary of molecular descriptors
    """
    mol = Chem.MolFromMol2File(ligand_mol2)
    if mol is None:
        return None
    
    try:
        features = {
            'mol_weight': Descriptors.MolWt(mol),
            'logP': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'num_h_donors': Descriptors.NumHDonors(mol),
            'num_h_acceptors': Descriptors.NumHAcceptors(mol),
            'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
            'fraction_csp3': Descriptors.FractionCSP3(mol),
            'num_heavy_atoms': mol.GetNumHeavyAtoms(),
            'num_rings': Descriptors.RingCount(mol),
            'num_heteroatoms': Descriptors.NumHeteroatoms(mol),
            'balaban_j': Descriptors.BalabanJ(mol),
            'bertz_ct': Descriptors.BertzCT(mol),
        }
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None


def extract_pose_features(vina_result: Dict, pose_idx: int, rmsd: float) -> Dict:
    """
    Extract features for a specific pose from Vina docking results.
    
    Args:
        vina_result: Dictionary from run_vina()
        pose_idx: Index of the pose (0-based)
        rmsd: RMSD of this pose to crystal structure
    
    Returns:
        Dictionary of pose-level features
    """
    scores = vina_result['scores']
    num_poses = len(scores)
    
    # Get the score for this specific pose
    pose_score = scores[pose_idx]
    
    # Calculate relative features
    score_rank = pose_idx + 1  # 1-based rank
    score_diff_from_best = pose_score - scores[0]
    score_diff_from_worst = scores[-1] - pose_score
    score_percentile = (num_poses - score_rank) / num_poses
    
    # Calculate score statistics relative to all poses
    score_mean = np.mean(scores)
    score_std = np.std(scores)
    score_zscore = (pose_score - score_mean) / (score_std + 1e-8)
    
    return {
        'vina_score': pose_score,
        'vina_rank': score_rank,
        'vina_score_diff_best': score_diff_from_best,
        'vina_score_diff_worst': score_diff_from_worst,
        'vina_score_percentile': score_percentile,
        'vina_score_zscore': score_zscore,
        'vina_num_poses': num_poses,
        'vina_score_range': scores[0] - scores[-1],
        'rmsd': rmsd
    }


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features computed from existing CSV columns.
    All features are computed per-complex to avoid data leakage.

    Args:
        df: DataFrame with vina score columns

    Returns:
        DataFrame with additional engineered features
    """
    new_df = df.copy()

    boltzmann_probs = np.zeros(len(df))
    gap_to_next = np.zeros(len(df))
    top2_gaps = np.zeros(len(df))

    kT = 0.592  # RT at 300K in kcal/mol (Vina scores in kcal/mol)

    for pdb_code, group in df.groupby('pdb_code'):
        idx = group.index
        scores = group['vina_score'].values

        # Boltzmann-weighted probability (physically motivated)
        # Vina scores are negative: more negative = more favorable
        exp_scores = np.exp(-scores / kT)
        boltzmann_probs[idx] = exp_scores / exp_scores.sum()

        # Gap from each pose to the next worse pose (in score order)
        # sorted_order[i] gives the index (within group) of rank i pose
        sorted_order = np.argsort(scores)  # ascending: most negative first
        group_gaps = np.zeros(len(scores))
        for j in range(len(sorted_order) - 1):
            group_gaps[sorted_order[j]] = scores[sorted_order[j + 1]] - scores[sorted_order[j]]
        gap_to_next[idx] = group_gaps

        # Top-2 score gap: how separated is the best pose from rank 2
        if len(scores) > 1:
            sorted_scores = np.sort(scores)
            top2_gaps[idx] = sorted_scores[1] - sorted_scores[0]

    new_df['boltzmann_prob'] = boltzmann_probs
    new_df['gap_to_next_pose'] = gap_to_next
    new_df['top2_score_gap'] = top2_gaps
    new_df['log_abs_score'] = np.log(np.abs(df['vina_score']) + 1)

    return new_df


def augment_with_pdbqt_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add pose-specific geometric features extracted from existing Vina PDBQT output files.
    Does not rerun Vina - reads already-computed output files.

    Features per pose:
        pose_rog: radius of gyration (compactness)
        pose_com_drift: distance of pose COM from rank-1 pose COM
        pose_max_extent: largest bounding box dimension
        pose_volume_approx: approximate bounding box volume

    Args:
        df: DataFrame with pdb_code and pose_idx columns

    Returns:
        DataFrame with additional geometry columns (NaN where files not found)
    """
    geo_cols = ['pose_rog', 'pose_com_drift', 'pose_max_extent', 'pose_volume_approx']
    new_df = df.copy()
    for col in geo_cols:
        new_df[col] = np.nan

    found = 0
    for pdb_code in df['pdb_code'].unique():
        pdbqt_file = OUTPUT_DIR / f"{pdb_code}_vina_output.pdbqt"
        if not pdbqt_file.exists():
            continue

        pose_coords_list = extract_pose_coordinates(pdbqt_file)
        if not pose_coords_list:
            continue

        found += 1
        ref_com = pose_coords_list[0].mean(axis=0)

        pose_feats = []
        for coords in pose_coords_list:
            if len(coords) == 0:
                pose_feats.append({col: np.nan for col in geo_cols})
                continue
            com = coords.mean(axis=0)
            centered = coords - com
            rog = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
            com_drift = np.linalg.norm(com - ref_com)
            extent = coords.max(axis=0) - coords.min(axis=0)
            pose_feats.append({
                'pose_rog': rog,
                'pose_com_drift': com_drift,
                'pose_max_extent': float(extent.max()),
                'pose_volume_approx': float(extent[0] * extent[1] * extent[2]),
            })

        complex_mask = new_df['pdb_code'] == pdb_code
        for row_idx in new_df[complex_mask].index:
            pose_idx = int(new_df.loc[row_idx, 'pose_idx'])
            if pose_idx < len(pose_feats):
                for col, val in pose_feats[pose_idx].items():
                    new_df.loc[row_idx, col] = val

    print(f"  Added PDBQT geometry features for {found}/{df['pdb_code'].nunique()} complexes")
    return new_df


def augment_with_mol2_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RDKit 2D molecular descriptors from mol2 files.
    Each descriptor is the same for all poses of the same ligand but helps the
    model understand which molecule types have unreliable Vina rankings.

    Args:
        df: DataFrame with pdb_code column

    Returns:
        DataFrame with additional molecular descriptor columns
    """
    mol2_cache: Dict[str, Dict] = {}

    for pdb_code in df['pdb_code'].unique():
        # Use direct path instead of slow rglob
        ligand_mol2 = STRUCTURES_DIR / pdb_code / f"{pdb_code}_ligand.mol2"
        if not ligand_mol2.exists():
            continue
        features = extract_rdkit_features(str(ligand_mol2))
        if features:
            mol2_cache[pdb_code] = features

    if not mol2_cache:
        print("  Warning: No mol2 features extracted")
        return df

    feat_names = list(next(iter(mol2_cache.values())).keys())
    new_df = df.copy()
    for feat in feat_names:
        new_df[feat] = new_df['pdb_code'].map(
            lambda x, f=feat: mol2_cache.get(x, {}).get(f, np.nan)
        )

    print(f"  Added {len(feat_names)} molecular features for "
          f"{len(mol2_cache)}/{df['pdb_code'].nunique()} complexes")
    return new_df


def prepare_training_data(vina_results: List[Dict]) -> pd.DataFrame:
    """
    Prepare pose-level feature matrix for training.
    Each pose becomes a training example with binary target (1=best pose, 0=other).
    
    Args:
        vina_results: List of Vina docking results with RMSD values
    
    Returns:
        DataFrame with pose-level features and binary target
    """
    records = []
    
    for result in vina_results:
        pdb_code = result['pdb_code']
        scores = result['scores']
        
        # Calculate RMSDs for all poses
        rmsds = calculate_all_pose_rmsds(result, pdb_code)
        if rmsds is None:
            print(f"Warning: Could not calculate RMSDs for {pdb_code}, skipping")
            continue
        
        if len(rmsds) != len(scores):
            print(f"Warning: RMSD/score count mismatch for {pdb_code}, skipping")
            continue
        
        # Find the best pose (lowest RMSD)
        best_pose_idx = np.argmin(rmsds)
        
        # Create a record for each pose
        for pose_idx in range(len(scores)):
            # Extract features for this pose
            features = extract_pose_features(result, pose_idx, rmsds[pose_idx])
            
            # Binary target: 1 if this is the best pose, 0 otherwise
            is_best_pose = 1 if pose_idx == best_pose_idx else 0
            
            record = {
                'pdb_code': pdb_code,
                'pose_idx': pose_idx,
                'is_best_pose': is_best_pose,
                'rmsd': rmsds[pose_idx]
            }
            record.update(features)
            
            records.append(record)
    
    return pd.DataFrame(records)


def train_rescoring_model(train_df: pd.DataFrame) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier to predict if a pose is the best (lowest RMSD).
    
    Args:
        train_df: DataFrame with pose-level features and is_best_pose target
    
    Returns:
        Trained XGBoost classifier
    """
    # Define feature columns (exclude identifiers and targets)
    feature_cols = [col for col in train_df.columns 
                   if col not in ['pdb_code', 'pose_idx', 'is_best_pose', 'rmsd']]
    
    X = train_df[feature_cols].values
    y = train_df['is_best_pose'].values
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate scale_pos_weight to handle class imbalance
    # (most poses are not the best pose)
    num_positives = np.sum(y == 1)
    num_negatives = np.sum(y == 0)
    scale_pos_weight = num_negatives / num_positives if num_positives > 0 else 1.0
    
    # Train classifier with regularization to reduce overfitting on small Vina-only features
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss'
    )

    model.fit(X, y)

    return model


def train_ranker_model(train_df: pd.DataFrame) -> Tuple[xgb.XGBRanker, List[str]]:
    """
    Train XGBoost ranker to order poses within each complex by predicted quality.
    Uses continuous RMSD-based relevance (1/(1+rmsd)) so the model learns to
    rank poses by proximity to the crystal structure rather than binary best/not-best.

    Args:
        train_df: DataFrame with pose-level features and rmsd column

    Returns:
        Tuple of (trained XGBRanker, list of feature column names)
    """
    feature_cols = [col for col in train_df.columns
                    if col not in ['pdb_code', 'pose_idx', 'is_best_pose', 'rmsd']]

    # Ranker requires data sorted by query group
    sorted_df = train_df.sort_values('pdb_code').reset_index(drop=True)

    X = sorted_df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Relevance: higher = better = lower RMSD. Maps RMSD in [0, inf) -> (0, 1].
    y_rel = 1.0 / (1.0 + sorted_df['rmsd'].values)

    # Query IDs: one integer per complex (same for all poses in a complex)
    pdb_codes = sorted_df['pdb_code'].values
    _, qid = np.unique(pdb_codes, return_inverse=True)

    ranker = xgb.XGBRanker(
        objective='rank:pairwise',
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    ranker.fit(X, y_rel, qid=qid)

    return ranker, feature_cols


def train_rf_ranker(train_df: pd.DataFrame) -> Tuple[RandomForestRegressor, List[str]]:
    """
    Train a Random Forest regressor as a pose ranker.
    Predicts 1/(1+rmsd) per pose; the pose with the highest predicted score is selected.

    Args:
        train_df: DataFrame with pose-level features and rmsd column

    Returns:
        Tuple of (trained RandomForestRegressor, list of feature column names)
    """
    feature_cols = [col for col in train_df.columns
                    if col not in ['pdb_code', 'pose_idx', 'is_best_pose', 'rmsd']]

    X = train_df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_rel = 1.0 / (1.0 + train_df['rmsd'].values)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X, y_rel)
    return model, feature_cols


def train_gb_ranker(train_df: pd.DataFrame) -> Tuple[GradientBoostingRegressor, List[str]]:
    """
    Train a Gradient Boosting regressor as a pose ranker.
    Predicts 1/(1+rmsd) per pose; the pose with the highest predicted score is selected.

    Args:
        train_df: DataFrame with pose-level features and rmsd column

    Returns:
        Tuple of (trained GradientBoostingRegressor, list of feature column names)
    """
    feature_cols = [col for col in train_df.columns
                    if col not in ['pdb_code', 'pose_idx', 'is_best_pose', 'rmsd']]

    X = train_df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_rel = 1.0 / (1.0 + train_df['rmsd'].values)

    model = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=RANDOM_SEED,
    )
    model.fit(X, y_rel)
    return model, feature_cols


def evaluate_model(model: xgb.XGBClassifier, test_df: pd.DataFrame) -> Dict:
    """
    Evaluate trained classifier on test set for pose selection.
    
    Args:
        model: Trained XGBoost classifier
        test_df: DataFrame with test features
    
    Returns:
        Dictionary of evaluation metrics including success rate
    """
    feature_cols = [col for col in test_df.columns 
                   if col not in ['pdb_code', 'pose_idx', 'is_best_pose', 'rmsd']]
    
    X_test = test_df[feature_cols].values
    y_test = test_df['is_best_pose'].values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Get prediction probabilities
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of being best pose
    
    # Group by complex to evaluate pose selection per complex
    test_df_copy = test_df.copy()
    test_df_copy['predicted_proba'] = y_proba
    
    # Calculate success rate
    vina_success = 0
    xgb_success = 0
    total_complexes = 0
    
    # Track RMSD of selected poses
    vina_selected_rmsds = []
    xgb_selected_rmsds = []
    
    for pdb_code in test_df_copy['pdb_code'].unique():
        complex_df = test_df_copy[test_df_copy['pdb_code'] == pdb_code]
        
        if len(complex_df) == 0:
            continue
        
        total_complexes += 1
        
        # Find the actual best pose (lowest RMSD)
        actual_best_idx = complex_df['rmsd'].idxmin()
        
        # Vina's choice: lowest score (rank 1)
        vina_best_idx = complex_df['vina_rank'].idxmin()
        vina_selected_rmsds.append(complex_df.loc[vina_best_idx, 'rmsd'])
        
        # XGBoost's choice: highest probability
        xgb_best_idx = complex_df['predicted_proba'].idxmax()
        xgb_selected_rmsds.append(complex_df.loc[xgb_best_idx, 'rmsd'])
        
        # Check if they selected the actual best pose
        if vina_best_idx == actual_best_idx:
            vina_success += 1
        if xgb_best_idx == actual_best_idx:
            xgb_success += 1
    
    # Calculate metrics
    vina_success_rate = vina_success / total_complexes if total_complexes > 0 else 0
    xgb_success_rate = xgb_success / total_complexes if total_complexes > 0 else 0

    mean_vina_rmsd = np.mean(vina_selected_rmsds) if vina_selected_rmsds else 0
    mean_xgb_rmsd = np.mean(xgb_selected_rmsds) if xgb_selected_rmsds else 0

    # 2 Å success rate (standard docking benchmark: pose within 2 Å of crystal)
    vina_success_2A = sum(r < 2.0 for r in vina_selected_rmsds)
    xgb_success_2A = sum(r < 2.0 for r in xgb_selected_rmsds)

    return {
        'total_complexes': total_complexes,
        'vina_success_rate': vina_success_rate,
        'xgb_success_rate': xgb_success_rate,
        'vina_success_count': vina_success,
        'xgb_success_count': xgb_success,
        'mean_vina_rmsd': mean_vina_rmsd,
        'mean_xgb_rmsd': mean_xgb_rmsd,
        'vina_success_2A_rate': vina_success_2A / total_complexes if total_complexes > 0 else 0,
        'xgb_success_2A_rate': xgb_success_2A / total_complexes if total_complexes > 0 else 0,
        'vina_success_2A_count': vina_success_2A,
        'xgb_success_2A_count': xgb_success_2A,
        'vina_selected_rmsds': vina_selected_rmsds,
        'xgb_selected_rmsds': xgb_selected_rmsds,
        'test_df': test_df_copy
    }


def evaluate_ranker(ranker: xgb.XGBRanker, test_df: pd.DataFrame,
                    feature_cols: List[str]) -> Dict:
    """
    Evaluate XGBoost ranker for pose selection.
    Predicts a relevance score per pose; the pose with the highest score is selected.

    Args:
        ranker: Trained XGBRanker
        test_df: DataFrame with test features and rmsd column
        feature_cols: List of feature column names used during training

    Returns:
        Dictionary of evaluation metrics (same keys as evaluate_model)
    """
    X_test = test_df[feature_cols].values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    ranker_scores = ranker.predict(X_test)

    test_df_copy = test_df.copy()
    test_df_copy['predicted_proba'] = ranker_scores  # reuse field for plot compatibility

    vina_success = 0
    xgb_success = 0
    total_complexes = 0
    vina_selected_rmsds = []
    xgb_selected_rmsds = []

    for pdb_code in test_df_copy['pdb_code'].unique():
        complex_df = test_df_copy[test_df_copy['pdb_code'] == pdb_code]
        if len(complex_df) == 0:
            continue

        total_complexes += 1
        actual_best_idx = complex_df['rmsd'].idxmin()
        vina_best_idx = complex_df['vina_rank'].idxmin()
        xgb_best_idx = complex_df['predicted_proba'].idxmax()

        vina_rmsd = complex_df.loc[vina_best_idx, 'rmsd']
        xgb_rmsd = complex_df.loc[xgb_best_idx, 'rmsd']

        vina_selected_rmsds.append(vina_rmsd)
        xgb_selected_rmsds.append(xgb_rmsd)

        if vina_best_idx == actual_best_idx:
            vina_success += 1
        if xgb_best_idx == actual_best_idx:
            xgb_success += 1

    vina_success_2A = sum(r < 2.0 for r in vina_selected_rmsds)
    xgb_success_2A = sum(r < 2.0 for r in xgb_selected_rmsds)

    return {
        'total_complexes': total_complexes,
        'vina_success_rate': vina_success / total_complexes if total_complexes > 0 else 0,
        'xgb_success_rate': xgb_success / total_complexes if total_complexes > 0 else 0,
        'vina_success_count': vina_success,
        'xgb_success_count': xgb_success,
        'mean_vina_rmsd': np.mean(vina_selected_rmsds) if vina_selected_rmsds else 0,
        'mean_xgb_rmsd': np.mean(xgb_selected_rmsds) if xgb_selected_rmsds else 0,
        'vina_success_2A_rate': vina_success_2A / total_complexes if total_complexes > 0 else 0,
        'xgb_success_2A_rate': xgb_success_2A / total_complexes if total_complexes > 0 else 0,
        'vina_success_2A_count': vina_success_2A,
        'xgb_success_2A_count': xgb_success_2A,
        'vina_selected_rmsds': vina_selected_rmsds,
        'xgb_selected_rmsds': xgb_selected_rmsds,
        'test_df': test_df_copy,
    }


def plot_rmsd_distribution_comparison(metrics: Dict, output_dir: Path) -> None:
    """
    Plot 1: RMSD distribution comparison between Vina and XGBoost selected poses.
    
    Args:
        metrics: Dictionary from evaluate_model()
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram comparison
    ax = axes[0]
    bins = np.linspace(0, max(max(metrics['vina_selected_rmsds']), max(metrics['xgb_selected_rmsds'])) + 0.5, 20)
    ax.hist(metrics['vina_selected_rmsds'], bins=bins, alpha=0.6, label=f'Vina (mean={metrics["mean_vina_rmsd"]:.2f}Å)', color='orange')
    ax.hist(metrics['xgb_selected_rmsds'], bins=bins, alpha=0.6, label=f'XGBoost (mean={metrics["mean_xgb_rmsd"]:.2f}Å)', color='green')
    ax.set_xlabel('RMSD to Crystal Structure (Å)', fontsize=12)
    ax.set_ylabel('Number of Complexes', fontsize=12)
    ax.set_title('RMSD Distribution of Selected Poses', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Box plot comparison
    ax = axes[1]
    data = [metrics['vina_selected_rmsds'], metrics['xgb_selected_rmsds']]
    bp = ax.boxplot(data, labels=['Vina', 'XGBoost'], patch_artist=True)
    bp['boxes'][0].set_facecolor('orange')
    bp['boxes'][1].set_facecolor('green')
    ax.set_ylabel('RMSD to Crystal Structure (Å)', fontsize=12)
    ax.set_title('RMSD Comparison (Box Plot)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rmsd_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved RMSD distribution comparison to {output_dir / 'rmsd_distribution_comparison.png'}")


def plot_success_rate_by_rank(metrics: Dict, test_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot 2: Success rate breakdown by which Vina rank was actually best.
    
    Args:
        metrics: Dictionary from evaluate_model()
        test_df: DataFrame with test data including predicted_proba
        output_dir: Directory to save plots
    """
    # Count which Vina rank was actually best for each complex
    vina_rank_success = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    xgb_rank_success = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    for pdb_code in test_df['pdb_code'].unique():
        complex_df = test_df[test_df['pdb_code'] == pdb_code]
        
        # Find which rank was actually best
        best_rank = complex_df.loc[complex_df['rmsd'].idxmin(), 'vina_rank']
        vina_rank_success[best_rank] += 1
        
        # Find which rank XGBoost selected
        xgb_best_idx = complex_df['predicted_proba'].idxmax()
        xgb_rank = complex_df.loc[xgb_best_idx, 'vina_rank']
        xgb_rank_success[xgb_rank] += 1
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ranks = [1, 2, 3, 4, 5]
    vina_counts = [vina_rank_success[r] for r in ranks]
    xgb_counts = [xgb_rank_success[r] for r in ranks]
    
    x = np.arange(len(ranks))
    width = 0.35
    
    ax.bar(x - width/2, vina_counts, width, label='Vina Selected', color='orange', alpha=0.8)
    ax.bar(x + width/2, xgb_counts, width, label='XGBoost Selected', color='green', alpha=0.8)
    
    ax.set_xlabel('Vina Rank of Selected Pose', fontsize=12)
    ax.set_ylabel('Number of Complexes', fontsize=12)
    ax.set_title('Which Vina Rank Was Selected by Each Method', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Rank {r}' for r in ranks])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rate_by_rank.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved success rate by rank to {output_dir / 'success_rate_by_rank.png'}")


def plot_feature_importance(model: xgb.XGBClassifier, feature_names: List[str], output_dir: Path) -> None:
    """
    Plot 3: Feature importance from XGBoost model.
    
    Args:
        model: Trained XGBoost classifier
        feature_names: List of feature names
        output_dir: Directory to save plots
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    # Top 15 features
    top_n = min(15, len(feature_names))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(top_n)
    ax.barh(y_pos, importance[indices[:top_n]], color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices[:top_n]])
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Top 15 Most Important Features (XGBoost)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature importance plot to {output_dir / 'feature_importance.png'}")


def plot_roc_curve(test_df: pd.DataFrame, model: xgb.XGBClassifier, output_dir: Path) -> None:
    """
    Plot 4: ROC curve for binary classification (predicting best pose).
    
    Args:
        test_df: DataFrame with test data
        model: Trained XGBoost classifier
        output_dir: Directory to save plots
    """
    from sklearn.metrics import roc_curve, auc
    
    feature_cols = [col for col in test_df.columns 
                   if col not in ['pdb_code', 'pose_idx', 'is_best_pose', 'rmsd', 'predicted_proba']]
    
    X_test = test_df[feature_cols].values
    y_test = test_df['is_best_pose'].values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve: Predicting Best Pose', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to {output_dir / 'roc_curve.png'}")


def plot_precision_recall_curve(test_df: pd.DataFrame, model: xgb.XGBClassifier, output_dir: Path) -> None:
    """
    Plot 5: Precision-Recall curve for binary classification.
    
    Args:
        test_df: DataFrame with test data
        model: Trained XGBoost classifier
        output_dir: Directory to save plots
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    feature_cols = [col for col in test_df.columns 
                   if col not in ['pdb_code', 'pose_idx', 'is_best_pose', 'rmsd', 'predicted_proba']]
    
    X_test = test_df[feature_cols].values
    y_test = test_df['is_best_pose'].values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    # Calculate baseline (random classifier)
    baseline = np.sum(y_test) / len(y_test)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    ax.axhline(y=baseline, color='red', linestyle='--', label=f'Baseline (random = {baseline:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve: Predicting Best Pose', fontsize=12)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved precision-recall curve to {output_dir / 'precision_recall_curve.png'}")


def plot_rmsd_vs_probability(test_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Plot 6: RMSD vs XGBoost probability scatter plot.
    
    Args:
        test_df: DataFrame with test data including predicted_proba
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by whether it's the best pose
    best_poses = test_df[test_df['is_best_pose'] == 1]
    other_poses = test_df[test_df['is_best_pose'] == 0]
    
    ax.scatter(other_poses['predicted_proba'], other_poses['rmsd'], 
              alpha=0.5, s=50, color='gray', label='Other poses')
    ax.scatter(best_poses['predicted_proba'], best_poses['rmsd'], 
              alpha=0.8, s=80, color='red', label='Best pose (lowest RMSD)')
    
    ax.set_xlabel('XGBoost Probability (of being best pose)', fontsize=12)
    ax.set_ylabel('RMSD to Crystal Structure (Å)', fontsize=12)
    ax.set_title('XGBoost Confidence vs Actual RMSD', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'rmsd_vs_probability.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved RMSD vs probability plot to {output_dir / 'rmsd_vs_probability.png'}")


def export_pose_scores_csv(test_df: pd.DataFrame, vina_results: List[Dict], output_dir: Path) -> None:
    """
    Export compact CSV with all pose scores and predictions per complex.
    
    Args:
        test_df: DataFrame with test data including predictions
        vina_results: List of Vina results with affinity data
        output_dir: Directory to save CSV
    """
    records = []
    
    # Create a mapping from pdb_code to affinity
    affinity_map = {}
    for result in vina_results:
        if 'affinity_nM' in result:
            affinity_map[result['pdb_code']] = result['affinity_nM']
    
    for pdb_code in test_df['pdb_code'].unique():
        complex_df = test_df[test_df['pdb_code'] == pdb_code].sort_values('pose_idx')
        
        record = {
            'pdb_code': pdb_code,
            'affinity_nM': affinity_map.get(pdb_code, None),
            'log_affinity': np.log10(affinity_map.get(pdb_code, 0) + 1) if affinity_map.get(pdb_code) else None
        }
        
        # Add scores and predictions for each pose
        for _, row in complex_df.iterrows():
            pose_idx = int(row['pose_idx'])
            record[f'pose_{pose_idx}_vina_score'] = row['vina_score']
            record[f'pose_{pose_idx}_vina_rank'] = int(row['vina_rank'])
            record[f'pose_{pose_idx}_rmsd'] = row['rmsd']
            record[f'pose_{pose_idx}_ranker_score'] = row['predicted_proba']
            record[f'pose_{pose_idx}_is_best'] = int(row['is_best_pose'])
            
            # Add feature values for this pose
            for col in complex_df.columns:
                if col.startswith('vina_') and col not in ['vina_score', 'vina_rank']:
                    record[f'pose_{pose_idx}_{col}'] = row[col]
        
        # Add summary columns
        best_pose_idx = complex_df.loc[complex_df['rmsd'].idxmin(), 'pose_idx']
        vina_selected_idx = complex_df.loc[complex_df['vina_rank'].idxmin(), 'pose_idx']
        xgb_selected_idx = complex_df.loc[complex_df['predicted_proba'].idxmax(), 'pose_idx']
        
        record['actual_best_pose_idx'] = int(best_pose_idx)
        record['vina_selected_pose_idx'] = int(vina_selected_idx)
        record['xgb_selected_pose_idx'] = int(xgb_selected_idx)
        record['vina_selected_rmsd'] = complex_df.loc[complex_df['vina_rank'].idxmin(), 'rmsd']
        record['xgb_selected_rmsd'] = complex_df.loc[complex_df['predicted_proba'].idxmax(), 'rmsd']
        record['vina_correct'] = 1 if vina_selected_idx == best_pose_idx else 0
        record['xgb_correct'] = 1 if xgb_selected_idx == best_pose_idx else 0
        
        records.append(record)
    
    df = pd.DataFrame(records)
    output_file = output_dir / 'pose_scores_detailed.csv'
    df.to_csv(output_file, index=False)
    print(f"Saved detailed pose scores to {output_file}")
    print(f"  Total complexes: {len(df)}")
    print(f"  Vina accuracy: {df['vina_correct'].mean():.3f}")
    print(f"  Best model accuracy: {df['xgb_correct'].mean():.3f}")


def export_score_comparison_csv(model_results: Dict, output_dir: Path) -> None:
    """
    Export a long-format per-pose CSV comparing all ranker model scores to Vina scores.
    One row per pose (not per complex), making it easy to inspect score distributions
    and compare model rankings directly against Vina.

    Columns: pdb_code, pose_idx, rmsd, is_best_pose, vina_score, vina_rank,
             vina_selected, {model}_score, {model}_rank, {model}_selected for each model.

    Args:
        model_results: Dict mapping model label to evaluate_ranker() result dict.
                       Each result dict must contain 'test_df' with 'predicted_proba' column.
        output_dir: Directory to save the CSV
    """
    # Build base dataframe from the first model's test_df (all share same base rows)
    first_result = next(iter(model_results.values()))
    base_df = first_result['test_df']

    out = base_df[['pdb_code', 'pose_idx', 'rmsd', 'is_best_pose',
                   'vina_score', 'vina_rank']].copy().reset_index(drop=True)

    # Vina's selected pose (rank 1 within each complex)
    out['vina_selected'] = (out['vina_rank'] == 1).astype(int)

    for label, metrics in model_results.items():
        scores = metrics['test_df']['predicted_proba'].values
        col_score = f'{label}_score'
        col_rank = f'{label}_rank'
        col_sel = f'{label}_selected'

        out[col_score] = scores

        # Rank within each complex: 1 = highest model score = predicted best pose
        out[col_rank] = out.groupby('pdb_code')[col_score] \
                           .rank(ascending=False, method='first').astype(int)
        out[col_sel] = (out[col_rank] == 1).astype(int)

    path = output_dir / 'score_comparison.csv'
    out.to_csv(path, index=False)
    print(f"Saved per-pose score comparison to {path}  ({len(out)} rows, {len(out.columns)} columns)")

    # Print a quick summary using metrics already computed by evaluate_ranker
    vina_rate = out.groupby('pdb_code').apply(
        lambda g: int(g.loc[g['vina_rank'].idxmin(), 'rmsd'] == g['rmsd'].min())
    ).mean()
    print(f"  Vina selects correct pose : {vina_rate:.3f}")
    for label, metrics in model_results.items():
        print(f"  {label} selects correct pose: {metrics['xgb_success_rate']:.3f}")


def load_data_from_csv(train_csv: Path, test_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load existing training and test data from CSV files.
    
    Args:
        train_csv: Path to training data CSV
        test_csv: Path to test data CSV
    
    Returns:
        Tuple of (train_df, test_df)
    """
    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    print(f"Loaded training data: {train_df.shape}")
    print(f"Loaded test data: {test_df.shape}")
    
    return train_df, test_df


def process_all_complexes(num_complexes: Optional[int] = None) -> List[Dict]:
    """
    Process all complexes and run Vina docking.
    
    Args:
        num_complexes: Limit number of complexes (None for all)
    
    Returns:
        List of Vina results with pose coordinates and RMSDs
    """
    print("Loading binding data...")
    binding_data = load_binding_data()
    print(f"Loaded {len(binding_data)} binding affinity records")
    
    print("Finding valid complexes...")
    complexes = get_complexes()
    print(f"Found {len(complexes)} valid complexes")
    
    if num_complexes:
        complexes = complexes[:num_complexes]
    
    vina_results = []
    
    for i, pdb_code in enumerate(complexes):
        print(f"Processing {pdb_code} ({i+1}/{len(complexes)})...")
        
        # Check if complex has binding data
        binding_row = binding_data[binding_data['pdb_code'] == pdb_code]
        if len(binding_row) == 0:
            print(f"  No binding data for {pdb_code}, skipping")
            continue
        
        # Run Vina
        vina_result = run_vina(pdb_code)
        if vina_result:
            # Calculate RMSDs
            rmsds = calculate_all_pose_rmsds(vina_result, pdb_code)
            if rmsds is not None:
                vina_result['rmsds'] = rmsds
                vina_result['affinity_nM'] = binding_row['affinity_nM'].values[0]
                vina_results.append(vina_result)
            else:
                print(f"  Could not calculate RMSDs for {pdb_code}, skipping")
    
    return vina_results


def main():
    """
    Main function to run the complete pose rescoring workflow.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Process PDBbind dataset for molecular docking pose rescoring with XGBoost'
    )
    parser.add_argument(
        '--load-csv',
        nargs=2,
        metavar=('TRAIN_CSV', 'TEST_CSV'),
        help='Load existing training and test CSV files instead of running Vina'
    )
    parser.add_argument(
        '--num-complexes',
        type=int,
        default=10,
        help='Number of complexes to process (default: 10)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating analysis plots'
    )
    parser.add_argument(
        '--no-augment',
        action='store_true',
        help='Skip augmenting features from mol2 and PDBQT files (faster but less accurate)'
    )
    args = parser.parse_args()

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.load_csv:
        # Load existing CSV files
        print("Loading data from CSV files...")
        train_csv = Path(args.load_csv[0])
        test_csv = Path(args.load_csv[1])
        train_df, test_df = load_data_from_csv(train_csv, test_csv)
    else:
        # Process complexes with Vina
        num_complexes = args.num_complexes
        vina_results = process_all_complexes(num_complexes)

        print(f"\nProcessed {len(vina_results)} complexes successfully")

        # Prepare training data
        print("Preparing training data...")
        train_df = prepare_training_data(vina_results)
        print(f"Training data shape: {train_df.shape}")

        if len(train_df) < 10:
            print("Not enough data for training")
            return

        # Split data by complex (not by pose) to avoid data leakage
        unique_complexes = train_df['pdb_code'].unique()
        train_complexes, test_complexes = train_test_split(
            unique_complexes, test_size=0.2, random_state=RANDOM_SEED
        )

        train_df_final = train_df[train_df['pdb_code'].isin(train_complexes)]
        test_df = train_df[train_df['pdb_code'].isin(test_complexes)]

        # Save results
        train_df_final.to_csv(OUTPUT_DIR / "training_data.csv", index=False)
        test_df.to_csv(OUTPUT_DIR / "test_data.csv", index=False)
        print(f"Results saved to {OUTPUT_DIR}")

        train_df = train_df_final

    # --- Feature augmentation ---
    if not args.no_augment:
        print("\nAugmenting features (this may take a few minutes)...")

        print("  Adding pose geometry from PDBQT files...")
        train_df = augment_with_pdbqt_features(train_df)
        test_df = augment_with_pdbqt_features(test_df)

        print("  Adding molecular descriptors from mol2 files...")
        train_df = augment_with_mol2_features(train_df)
        test_df = augment_with_mol2_features(test_df)

        print("  Engineering derived features...")
        train_df = engineer_features(train_df)
        test_df = engineer_features(test_df)

        print(f"  Feature count after augmentation: {train_df.shape[1]} columns")

    # --- Train all ranker models ---
    models_to_train = {
        'xgb_ranker': train_ranker_model,
        'rf_ranker':  train_rf_ranker,
        'gb_ranker':  train_gb_ranker,
    }
    trained = {}
    for name, train_fn in models_to_train.items():
        print(f"\nTraining {name}...")
        trained[name] = train_fn(train_df)  # returns (model, feature_cols)

    # --- Evaluate all models ---
    model_results = {}
    for name, (model, feat_cols) in trained.items():
        print(f"Evaluating {name}...")
        model_results[name] = evaluate_ranker(model, test_df, feat_cols)

    # Choose best model by 2 Å success rate (tie-break: best-pose selection rate)
    best_name = max(
        model_results,
        key=lambda k: (model_results[k]['xgb_success_2A_rate'],
                       model_results[k]['xgb_success_rate'])
    )
    best_metrics = model_results[best_name]
    best_model, _ = trained[best_name]

    # --- Print results ---
    n = best_metrics['total_complexes']
    print(f"\n{'='*60}")
    print("POSE SELECTION PERFORMANCE")
    print(f"{'='*60}")
    print(f"Total test complexes: {n}")
    print(f"\nAutoDock Vina (baseline):")
    print(f"  Best-pose selection rate : {best_metrics['vina_success_rate']:.3f} "
          f"({best_metrics['vina_success_count']}/{n})")
    print(f"  Poses within 2 Å        : {best_metrics['vina_success_2A_rate']:.3f} "
          f"({best_metrics['vina_success_2A_count']}/{n})")
    print(f"  Mean RMSD of selected   : {best_metrics['mean_vina_rmsd']:.3f} Å")

    for name, metrics in model_results.items():
        print(f"\n--- {name} ---")
        print(f"  Best-pose selection rate : {metrics['xgb_success_rate']:.3f} "
              f"({metrics['xgb_success_count']}/{n})")
        print(f"  Poses within 2 Å        : {metrics['xgb_success_2A_rate']:.3f} "
              f"({metrics['xgb_success_2A_count']}/{n})")
        print(f"  Mean RMSD of selected   : {metrics['mean_xgb_rmsd']:.3f} Å")

    print(f"\n{'='*60}")
    print(f"Best model: {best_name}")

    # --- Save performance metrics ---
    metrics_file = OUTPUT_DIR / 'pose_selection_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("POSE SELECTION PERFORMANCE METRICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total test complexes: {n}\n\n")
        f.write("AutoDock Vina (baseline):\n")
        f.write(f"  Best-pose selection rate : {best_metrics['vina_success_rate']:.3f} "
                f"({best_metrics['vina_success_count']}/{n})\n")
        f.write(f"  Poses within 2 Å        : {best_metrics['vina_success_2A_rate']:.3f} "
                f"({best_metrics['vina_success_2A_count']}/{n})\n")
        f.write(f"  Mean RMSD               : {best_metrics['mean_vina_rmsd']:.3f} Å\n\n")
        for name, metrics in model_results.items():
            f.write(f"{name}{'  <-- best' if name == best_name else ''}:\n")
            f.write(f"  Best-pose selection rate : {metrics['xgb_success_rate']:.3f} "
                    f"({metrics['xgb_success_count']}/{n})\n")
            f.write(f"  Poses within 2 Å        : {metrics['xgb_success_2A_rate']:.3f} "
                    f"({metrics['xgb_success_2A_count']}/{n})\n")
            f.write(f"  Mean RMSD               : {metrics['mean_xgb_rmsd']:.3f} Å\n\n")

    print(f"Saved metrics to {metrics_file}")

    # --- Export per-pose score comparison CSV ---
    print("\nExporting score comparison CSV...")
    export_score_comparison_csv(model_results, OUTPUT_DIR)

    # --- Export detailed pose scores CSV (best model) ---
    print("\nExporting detailed pose scores...")
    if args.load_csv:
        vina_results_for_export = []
    else:
        test_complexes_list = test_df['pdb_code'].unique()
        vina_results_for_export = [vr for vr in vina_results if vr['pdb_code'] in test_complexes_list]

    export_pose_scores_csv(best_metrics['test_df'], vina_results_for_export, OUTPUT_DIR)

    # --- Generate plots (best model) ---
    if not args.no_plots:
        print("\nGenerating visualization plots...")
        try:
            feature_cols = [col for col in test_df.columns
                            if col not in ['pdb_code', 'pose_idx', 'is_best_pose', 'rmsd', 'predicted_proba']]

            plot_rmsd_distribution_comparison(best_metrics, OUTPUT_DIR)
            plot_success_rate_by_rank(best_metrics, best_metrics['test_df'], OUTPUT_DIR)
            plot_feature_importance(best_model, feature_cols, OUTPUT_DIR)
            plot_rmsd_vs_probability(best_metrics['test_df'], OUTPUT_DIR)

            print("\nAll plots saved successfully!")
        except Exception as e:
            print(f"\nWarning: Could not generate some plots: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
