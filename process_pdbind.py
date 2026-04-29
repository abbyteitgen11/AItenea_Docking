"""
process_pdbind.py

Process PDBbind dataset for molecular docking rescoring with XGBoost.

Workflow:
1. Load binding affinity data from PDBbind index
2. Run AutoDock Vina on protein-ligand complexes
3. Extract RDKit 2D molecular features
4. Train XGBoost model to rescore Vina poses
"""

import math
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
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import json
import optuna
from sklearn.model_selection import train_test_split, GroupKFold, KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial import KDTree
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


# Configuration
INDEX_DIR      = Path("PDBind_2020_index")
STRUCTURES_DIR = Path("PDBind_2020")
CASF_DIR       = Path("CASF-2016/coreset")
OUTPUT_DIR     = Path("output")
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


def join_affinity_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join experimental binding affinity onto a per-pose DataFrame.

    Calls load_binding_data() to read the PDBbind index file, converts
    affinity_nM to pKd and ΔG (kcal/mol), then left-merges onto df by
    pdb_code.  Complexes absent from the index get NaN for both affinity
    columns.

    Args:
        df: Per-pose DataFrame with at minimum a 'pdb_code' column.

    Returns:
        Copy of df with two new columns:
            exp_affinity_pKd       (-log10(Kd_in_M)), typically 2–12 for PDBbind
            exp_affinity_kcal_mol  (-1.364 * pKd), same sign convention as Vina
    """
    try:
        binding_df = load_binding_data()
    except FileNotFoundError as e:
        print(f"  Warning: {e} — skipping affinity labels")
        df = df.copy()
        df['exp_affinity_pKd'] = np.nan
        df['exp_affinity_kcal_mol'] = np.nan
        return df

    binding_df = binding_df[['pdb_code', 'affinity_nM']].copy()
    binding_df['exp_affinity_pKd'] = -np.log10(binding_df['affinity_nM'] * 1e-9)
    binding_df['exp_affinity_kcal_mol'] = -1.364 * binding_df['exp_affinity_pKd']

    out = df.merge(
        binding_df[['pdb_code', 'exp_affinity_pKd', 'exp_affinity_kcal_mol']],
        on='pdb_code',
        how='left'
    )
    n_complexes_with = out.loc[out['exp_affinity_pKd'].notna(), 'pdb_code'].nunique()
    n_complexes_total = df['pdb_code'].nunique()
    print(f"  Affinity labels: {n_complexes_with}/{n_complexes_total} complexes have "
          f"experimental data ({n_complexes_total - n_complexes_with} missing → NaN)")
    return out


def get_complexes() -> List[str]:
    """
    Get list of PDB codes that have both ligand and protein files.
    
    Returns:
        List of valid PDB codes
    """
    if not STRUCTURES_DIR.exists():
        raise FileNotFoundError(f"Structures directory not found: {STRUCTURES_DIR}")

    # Exclude CASF-2016 complexes — they are reserved for the fixed test set
    casf_codes = load_casf_pdb_codes() if CASF_DIR.exists() else set()

    valid_complexes = []

    for pdb_dir in STRUCTURES_DIR.rglob("*"):
        if not pdb_dir.is_dir():
            continue

        pdb_code = pdb_dir.name
        if pdb_code in casf_codes:
            continue  # skip — goes to CASF test set

        ligand_file = pdb_dir / f"{pdb_code}_ligand.mol2"
        protein_file = pdb_dir / f"{pdb_code}_protein.pdb"

        if ligand_file.exists() and protein_file.exists():
            valid_complexes.append(pdb_code)

    valid_complexes.sort()
    return valid_complexes


def get_complex_path(pdb_code: str) -> Path:
    """
    Find the full path to a complex directory given its PDB code.
    Checks CASF_DIR first (direct lookup), then searches STRUCTURES_DIR recursively
    to handle year-based subdirectory layouts (e.g. PDBind_2020/1981-2000/pdb_code/).

    Args:
        pdb_code: The PDB code to find

    Returns:
        Path to the complex directory

    Raises:
        FileNotFoundError: If the complex directory is not found
    """
    # Fast direct check in CASF_DIR (no rglob needed)
    if CASF_DIR.exists():
        casf_path = CASF_DIR / pdb_code
        if casf_path.is_dir():
            return casf_path
    # Recursive search in STRUCTURES_DIR (handles year subdirs)
    for pdb_dir in STRUCTURES_DIR.rglob(pdb_code):
        if pdb_dir.is_dir():
            return pdb_dir
    raise FileNotFoundError(f"Complex directory not found for {pdb_code}")


def load_casf_pdb_codes(casf_dir: Path = CASF_DIR) -> set:
    """Return the set of PDB codes in the CASF-2016 coreset directory."""
    if not casf_dir.exists():
        raise FileNotFoundError(f"CASF directory not found: {casf_dir}")
    return {d.name for d in casf_dir.iterdir() if d.is_dir()}


def get_casf_complexes(casf_dir: Path = CASF_DIR) -> List[str]:
    """Return sorted list of valid CASF-2016 PDB codes (ligand.mol2 + protein.pdb present)."""
    if not casf_dir.exists():
        raise FileNotFoundError(f"CASF directory not found: {casf_dir}")
    valid = []
    for pdb_dir in sorted(casf_dir.iterdir()):
        if not pdb_dir.is_dir():
            continue
        code = pdb_dir.name
        if (pdb_dir / f"{code}_ligand.mol2").exists() and \
           (pdb_dir / f"{code}_protein.pdb").exists():
            valid.append(code)
    return valid


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
    df = df.reset_index(drop=True)
    new_df = df.copy()

    boltzmann_probs = np.zeros(len(df))
    gap_to_next = np.zeros(len(df))
    top2_gaps = np.zeros(len(df))

    kT = 0.592  # RT at 300K in kcal/mol (Vina scores in kcal/mol)

    for _, group in df.groupby('pdb_code'):
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

    # Size-normalized Vina score (only when mol2 features have been added)
    if 'num_heavy_atoms' in df.columns:
        new_df['vina_score_per_atom'] = df['vina_score'] / (df['num_heavy_atoms'] + 1)

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
        try:
            ligand_mol2 = get_complex_path(pdb_code) / f"{pdb_code}_ligand.mol2"
        except FileNotFoundError:
            continue
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


def augment_affinity_with_fingerprints(
    aff_df: pd.DataFrame,
    n_bits: int = 2048,
) -> pd.DataFrame:
    """
    Add Morgan ECFP4 fingerprint columns to a per-complex affinity DataFrame.

    Adds n_bits columns named morgan_fp_0 … morgan_fp_{n_bits-1}.
    Rows where the mol2 file is missing or unparseable are left as NaN.
    Does NOT modify the main per-pose DataFrames.

    Args:
        aff_df:  Per-complex DataFrame (one row per complex) from prepare_affinity_data().
        n_bits:  Fingerprint length in bits (default 2048).

    Returns:
        Copy of aff_df with n_bits additional columns.
    """
    fp_cols = [f'morgan_fp_{i}' for i in range(n_bits)]
    new_df = aff_df.copy()
    # Pre-allocate as float (NaN-compatible)
    fp_matrix = np.full((len(new_df), n_bits), np.nan, dtype=np.float32)

    found = 0
    for i, row in enumerate(new_df.itertuples(index=False)):
        mol2 = STRUCTURES_DIR / row.pdb_code / f"{row.pdb_code}_ligand.mol2"
        fp = compute_morgan_fingerprint(mol2, n_bits)
        if fp is not None:
            fp_matrix[i] = fp
            found += 1

    fp_df = pd.DataFrame(fp_matrix, columns=fp_cols, index=new_df.index)
    new_df = pd.concat([new_df, fp_df], axis=1)
    print(f"  Morgan fingerprints: {found}/{len(new_df)} complexes")
    return new_df


def load_protein_heavy_atom_coords(protein_pdb: Path) -> Optional[np.ndarray]:
    """
    Parse a protein PDB file and return heavy-atom (non-hydrogen) XYZ coordinates.

    Args:
        protein_pdb: Path to the protein PDB file

    Returns:
        numpy array of shape (N, 3), or None if parsing fails
    """
    coords = []
    try:
        with open(protein_pdb, 'r') as f:
            for line in f:
                if not (line.startswith('ATOM') or line.startswith('HETATM')):
                    continue
                # Atom name is in columns 12-16; element in cols 76-78 (may be absent)
                atom_name = line[12:16].strip()
                # Skip hydrogens by atom name (H, 1H, 2H, HD, HE, …)
                if atom_name.startswith('H') or (len(atom_name) > 1 and atom_name[1] == 'H'):
                    continue
                # Also skip by element column if present
                if len(line) >= 78:
                    element = line[76:78].strip()
                    if element.upper() == 'H':
                        continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except (ValueError, IndexError):
                    continue
    except Exception as e:
        print(f"    Warning: Could not parse protein PDB {protein_pdb}: {e}")
        return None

    if not coords:
        return None
    return np.array(coords, dtype=np.float32)


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


_HYDROPHOBIC_ELEMENTS = {'C', 'S'}
_POLAR_ELEMENTS = {'N', 'O'}
_AROMATIC_RESNAMES = {'PHE', 'TRP', 'TYR', 'HIS', 'ARG'}


def augment_with_contact_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add protein–ligand contact features by reading already-computed PDBQT output files
    and the original protein PDB files.  Does NOT rerun Vina.

    For each complex one KDTree is built from the protein heavy atoms and reused across
    all 5 poses, keeping runtime to ~5–10 minutes for 5,300 complexes.

    Args:
        df: DataFrame with pdb_code and pose_idx columns

    Returns:
        DataFrame with additional contact feature columns (NaN where files missing)
    """
    contact_cols = ['contact_n_3A', 'contact_n_4A', 'contact_n_5A',
                    'contact_min_dist', 'contact_score_lj',
                    'contact_buried_frac', 'contact_n_per_atom',
                    'contact_n_hydrophobic', 'contact_n_hbond', 'contact_n_aromatic',
                    'contact_score_gaussian', 'contact_hbond_normalized']
    new_df = df.copy()
    for col in contact_cols:
        new_df[col] = np.nan

    found = 0
    for pdb_code in df['pdb_code'].unique():
        try:
            protein_pdb = get_complex_path(pdb_code) / f"{pdb_code}_protein.pdb"
        except FileNotFoundError:
            continue
        pdbqt_file = OUTPUT_DIR / f"{pdb_code}_vina_output.pdbqt"

        if not protein_pdb.exists() or not pdbqt_file.exists():
            continue

        protein_result = load_protein_atoms(protein_pdb)
        if protein_result is None:
            continue
        protein_coords, prot_elements, prot_resnames = protein_result
        if len(protein_coords) == 0:
            continue

        protein_tree = KDTree(protein_coords)

        # Build subset KDTrees for atom-type-aware features
        prot_elem_arr = np.array(prot_elements)
        prot_res_arr  = np.array(prot_resnames)

        hphob_mask   = np.array([e in _HYDROPHOBIC_ELEMENTS for e in prot_elements])
        polar_mask    = np.array([e in _POLAR_ELEMENTS       for e in prot_elements])
        aromatic_mask = np.array([r in _AROMATIC_RESNAMES    for r in prot_resnames])

        protein_hydrophobic_tree = KDTree(protein_coords[hphob_mask])   if hphob_mask.any()   else None
        protein_polar_tree       = KDTree(protein_coords[polar_mask])    if polar_mask.any()    else None
        protein_aromatic_tree    = KDTree(protein_coords[aromatic_mask]) if aromatic_mask.any() else None

        pose_atoms_list = extract_pose_atoms(pdbqt_file)
        if not pose_atoms_list:
            continue

        found += 1
        complex_mask = new_df['pdb_code'] == pdb_code
        for row_idx in new_df[complex_mask].index:
            pose_idx = int(new_df.loc[row_idx, 'pose_idx'])
            if pose_idx < len(pose_atoms_list):
                pose_coords, pose_types = pose_atoms_list[pose_idx]
                if len(pose_coords) > 0:
                    feats = compute_pose_contact_features(
                        pose_coords,
                        protein_tree,
                        ligand_atom_types=pose_types,
                        protein_hydrophobic_tree=protein_hydrophobic_tree,
                        protein_polar_tree=protein_polar_tree,
                        protein_aromatic_tree=protein_aromatic_tree,
                    )
                    for col, val in feats.items():
                        new_df.loc[row_idx, col] = val

    print(f"  Added contact features for {found}/{df['pdb_code'].nunique()} complexes")
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


def train_ranker_model(
    train_df: pd.DataFrame,
    hyperparams: Optional[Dict] = None,
) -> Tuple[xgb.XGBRanker, List[str]]:
    """
    Train XGBoost ranker to order poses within each complex by predicted quality.
    Uses continuous RMSD-based relevance (1/(1+rmsd)) so the model learns to
    rank poses by proximity to the crystal structure rather than binary best/not-best.

    Note: XGBoost 3.x ranking eval metrics require integer labels, so early stopping
    via eval_set is not compatible with our continuous 1/(1+rmsd) target. The GB
    ranker uses n_iter_no_change for early stopping instead.

    Args:
        train_df: DataFrame with pose-level features and rmsd column
        hyperparams: Optional dict of hyperparameters to override defaults

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

    params = {
        'objective': 'rank:pairwise',
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        **(hyperparams or {}),
    }
    ranker = xgb.XGBRanker(
        **params,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    ranker.fit(X, y_rel, qid=qid)

    return ranker, feature_cols


def optimize_ranker_hyperparams(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    n_trials: int = 30,
    storage: Optional[str] = None,
) -> Dict:
    """
    Use Optuna to find optimal RF ranker hyperparameters via 5-fold GroupKFold CV.

    CV is run on the combined train+val data so Optuna sees more complexes. The final
    model is still trained on train_df only. Poses from the same complex are always in
    the same fold (GroupKFold) to avoid data leakage. Objective: maximise mean neg-MSE
    on the 1/(1+rmsd) target.

    Args:
        train_df:     Per-pose training DataFrame (augmented).
        val_df:       Per-pose validation DataFrame (augmented).
        feature_cols: Feature column names.
        n_trials:     Number of NEW Optuna trials to run.
        storage:      SQLite URL for persisting/resuming trials, e.g.
                      "sqlite:///output/optuna_studies.db". None = in-memory only.

    Returns:
        Dict of best hyperparameters found across all trials.
    """
    cv_df = pd.concat([train_df, val_df], ignore_index=True)
    available_cols = [c for c in feature_cols if c in cv_df.columns]
    X = np.nan_to_num(cv_df[available_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = 1.0 / (1.0 + cv_df['rmsd'].values)
    _, groups = np.unique(cv_df['pdb_code'].values, return_inverse=True)
    gkf = GroupKFold(n_splits=5)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_categorical('n_estimators', [200, 300, 500, 750]),
            'max_depth':        trial.suggest_categorical('max_depth', [5, 8, 10, 15, 20]),
            'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 3, 5, 10]),
            'max_features':     trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7]),
        }
        scores = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            rf = RandomForestRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)
            rf.fit(X[train_idx], y[train_idx])
            scores.append(-mean_squared_error(y[val_idx], rf.predict(X[val_idx])))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        storage=storage,
        study_name='rf_ranker',
        load_if_exists=True,
    )
    n_existing = len(study.trials)
    if n_existing:
        print(f"  Loaded {n_existing} existing RF ranker trials from storage")
    study.optimize(objective, n_trials=n_trials)
    print(f"  RF ranker best CV score: {study.best_value:.6f} "
          f"(over {len(study.trials)} total trials)")
    return study.best_params


def optimize_gb_ranker_hyperparams(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    n_trials: int = 30,
    storage: Optional[str] = None,
) -> Dict:
    """
    Use Optuna to find optimal GB ranker hyperparameters via 5-fold GroupKFold CV.

    CV is run on the combined train+val data. Objective: maximise mean neg-MSE
    on the 1/(1+rmsd) target. study_name='gb_ranker'.
    """
    cv_df = pd.concat([train_df, val_df], ignore_index=True)
    available_cols = [c for c in feature_cols if c in cv_df.columns]
    X = np.nan_to_num(cv_df[available_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = 1.0 / (1.0 + cv_df['rmsd'].values)
    _, groups = np.unique(cv_df['pdb_code'].values, return_inverse=True)
    gkf = GroupKFold(n_splits=5)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_categorical('n_estimators', [100, 200, 300, 500]),
            'max_depth':        trial.suggest_categorical('max_depth', [3, 4, 5, 6, 7]),
            'learning_rate':    trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2]),
            'subsample':        trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 0.9, 1.0]),
            'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 5, 10]),
        }
        scores = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            gb = GradientBoostingRegressor(**params, random_state=RANDOM_SEED)
            gb.fit(X[train_idx], y[train_idx])
            scores.append(-mean_squared_error(y[val_idx], gb.predict(X[val_idx])))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        storage=storage,
        study_name='gb_ranker',
        load_if_exists=True,
    )
    n_existing = len(study.trials)
    if n_existing:
        print(f"  Loaded {n_existing} existing GB ranker trials from storage")
    study.optimize(objective, n_trials=n_trials)
    print(f"  GB ranker best CV score: {study.best_value:.6f} "
          f"(over {len(study.trials)} total trials)")
    return study.best_params


def optimize_xgb_ranker_hyperparams(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    n_trials: int = 30,
    storage: Optional[str] = None,
) -> Dict:
    """
    Use Optuna to find optimal XGBRanker hyperparameters via 5-fold GroupKFold CV.

    CV is run on the combined train+val data. XGBRanker (rank:pairwise) is used for
    training; scoring uses neg-MSE on 1/(1+rmsd) (same proxy as RF/GB).
    study_name='xgb_ranker'.
    """
    cv_df = pd.concat([train_df, val_df], ignore_index=True)
    available_cols = [c for c in feature_cols if c in cv_df.columns]
    # Sort by pdb_code so qid groups are contiguous
    cv_df = cv_df.sort_values('pdb_code').reset_index(drop=True)
    X = np.nan_to_num(cv_df[available_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = 1.0 / (1.0 + cv_df['rmsd'].values)
    _, groups = np.unique(cv_df['pdb_code'].values, return_inverse=True)
    gkf = GroupKFold(n_splits=5)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_categorical('n_estimators', [100, 200, 300, 500]),
            'max_depth':        trial.suggest_categorical('max_depth', [3, 4, 5, 6, 7]),
            'learning_rate':    trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2]),
            'subsample':        trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 0.9]),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6, 0.7, 0.8, 0.9]),
            'min_child_weight': trial.suggest_categorical('min_child_weight', [1, 3, 5, 7]),
        }
        scores = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            # Build contiguous qid for training fold
            train_groups = groups[train_idx]
            _, qid_train = np.unique(train_groups, return_inverse=True)
            # Sort training fold by pdb_code for XGBRanker
            sort_order = np.argsort(train_groups, kind='stable')
            X_tr = X[train_idx][sort_order]
            y_tr = y[train_idx][sort_order]
            qid_tr = qid_train[sort_order]

            ranker = xgb.XGBRanker(
                objective='rank:pairwise',
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbosity=0,
                **params,
            )
            ranker.fit(X_tr, y_tr, qid=qid_tr)
            pred = ranker.predict(X[val_idx])
            scores.append(-mean_squared_error(y[val_idx], pred))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        storage=storage,
        study_name='xgb_ranker',
        load_if_exists=True,
    )
    n_existing = len(study.trials)
    if n_existing:
        print(f"  Loaded {n_existing} existing XGB ranker trials from storage")
    study.optimize(objective, n_trials=n_trials)
    print(f"  XGB ranker best CV score: {study.best_value:.6f} "
          f"(over {len(study.trials)} total trials)")
    return study.best_params


def optimize_rf_affinity_hyperparams(
    train_affinity_df: pd.DataFrame,
    val_affinity_df: pd.DataFrame,
    feature_cols: List[str],
    n_trials: int = 30,
    storage: Optional[str] = None,
    study_name: str = 'rf_affinity',
) -> Dict:
    """Optuna search for RF affinity hyperparameters (5-fold KFold on train+val)."""
    cv_df = pd.concat([train_affinity_df, val_affinity_df], ignore_index=True)
    available_cols = [c for c in feature_cols if c in cv_df.columns]
    X = np.nan_to_num(cv_df[available_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = cv_df['exp_affinity_kcal_mol'].values
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    def objective(trial):
        params = {
            'n_estimators':    trial.suggest_categorical('n_estimators',    [100, 200, 300, 500]),
            'max_depth':       trial.suggest_categorical('max_depth',       [5, 8, 10, 15, None]),
            'max_features':    trial.suggest_categorical('max_features',    ['sqrt', 'log2', 0.3, 0.5]),
            'min_samples_leaf':trial.suggest_categorical('min_samples_leaf',[1, 2, 5, 10]),
        }
        scores = []
        for train_idx, val_idx in kf.split(X):
            rf = RandomForestRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)
            rf.fit(X[train_idx], y[train_idx])
            scores.append(-mean_squared_error(y[val_idx], rf.predict(X[val_idx])))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize', storage=storage,
        study_name=study_name, load_if_exists=True,
    )
    n_existing = len(study.trials)
    if n_existing:
        print(f"  Loaded {n_existing} existing RF affinity trials from storage")
    study.optimize(objective, n_trials=n_trials)
    print(f"  RF affinity best CV score: {study.best_value:.6f} "
          f"(over {len(study.trials)} total trials)")
    return study.best_params


def optimize_affinity_hyperparams(
    train_affinity_df: pd.DataFrame,
    val_affinity_df: pd.DataFrame,
    feature_cols: List[str],
    n_trials: int = 30,
    storage: Optional[str] = None,
    study_name: str = 'gb_affinity',
) -> Dict:
    """
    Use Optuna to find optimal GB affinity hyperparameters via 5-fold KFold CV.

    CV is run on the combined train+val data so Optuna sees more complexes. The final
    model is still trained on train_affinity_df only. One row per complex; regular KFold
    is appropriate (no grouping needed). Objective: maximise mean neg-MSE on ΔG target.

    Args:
        train_affinity_df: Per-complex training DataFrame from prepare_affinity_data().
        val_affinity_df:   Per-complex validation DataFrame from prepare_affinity_data().
        feature_cols:      Feature column names.
        n_trials:          Number of NEW Optuna trials to run.
        storage:           SQLite URL for persisting/resuming trials, e.g.
                           "sqlite:///output/optuna_studies.db". None = in-memory only.

    Returns:
        Dict of best hyperparameters found across all trials.
    """
    cv_df = pd.concat([train_affinity_df, val_affinity_df], ignore_index=True)
    available_cols = [c for c in feature_cols if c in cv_df.columns]
    X = np.nan_to_num(cv_df[available_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = cv_df['exp_affinity_kcal_mol'].values
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_categorical('n_estimators', [100, 200, 300, 500]),
            'max_depth':        trial.suggest_categorical('max_depth', [3, 4, 5, 6, 7]),
            'learning_rate':    trial.suggest_categorical('learning_rate', [0.01, 0.05, 0.1, 0.2]),
            'subsample':        trial.suggest_categorical('subsample', [0.6, 0.7, 0.8, 0.9, 1.0]),
            'min_samples_leaf': trial.suggest_categorical('min_samples_leaf', [1, 2, 5, 10]),
        }
        scores = []
        for train_idx, val_idx in kf.split(X):
            gb = GradientBoostingRegressor(**params, random_state=RANDOM_SEED)
            gb.fit(X[train_idx], y[train_idx])
            scores.append(-mean_squared_error(y[val_idx], gb.predict(X[val_idx])))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        storage=storage,
        study_name=study_name,
        load_if_exists=True,
    )
    n_existing = len(study.trials)
    if n_existing:
        print(f"  Loaded {n_existing} existing {study_name} trials from storage")
    study.optimize(objective, n_trials=n_trials)
    print(f"  {study_name} best CV score: {study.best_value:.6f} "
          f"(over {len(study.trials)} total trials)")
    return study.best_params


def optimize_svr_affinity_hyperparams(
    train_affinity_df: pd.DataFrame,
    val_affinity_df: pd.DataFrame,
    feature_cols: List[str],
    n_trials: int = 30,
    storage: Optional[str] = None,
    study_name: str = 'svr_affinity',
) -> Dict:
    """Optuna search for SVR affinity hyperparameters (5-fold KFold on train+val)."""
    cv_df = pd.concat([train_affinity_df, val_affinity_df], ignore_index=True)
    available_cols = [c for c in feature_cols if c in cv_df.columns]
    X = np.nan_to_num(cv_df[available_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = cv_df['exp_affinity_kcal_mol'].values
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    def objective(trial):
        C       = trial.suggest_categorical('C',       [0.1, 1.0, 10.0, 100.0])
        epsilon = trial.suggest_categorical('epsilon', [0.01, 0.05, 0.1, 0.5])
        gamma   = trial.suggest_categorical('gamma',   ['scale', 'auto', 0.01, 0.1])
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma)),
        ])
        scores = []
        for train_idx, val_idx in kf.split(X):
            pipe.fit(X[train_idx], y[train_idx])
            scores.append(-mean_squared_error(y[val_idx], pipe.predict(X[val_idx])))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize', storage=storage,
        study_name=study_name, load_if_exists=True,
    )
    n_existing = len(study.trials)
    if n_existing:
        print(f"  Loaded {n_existing} existing SVR affinity trials from storage")
    study.optimize(objective, n_trials=n_trials)
    print(f"  SVR affinity best CV score: {study.best_value:.6f} "
          f"(over {len(study.trials)} total trials)")
    return study.best_params


def optimize_xgb_affinity_hyperparams(
    train_affinity_df: pd.DataFrame,
    val_affinity_df: pd.DataFrame,
    feature_cols: List[str],
    n_trials: int = 30,
    storage: Optional[str] = None,
    study_name: str = 'xgb_affinity',
) -> Dict:
    """Optuna search for XGBRegressor affinity hyperparameters (5-fold KFold on train+val)."""
    cv_df = pd.concat([train_affinity_df, val_affinity_df], ignore_index=True)
    available_cols = [c for c in feature_cols if c in cv_df.columns]
    X = np.nan_to_num(cv_df[available_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = cv_df['exp_affinity_kcal_mol'].values
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    def objective(trial):
        params = {
            'n_estimators':     trial.suggest_categorical('n_estimators',    [100, 200, 300, 500]),
            'max_depth':        trial.suggest_categorical('max_depth',        [3, 4, 5, 6, 7]),
            'learning_rate':    trial.suggest_categorical('learning_rate',    [0.01, 0.05, 0.1, 0.2]),
            'subsample':        trial.suggest_categorical('subsample',        [0.6, 0.7, 0.8, 0.9]),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.6, 0.7, 0.8, 0.9]),
        }
        scores = []
        for train_idx, val_idx in kf.split(X):
            reg = xgb.XGBRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1,
                                   verbosity=0)
            reg.fit(X[train_idx], y[train_idx])
            scores.append(-mean_squared_error(y[val_idx], reg.predict(X[val_idx])))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize', storage=storage,
        study_name=study_name, load_if_exists=True,
    )
    n_existing = len(study.trials)
    if n_existing:
        print(f"  Loaded {n_existing} existing XGB affinity trials from storage")
    study.optimize(objective, n_trials=n_trials)
    print(f"  XGB affinity best CV score: {study.best_value:.6f} "
          f"(over {len(study.trials)} total trials)")
    return study.best_params


def optimize_ridge_affinity_hyperparams(
    train_affinity_df: pd.DataFrame,
    val_affinity_df: pd.DataFrame,
    feature_cols: List[str],
    n_trials: int = 30,
    storage: Optional[str] = None,
    study_name: str = 'ridge_affinity',
) -> Dict:
    """Optuna search for Ridge affinity alpha (5-fold KFold on train+val)."""
    cv_df = pd.concat([train_affinity_df, val_affinity_df], ignore_index=True)
    available_cols = [c for c in feature_cols if c in cv_df.columns]
    X = np.nan_to_num(cv_df[available_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = cv_df['exp_affinity_kcal_mol'].values
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    def objective(trial):
        alpha = trial.suggest_categorical('alpha', [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])
        pipe = Pipeline([('scaler', StandardScaler()), ('ridge', Ridge(alpha=alpha))])
        scores = []
        for train_idx, val_idx in kf.split(X):
            pipe.fit(X[train_idx], y[train_idx])
            scores.append(-mean_squared_error(y[val_idx], pipe.predict(X[val_idx])))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize', storage=storage,
        study_name=study_name, load_if_exists=True,
    )
    n_existing = len(study.trials)
    if n_existing:
        print(f"  Loaded {n_existing} existing Ridge affinity trials from storage")
    study.optimize(objective, n_trials=n_trials)
    print(f"  Ridge affinity best CV score: {study.best_value:.6f} "
          f"(over {len(study.trials)} total trials)")
    return study.best_params


_MLP_LAYER_SIZES = [(64,), (128,), (256,), (128, 64), (256, 128), (256, 128, 64)]


def optimize_mlp_affinity_hyperparams(
    train_affinity_df: pd.DataFrame,
    val_affinity_df: pd.DataFrame,
    feature_cols: List[str],
    n_trials: int = 30,
    storage: Optional[str] = None,
    study_name: str = 'mlp_affinity',
) -> Dict:
    """Optuna search for MLP affinity hyperparameters (5-fold KFold on train+val).

    hidden_layer_sizes is stored as an integer index into _MLP_LAYER_SIZES
    (tuples are not JSON-serializable).
    """
    cv_df = pd.concat([train_affinity_df, val_affinity_df], ignore_index=True)
    available_cols = [c for c in feature_cols if c in cv_df.columns]
    X = np.nan_to_num(cv_df[available_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = cv_df['exp_affinity_kcal_mol'].values
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    def objective(trial):
        layer_idx = trial.suggest_categorical('hidden_layer_sizes', list(range(len(_MLP_LAYER_SIZES))))
        alpha = trial.suggest_categorical('alpha', [1e-4, 1e-3, 0.01, 0.1])
        lr_init = trial.suggest_categorical('learning_rate_init', [1e-4, 1e-3, 0.01])
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=_MLP_LAYER_SIZES[layer_idx],
                alpha=alpha,
                learning_rate_init=lr_init,
                max_iter=500, early_stopping=True, random_state=RANDOM_SEED,
            )),
        ])
        scores = []
        for train_idx, val_idx in kf.split(X):
            pipe.fit(X[train_idx], y[train_idx])
            scores.append(-mean_squared_error(y[val_idx], pipe.predict(X[val_idx])))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize', storage=storage,
        study_name=study_name, load_if_exists=True,
    )
    n_existing = len(study.trials)
    if n_existing:
        print(f"  Loaded {n_existing} existing MLP affinity trials from storage")
    study.optimize(objective, n_trials=n_trials)
    print(f"  MLP affinity best CV score: {study.best_value:.6f} "
          f"(over {len(study.trials)} total trials)")
    return study.best_params


def train_rf_ranker(
    train_df: pd.DataFrame,
    hyperparams: Optional[Dict] = None
) -> Tuple[RandomForestRegressor, List[str]]:
    """
    Train a Random Forest regressor as a pose ranker.
    Predicts 1/(1+rmsd) per pose; the pose with the highest predicted score is selected.

    Args:
        train_df:    DataFrame with pose-level features and rmsd column.
        hyperparams: Optional dict of hyperparameters to override the defaults
                     (e.g. from optimize_ranker_hyperparams()).

    Returns:
        Tuple of (trained RandomForestRegressor, list of feature column names)
    """
    feature_cols = [col for col in train_df.columns
                    if col not in ['pdb_code', 'pose_idx', 'is_best_pose', 'rmsd']]

    X = train_df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_rel = 1.0 / (1.0 + train_df['rmsd'].values)

    params = {'n_estimators': 300, 'max_depth': 10, 'min_samples_leaf': 5,
              **(hyperparams or {})}
    model = RandomForestRegressor(**params, random_state=RANDOM_SEED, n_jobs=-1)
    model.fit(X, y_rel)
    return model, feature_cols


def train_gb_ranker(
    train_df: pd.DataFrame,
    hyperparams: Optional[Dict] = None
) -> Tuple[GradientBoostingRegressor, List[str]]:
    """
    Train a Gradient Boosting regressor as a pose ranker.
    Predicts 1/(1+rmsd) per pose; the pose with the highest predicted score is selected.

    Args:
        train_df:    DataFrame with pose-level features and rmsd column.
        hyperparams: Optional dict of hyperparameters to override the defaults.

    Returns:
        Tuple of (trained GradientBoostingRegressor, list of feature column names)
    """
    feature_cols = [col for col in train_df.columns
                    if col not in ['pdb_code', 'pose_idx', 'is_best_pose', 'rmsd']]

    X = train_df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_rel = 1.0 / (1.0 + train_df['rmsd'].values)

    params = {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05,
              'subsample': 0.8, 'min_samples_leaf': 5,
              'n_iter_no_change': 10, 'validation_fraction': 0.1, 'tol': 1e-4,
              **(hyperparams or {})}
    model = GradientBoostingRegressor(**params, random_state=RANDOM_SEED)
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
    spearman_scores = []  # per-complex Spearman ρ(model_score, rmsd) — should be negative

    # Multi-threshold accumulators
    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    vina_at_thresh = {t: 0 for t in thresholds}
    xgb_at_thresh  = {t: 0 for t in thresholds}

    for pdb_code in test_df_copy['pdb_code'].unique():
        complex_df = test_df_copy[test_df_copy['pdb_code'] == pdb_code]
        if len(complex_df) == 0:
            continue

        total_complexes += 1
        actual_best_idx = complex_df['rmsd'].idxmin()
        vina_best_idx = complex_df['vina_rank'].idxmin()
        xgb_best_idx = complex_df['predicted_proba'].idxmax()

        vina_rmsd = complex_df.loc[vina_best_idx, 'rmsd']
        xgb_rmsd  = complex_df.loc[xgb_best_idx,  'rmsd']

        vina_selected_rmsds.append(vina_rmsd)
        xgb_selected_rmsds.append(xgb_rmsd)

        if vina_best_idx == actual_best_idx:
            vina_success += 1
        if xgb_best_idx == actual_best_idx:
            xgb_success += 1

        for t in thresholds:
            if vina_rmsd < t:
                vina_at_thresh[t] += 1
            if xgb_rmsd < t:
                xgb_at_thresh[t] += 1

        # Per-complex Spearman correlation (model score vs RMSD)
        if len(complex_df) >= 2:
            rho, _ = spearmanr(complex_df['predicted_proba'].values, complex_df['rmsd'].values)
            if not np.isnan(rho):
                spearman_scores.append(rho)

    n = total_complexes if total_complexes > 0 else 1
    result = {
        'total_complexes': total_complexes,
        'vina_success_rate': vina_success / n,
        'xgb_success_rate':  xgb_success  / n,
        'vina_success_count': vina_success,
        'xgb_success_count':  xgb_success,
        'mean_vina_rmsd': np.mean(vina_selected_rmsds) if vina_selected_rmsds else 0,
        'mean_xgb_rmsd':  np.mean(xgb_selected_rmsds)  if xgb_selected_rmsds  else 0,
        'median_vina_rmsd': float(np.median(vina_selected_rmsds)) if vina_selected_rmsds else 0,
        'median_xgb_rmsd':  float(np.median(xgb_selected_rmsds))  if xgb_selected_rmsds  else 0,
        'mean_rmsd_improvement': (np.mean(vina_selected_rmsds) - np.mean(xgb_selected_rmsds))
                                  if vina_selected_rmsds else 0,
        'spearman_mean': float(np.mean(spearman_scores)) if spearman_scores else 0,
        'vina_selected_rmsds': vina_selected_rmsds,
        'xgb_selected_rmsds':  xgb_selected_rmsds,
        'test_df': test_df_copy,
    }
    # Add threshold-specific success rates
    for t in thresholds:
        key = str(t).replace('.', 'p')   # e.g. "2.0" -> "2p0"
        result[f'vina_success_{key}A_rate']  = vina_at_thresh[t] / n
        result[f'xgb_success_{key}A_rate']   = xgb_at_thresh[t]  / n
        result[f'vina_success_{key}A_count'] = vina_at_thresh[t]
        result[f'xgb_success_{key}A_count']  = xgb_at_thresh[t]

    # Keep the 2Å keys that the rest of the code already references
    result['vina_success_2A_rate']  = result['vina_success_2p0A_rate']
    result['xgb_success_2A_rate']   = result['xgb_success_2p0A_rate']
    result['vina_success_2A_count'] = result['vina_success_2p0A_count']
    result['xgb_success_2A_count']  = result['xgb_success_2p0A_count']
    return result


def evaluate_ensemble(model_results: Dict, test_df: pd.DataFrame) -> Dict:
    """
    Build an ensemble by averaging per-complex min-max normalised scores from all rankers.
    The ensemble does not require additional training; it combines the three already-evaluated
    models to reduce variance.

    Args:
        model_results: Dict mapping model label -> evaluate_ranker() result dict
        test_df: Original test DataFrame (without predictions)

    Returns:
        Metrics dict with the same keys as evaluate_ranker()
    """
    # Collect predicted_proba from each model's test_df (same row order)
    norm_scores_list = []
    for metrics in model_results.values():
        scores = metrics['test_df']['predicted_proba'].values.astype(float)
        lo, hi = scores.min(), scores.max()
        norm_scores_list.append((scores - lo) / (hi - lo + 1e-10))

    ensemble_scores = np.mean(norm_scores_list, axis=0)

    # Build a combined test_df with the ensemble scores
    combined_df = test_df.copy().reset_index(drop=True)
    # Align with any model's test_df (they all share the same pdb_code/pose_idx order)
    ref_df = next(iter(model_results.values()))['test_df'].reset_index(drop=True)
    combined_df['predicted_proba'] = ensemble_scores

    # Evaluate using the same logic as evaluate_ranker
    vina_success = 0
    xgb_success = 0
    total_complexes = 0
    vina_selected_rmsds = []
    xgb_selected_rmsds = []
    spearman_scores = []

    thresholds = [1.0, 1.5, 2.0, 2.5, 3.0]
    vina_at_thresh = {t: 0 for t in thresholds}
    xgb_at_thresh  = {t: 0 for t in thresholds}

    for pdb_code in ref_df['pdb_code'].unique():
        mask = ref_df['pdb_code'] == pdb_code
        complex_ref = ref_df[mask]
        complex_scores = ensemble_scores[mask.values]

        if len(complex_ref) == 0:
            continue
        total_complexes += 1

        actual_best_idx = complex_ref['rmsd'].idxmin()
        vina_best_idx = complex_ref['vina_rank'].idxmin()
        xgb_best_local = np.argmax(complex_scores)
        xgb_best_idx = complex_ref.index[xgb_best_local]

        vina_rmsd = complex_ref.loc[vina_best_idx, 'rmsd']
        xgb_rmsd = complex_ref.loc[xgb_best_idx, 'rmsd']

        vina_selected_rmsds.append(vina_rmsd)
        xgb_selected_rmsds.append(xgb_rmsd)

        if vina_best_idx == actual_best_idx:
            vina_success += 1
        if xgb_best_idx == actual_best_idx:
            xgb_success += 1

        for t in thresholds:
            if vina_rmsd < t:
                vina_at_thresh[t] += 1
            if xgb_rmsd < t:
                xgb_at_thresh[t] += 1

        if len(complex_ref) >= 2:
            rho, _ = spearmanr(complex_scores, complex_ref['rmsd'].values)
            if not np.isnan(rho):
                spearman_scores.append(rho)

    # Store ensemble scores back in combined_df for export / plots
    combined_df = ref_df.copy()
    combined_df['predicted_proba'] = ensemble_scores

    n = total_complexes if total_complexes > 0 else 1
    result = {
        'total_complexes': total_complexes,
        'vina_success_rate': vina_success / n,
        'xgb_success_rate': xgb_success / n,
        'vina_success_count': vina_success,
        'xgb_success_count': xgb_success,
        'mean_vina_rmsd': np.mean(vina_selected_rmsds) if vina_selected_rmsds else 0,
        'mean_xgb_rmsd': np.mean(xgb_selected_rmsds) if xgb_selected_rmsds else 0,
        'median_vina_rmsd': float(np.median(vina_selected_rmsds)) if vina_selected_rmsds else 0,
        'median_xgb_rmsd': float(np.median(xgb_selected_rmsds)) if xgb_selected_rmsds else 0,
        'mean_rmsd_improvement': (np.mean(vina_selected_rmsds) - np.mean(xgb_selected_rmsds))
                                  if vina_selected_rmsds else 0,
        'spearman_mean': float(np.mean(spearman_scores)) if spearman_scores else 0,
        'vina_selected_rmsds': vina_selected_rmsds,
        'xgb_selected_rmsds': xgb_selected_rmsds,
        'test_df': combined_df,
    }
    for t in thresholds:
        key = str(t).replace('.', 'p')
        result[f'vina_success_{key}A_rate']  = vina_at_thresh[t] / n
        result[f'xgb_success_{key}A_rate']   = xgb_at_thresh[t]  / n
        result[f'vina_success_{key}A_count'] = vina_at_thresh[t]
        result[f'xgb_success_{key}A_count']  = xgb_at_thresh[t]
    result['vina_success_2A_rate']  = result['vina_success_2p0A_rate']
    result['xgb_success_2A_rate']   = result['xgb_success_2p0A_rate']
    result['vina_success_2A_count'] = result['vina_success_2p0A_count']
    result['xgb_success_2A_count']  = result['xgb_success_2p0A_count']
    return result


def prepare_affinity_data(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Build the per-complex affinity regression dataset from the rank-1 Vina pose.

    Args:
        df:           Per-pose DataFrame with affinity columns from join_affinity_labels().
        feature_cols: Feature column names (same set used by the pose rankers).

    Returns:
        Per-complex DataFrame (one row per complex) with feature_cols plus
        pdb_code, exp_affinity_pKd, exp_affinity_kcal_mol, vina_score, rmsd, is_best_pose.
    """
    rank1_df = df[df['vina_rank'] == 1].copy()
    rank1_df = rank1_df.drop_duplicates(subset='pdb_code', keep='first')

    before = rank1_df['pdb_code'].nunique()
    rank1_df = rank1_df.dropna(subset=['exp_affinity_kcal_mol'])
    after = rank1_df['pdb_code'].nunique()
    print(f"  Affinity dataset: {after} complexes with labels "
          f"(dropped {before - after} without experimental affinity)")

    meta_cols = ['pdb_code', 'exp_affinity_pKd', 'exp_affinity_kcal_mol',
                 'vina_score', 'rmsd', 'is_best_pose']
    feat_present = [c for c in feature_cols if c in rank1_df.columns]
    keep_cols = meta_cols + [c for c in feat_present if c not in meta_cols]
    return rank1_df[keep_cols].reset_index(drop=True)


def train_affinity_model(
    train_affinity_df: pd.DataFrame,
    feature_cols: List[str],
    hyperparams_rf: Optional[Dict] = None,
    hyperparams_gb: Optional[Dict] = None,
    hyperparams_svr: Optional[Dict] = None,
    hyperparams_xgb: Optional[Dict] = None,
    hyperparams_ridge: Optional[Dict] = None,
    hyperparams_mlp: Optional[Dict] = None,
    val_affinity_df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Train RF, GB, SVR, XGBoost, Ridge, and MLP regressors to predict binding affinity (ΔG, kcal/mol).

    Args:
        train_affinity_df: Per-complex training DataFrame from prepare_affinity_data().
        feature_cols:      Feature column names.
        hyperparams_rf:    Optional RF hyperparameters (from optimize_rf_affinity_hyperparams).
        hyperparams_gb:    Optional GB hyperparameters (from optimize_affinity_hyperparams).
        hyperparams_svr:   Optional SVR hyperparameters (C, epsilon, gamma).
        hyperparams_xgb:   Optional XGBRegressor hyperparameters.
        hyperparams_ridge: Optional Ridge hyperparameters (alpha).
        hyperparams_mlp:   Optional MLP hyperparameters (hidden_layer_sizes index, alpha, lr).
        val_affinity_df:   Optional validation DataFrame for XGBoost early stopping.

    Returns:
        Dict mapping model name → (fitted model, feature_cols used).
    """
    available_cols = [c for c in feature_cols if c in train_affinity_df.columns]
    X = train_affinity_df[available_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = train_affinity_df['exp_affinity_kcal_mol'].values

    rf_params = {'n_estimators': 300, 'max_depth': 10, 'min_samples_leaf': 5,
                 **(hyperparams_rf or {})}
    rf = RandomForestRegressor(**rf_params, random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X, y)

    gb_params = {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05,
                 'subsample': 0.8, 'min_samples_leaf': 5,
                 'n_iter_no_change': 10, 'validation_fraction': 0.1, 'tol': 1e-4,
                 **(hyperparams_gb or {})}
    gb = GradientBoostingRegressor(**gb_params, random_state=RANDOM_SEED)
    gb.fit(X, y)

    svr_params = {'C': 1.0, 'epsilon': 0.1, 'gamma': 'scale', **(hyperparams_svr or {})}
    svr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', **svr_params)),
    ])
    svr_pipe.fit(X, y)

    xgb_params = {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05,
                  'subsample': 0.8, 'colsample_bytree': 0.8, **(hyperparams_xgb or {})}
    # early_stopping_rounds goes in the constructor in XGBoost 3.x
    early_stop = 30 if val_affinity_df is not None else None
    xgb_reg = xgb.XGBRegressor(**xgb_params, random_state=RANDOM_SEED, n_jobs=-1,
                                verbosity=0, early_stopping_rounds=early_stop)
    if val_affinity_df is not None:
        X_val = np.nan_to_num(val_affinity_df[available_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
        y_val = val_affinity_df['exp_affinity_kcal_mol'].values
        xgb_reg.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    else:
        xgb_reg.fit(X, y)

    ridge_alpha = (hyperparams_ridge or {}).get('alpha', 1.0)
    ridge_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=ridge_alpha)),
    ])
    ridge_pipe.fit(X, y)

    mlp_p = hyperparams_mlp or {}
    layer_idx = mlp_p.get('hidden_layer_sizes', 4)   # default index 4 → (256, 128)
    mlp_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(
            hidden_layer_sizes=_MLP_LAYER_SIZES[layer_idx],
            alpha=mlp_p.get('alpha', 1e-3),
            learning_rate_init=mlp_p.get('learning_rate_init', 1e-3),
            max_iter=500, early_stopping=True, random_state=RANDOM_SEED,
        )),
    ])
    mlp_pipe.fit(X, y)

    print(f"  Trained affinity models on {len(train_affinity_df)} complexes, "
          f"{len(available_cols)} features")
    return {
        'rf_affinity':    (rf,         available_cols),
        'gb_affinity':    (gb,         available_cols),
        'svr_affinity':   (svr_pipe,   available_cols),
        'xgb_affinity':  (xgb_reg,    available_cols),
        'ridge_affinity': (ridge_pipe, available_cols),
        'mlp_affinity':   (mlp_pipe,   available_cols),
    }


def evaluate_affinity_model(
    models: Dict,
    test_affinity_df: pd.DataFrame,
    feature_cols: List[str]
) -> Dict:
    """
    Evaluate binding affinity models and Vina baseline on the test set.

    Returns a dict mapping model name → metrics dict with keys:
        pearson_r, spearman_r, rmse, mae, y_true, y_pred, rmsd_rank1
    Includes a 'vina_baseline' entry using raw Vina rank-1 score.
    """
    y_true = test_affinity_df['exp_affinity_kcal_mol'].values
    rmsd_rank1 = test_affinity_df['rmsd'].values

    results = {}

    y_vina = test_affinity_df['vina_score'].values
    r_p, _ = pearsonr(y_vina, y_true)
    r_s, _ = spearmanr(y_vina, y_true)
    results['vina_baseline'] = {
        'pearson_r':  float(r_p),
        'spearman_r': float(r_s),
        'rmse':       float(np.sqrt(mean_squared_error(y_true, y_vina))),
        'mae':        float(np.mean(np.abs(y_true - y_vina))),
        'y_true':     y_true,
        'y_pred':     y_vina,
        'rmsd_rank1': rmsd_rank1,
    }

    for name, (model, cols_used) in models.items():
        available = [c for c in cols_used if c in test_affinity_df.columns]
        X = test_affinity_df[available].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = model.predict(X)
        r_p, _ = pearsonr(y_pred, y_true)
        r_s, _ = spearmanr(y_pred, y_true)
        results[name] = {
            'pearson_r':  float(r_p),
            'spearman_r': float(r_s),
            'rmse':       float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae':        float(np.mean(np.abs(y_true - y_pred))),
            'y_true':     y_true,
            'y_pred':     y_pred,
            'rmsd_rank1': rmsd_rank1,
        }

    return results


def print_affinity_comparison_table(
    metrics_a: Dict,
    metrics_b: Dict,
    label_a: str = 'Current Features',
    label_b: str = 'Fingerprints Only',
    split: str = 'test',
) -> None:
    """
    Print a formatted side-by-side comparison of affinity metrics for two feature sets.

    Args:
        metrics_a: Output of evaluate_affinity_model() for feature set A.
        metrics_b: Output of evaluate_affinity_model() for feature set B.
        label_a:   Display name for feature set A.
        label_b:   Display name for feature set B.
        split:     Label for the data split (e.g. 'test' or 'val').
    """
    all_models = sorted(
        set(metrics_a.keys()) | set(metrics_b.keys()),
        key=lambda k: (k == 'vina_baseline', k),
    )

    hdr_a = f'── {label_a} ──'
    hdr_b = f'── {label_b} ──'
    col_w = max(len(hdr_a), len(hdr_b), 24)

    print(f"\nAffinity comparison ({split} set):")
    print(f"{'Model':<22}  {hdr_a:>{col_w}}  {hdr_b:>{col_w}}")
    print(f"{'':22}  {'Pearson r':>10} {'Spearman':>10} {'RMSE':>7}  "
          f"{'Pearson r':>10} {'Spearman':>10} {'RMSE':>7}")
    print('-' * (22 + 2 + col_w*2 + 10))

    for name in all_models:
        ma = metrics_a.get(name, {})
        mb = metrics_b.get(name, {})
        pr_a  = f"{ma.get('pearson_r',  float('nan')):+.3f}" if ma else '   n/a'
        sp_a  = f"{ma.get('spearman_r', float('nan')):+.3f}" if ma else '   n/a'
        rm_a  = f"{ma.get('rmse',       float('nan')):.3f}"  if ma else '  n/a'
        pr_b  = f"{mb.get('pearson_r',  float('nan')):+.3f}" if mb else '   n/a'
        sp_b  = f"{mb.get('spearman_r', float('nan')):+.3f}" if mb else '   n/a'
        rm_b  = f"{mb.get('rmse',       float('nan')):.3f}"  if mb else '  n/a'
        print(f"  {name:<20}  {pr_a:>10} {sp_a:>10} {rm_a:>7}  "
              f"{pr_b:>10} {sp_b:>10} {rm_b:>7}")


def plot_affinity_predictions(
    affinity_metrics: Dict,
    output_dir: Path,
    split: str = 'test',
) -> None:
    """
    Grid of scatter plots showing ALL affinity models vs exp ΔG.
    Order: Vina baseline first, then ML models sorted by Pearson r descending.
    Points coloured by rank-1 pose RMSD (0–10 Å, RdYlGn_r colourmap).
    All subplots share the same x/y axis limits for direct comparison.

    Args:
        affinity_metrics: Dict from evaluate_affinity_model().
        output_dir:       Directory to save the plot.
        split:            'test' or 'val' — controls filename and figure title.
    """
    ml_names = sorted(
        [k for k in affinity_metrics if k != 'vina_baseline'],
        key=lambda k: affinity_metrics[k]['pearson_r'],
        reverse=True,
    )
    ordered = ['vina_baseline'] + ml_names

    # Shared axis limits across ALL models
    all_vals = []
    for name in ordered:
        m = affinity_metrics[name]
        all_vals.extend([m['y_true'], m['y_pred']])
    shared_lims = [min(v.min() for v in all_vals) - 0.5,
                   max(v.max() for v in all_vals) + 0.5]

    n_models = len(ordered)
    n_cols = min(n_models, 3)
    n_rows = math.ceil(n_models / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()
    fig.suptitle(f'Binding Affinity Predictions ({split.capitalize()} set)', fontsize=14)

    last_sc = None
    for i, name in enumerate(ordered):
        ax = axes_flat[i]
        m = affinity_metrics[name]
        rmsd_colors = np.clip(m['rmsd_rank1'], 0, 10)

        sc = ax.scatter(m['y_true'], m['y_pred'], c=rmsd_colors, cmap='RdYlGn_r',
                        vmin=0, vmax=10, alpha=0.6, s=20, edgecolors='none')
        last_sc = sc

        ax.plot(shared_lims, shared_lims, 'k--', lw=1, alpha=0.5)
        ax.set_xlim(shared_lims)
        ax.set_ylim(shared_lims)

        label = 'Vina score' if name == 'vina_baseline' else name
        ax.set_xlabel('Experimental ΔG (kcal/mol)', fontsize=11)
        ax.set_ylabel(f'Predicted — {label} (kcal/mol)', fontsize=11)
        ax.set_title(
            f'{label}\nPearson r={m["pearson_r"]:.3f}  '
            f'Spearman r={m["spearman_r"]:.3f}  '
            f'RMSE={m["rmse"]:.2f} kcal/mol',
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n_models, len(axes_flat)):
        axes_flat[j].set_visible(False)

    # Colorbar on last used axis
    if last_sc is not None:
        fig.colorbar(last_sc, ax=axes_flat[n_models - 1], label='Rank-1 pose RMSD (Å)')

    plt.tight_layout()
    out_path = output_dir / f'affinity_predictions_{split}.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved affinity scatter plot to {out_path}")


def export_affinity_predictions_csv(
    test_affinity_df: pd.DataFrame,
    affinity_metrics: Dict,
    output_dir: Path
) -> None:
    """
    Export one row per test complex: pdb_code, experimental labels, Vina score,
    and the best ML model's predicted ΔG.
    """
    ml_names = [k for k in affinity_metrics if k != 'vina_baseline']
    best_ml_name = max(ml_names, key=lambda k: affinity_metrics[k]['pearson_r'])

    out = test_affinity_df[['pdb_code', 'exp_affinity_pKd',
                             'exp_affinity_kcal_mol', 'vina_score']].copy()
    out['predicted_affinity_kcal_mol'] = affinity_metrics[best_ml_name]['y_pred']
    out['best_model'] = best_ml_name

    path = output_dir / 'affinity_predictions.csv'
    out.to_csv(path, index=False)
    print(f"Saved affinity predictions to {path}  ({len(out)} complexes, "
          f"model={best_ml_name})")


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


def plot_success_vs_rmsd_threshold(model_results: Dict, output_dir: Path) -> None:
    """
    Plot success rate (fraction of complexes with selected pose within threshold)
    as a function of RMSD cutoff for each model and the Vina baseline.
    This is the standard figure used to compare docking rescoring methods.

    Args:
        model_results: Dict mapping model label -> evaluate_ranker() result dict
        output_dir: Directory to save the plot
    """
    thresholds = np.arange(0.25, 5.25, 0.25)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Vina baseline (same for all models — use first result)
    first_metrics = next(iter(model_results.values()))
    vina_rmsds = np.array(first_metrics['vina_selected_rmsds'])
    vina_rates = [(vina_rmsds < t).mean() for t in thresholds]
    ax.plot(thresholds, vina_rates, 'k--', lw=2, label='Vina (baseline)', zorder=5)

    colors = plt.cm.tab10(np.linspace(0, 0.6, len(model_results)))
    for (label, metrics), color in zip(model_results.items(), colors):
        xgb_rmsds = np.array(metrics['xgb_selected_rmsds'])
        rates = [(xgb_rmsds < t).mean() for t in thresholds]
        ax.plot(thresholds, rates, lw=2, label=label, color=color)

    ax.axvline(x=2.0, color='gray', linestyle=':', alpha=0.7, label='2 Å threshold')
    ax.set_xlabel('RMSD Threshold (Å)', fontsize=12)
    ax.set_ylabel('Fraction of Complexes with Pose within Threshold', fontsize=12)
    ax.set_title('Pose Selection Success Rate vs RMSD Threshold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'success_vs_threshold.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved success-vs-threshold plot to {output_dir / 'success_vs_threshold.png'}")


def plot_rmsd_cdf(model_results: Dict, output_dir: Path) -> None:
    """
    Plot empirical CDF of selected-pose RMSD for each model and the Vina baseline.
    Curves that rise faster (more mass at low RMSD) indicate better rescoring.

    Args:
        model_results: Dict mapping model label -> evaluate_ranker() result dict
        output_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    first_metrics = next(iter(model_results.values()))
    vina_rmsds = np.sort(first_metrics['vina_selected_rmsds'])
    y = np.arange(1, len(vina_rmsds) + 1) / len(vina_rmsds)
    ax.plot(vina_rmsds, y, 'k--', lw=2, label='Vina (baseline)', zorder=5)

    colors = plt.cm.tab10(np.linspace(0, 0.6, len(model_results)))
    for (label, metrics), color in zip(model_results.items(), colors):
        xgb_rmsds = np.sort(metrics['xgb_selected_rmsds'])
        y = np.arange(1, len(xgb_rmsds) + 1) / len(xgb_rmsds)
        ax.plot(xgb_rmsds, y, lw=2, label=label, color=color)

    ax.axvline(x=2.0, color='gray', linestyle=':', alpha=0.7, label='2 Å threshold')
    ax.set_xlabel('RMSD to Crystal Structure (Å)', fontsize=12)
    ax.set_ylabel('Cumulative Fraction of Complexes', fontsize=12)
    ax.set_title('CDF of Selected Pose RMSD', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'rmsd_cdf.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved RMSD CDF plot to {output_dir / 'rmsd_cdf.png'}")


def plot_spearman_per_complex(model_results: Dict, output_dir: Path) -> None:
    """
    For each test complex, compute the Spearman rank correlation between model score and
    per-pose RMSD. A negative correlation means the model correctly ranks lower-RMSD poses
    higher. Shown as overlapping histograms, one per model.

    Args:
        model_results: Dict mapping model label -> evaluate_ranker() result dict
                       (each result['test_df'] must have predicted_proba and rmsd columns)
        output_dir: Directory to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 0.6, len(model_results)))
    for (label, metrics), color in zip(model_results.items(), colors):
        df = metrics['test_df']
        rhos = []
        for pdb_code in df['pdb_code'].unique():
            cdf = df[df['pdb_code'] == pdb_code]
            if len(cdf) >= 2:
                rho, _ = spearmanr(cdf['predicted_proba'].values, cdf['rmsd'].values)
                if not np.isnan(rho):
                    rhos.append(rho)
        ax.hist(rhos, bins=30, alpha=0.5, color=color, label=f'{label} (mean={np.mean(rhos):.2f})')

    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Spearman ρ (model score vs RMSD) per complex', fontsize=12)
    ax.set_ylabel('Number of Complexes', fontsize=12)
    ax.set_title('Per-Complex Score–RMSD Ranking Quality\n'
                 '(negative ρ = model ranks lower-RMSD poses higher)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'spearman_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Spearman distribution plot to {output_dir / 'spearman_distribution.png'}")


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


def load_data_from_csv(
    train_csv: Path,
    test_csv: Path,
    val_csv: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """
    Load existing training, validation (optional), and test data from CSV files.

    Args:
        train_csv: Path to training data CSV
        test_csv:  Path to test data CSV
        val_csv:   Optional path to validation data CSV

    Returns:
        Tuple of (train_df, val_df or None, test_df)
    """
    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)
    val_df   = None
    if val_csv is not None:
        if not val_csv.exists():
            raise FileNotFoundError(f"Validation CSV not found: {val_csv}")
        val_df = pd.read_csv(val_csv)
        print(f"Loaded validation data: {val_df.shape}")

    print(f"Loaded training data: {train_df.shape}")
    print(f"Loaded test data: {test_df.shape}")

    return train_df, val_df, test_df


def split_val_from_train(
    train_df: pd.DataFrame,
    val_frac: float = 0.2,
    random_seed: int = RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carve out a validation set from the training DataFrame by complex.

    Complexes (not individual poses) are split to avoid data leakage.

    Args:
        train_df:    Per-pose training DataFrame with 'pdb_code' column.
        val_frac:    Fraction of complexes to use for validation (default 0.2).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (reduced_train_df, val_df).
    """
    unique = train_df['pdb_code'].unique()
    train_codes, val_codes = train_test_split(
        unique, test_size=val_frac, random_state=random_seed
    )
    return (
        train_df[train_df['pdb_code'].isin(train_codes)].copy(),
        train_df[train_df['pdb_code'].isin(val_codes)].copy(),
    )


def process_all_complexes(
    num_complexes: Optional[int] = None,
    complexes: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Process complexes and run Vina docking.

    Args:
        num_complexes: Limit number of complexes (None for all). Ignored when
                       `complexes` is provided explicitly.
        complexes:     Explicit list of PDB codes to process. If None, discovers
                       non-CASF complexes from STRUCTURES_DIR via get_complexes().

    Returns:
        List of Vina results with pose coordinates and RMSDs
    """
    print("Loading binding data...")
    binding_data = load_binding_data()
    print(f"Loaded {len(binding_data)} binding affinity records")

    if complexes is None:
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
        nargs='+',
        metavar='CSV',
        help='Load pre-computed pose CSVs instead of running Vina. '
             'Provide TRAIN_CSV TEST_CSV [VAL_CSV]. '
             'TEST_CSV should be the fixed CASF-2016 set. '
             'If VAL_CSV is omitted, a validation set is carved from TRAIN_CSV '
             'using --val-frac.'
    )
    parser.add_argument(
        '--val-frac',
        type=float,
        default=0.1,
        help='Fraction of non-CASF complexes to reserve as validation set '
             '(default: 0.1)'
    )
    parser.add_argument(
        '--casf-dir',
        type=str,
        default=str(CASF_DIR),
        help=f'Path to CASF-2016 coreset directory (default: {CASF_DIR})'
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
    parser.add_argument(
        '--no-contact-features',
        action='store_true',
        help='Skip protein-ligand contact feature augmentation. Use with --load-csv to compare '
             'performance with vs without contact features while keeping all other augmentation.'
    )
    parser.add_argument(
        '--no-affinity',
        action='store_true',
        help='Skip binding affinity prediction step (faster if only pose ranking is needed)'
    )
    parser.add_argument(
        '--optimize-hyperparams',
        action='store_true',
        help='Run Optuna hyperparameter search for RF ranker and GB affinity (~20 min extra)'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=30,
        help='Number of Optuna trials per model when --optimize-hyperparams is set (default: 30)'
    )
    parser.add_argument(
        '--optuna-db',
        type=str,
        default=None,
        help='Path to SQLite DB for saving/resuming Optuna trials '
             '(e.g. output/optuna_studies.db). Omit to run in-memory only (no persistence).'
    )
    parser.add_argument(
        '--load-hyperparams',
        type=str,
        default=None,
        metavar='JSON_PATH',
        help='Load previously optimised hyperparameters from a JSON file and apply them '
             'without running Optuna (e.g. output/best_hyperparams.json). '
             'If --optimize-hyperparams is also set, Optuna runs and overwrites these.'
    )
    parser.add_argument(
        '--load-hyperparams-fp',
        type=str,
        default=None,
        metavar='JSON_PATH',
        help='Load previously optimised hyperparameters for fingerprint-only affinity models '
             '(e.g. output/best_hyperparams_fp.json). Used with --affinity-compare-features.'
    )
    parser.add_argument(
        '--affinity-compare-features',
        action='store_true',
        help='Compare affinity models trained on current features vs Morgan fingerprints only. '
             'Generates separate plots and a side-by-side summary table.'
    )
    _ALL_OPT_MODELS = [
        'rf_ranker', 'gb_ranker', 'xgb_ranker',
        'rf_affinity', 'gb_affinity', 'svr_affinity', 'xgb_affinity', 'ridge_affinity', 'mlp_affinity',
        'rf_affinity_fp', 'gb_affinity_fp', 'svr_affinity_fp', 'xgb_affinity_fp',
        'ridge_affinity_fp', 'mlp_affinity_fp',
    ]
    parser.add_argument(
        '--optimize-models',
        nargs='+',
        choices=_ALL_OPT_MODELS,
        metavar='MODEL',
        default=None,
        help=(
            'Restrict hyperparameter optimisation to specific models '
            '(requires --optimize-hyperparams). '
            f'Choices: {", ".join(_ALL_OPT_MODELS)}. '
            'Default: all models. '
            'Example: --optimize-models mlp_affinity gb_affinity_fp mlp_affinity_fp'
        )
    )
    args = parser.parse_args()

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.load_csv:
        # Load existing CSV files
        print("Loading data from CSV files...")
        csvs = args.load_csv
        val_csv = Path(csvs[2]) if len(csvs) >= 3 else None
        train_df, val_df, test_df = load_data_from_csv(
            Path(csvs[0]), Path(csvs[1]), val_csv
        )
        if val_df is None:
            train_df, val_df = split_val_from_train(train_df, args.val_frac)
            print(f"  Val set carved from training data: "
                  f"{val_df['pdb_code'].nunique()} complexes "
                  f"({val_df.shape[0]} poses)")
            val_df.to_csv(OUTPUT_DIR / "val_data.csv", index=False)
            print(f"  Saved val set to {OUTPUT_DIR / 'val_data.csv'}")
            print(f"  Remaining training: "
                  f"{train_df['pdb_code'].nunique()} complexes "
                  f"({train_df.shape[0]} poses)")
    else:
        # --- Process CASF-2016 complexes (fixed test set) ---
        casf_dir = Path(args.casf_dir)
        print(f"\nProcessing CASF-2016 test complexes from {casf_dir}...")
        casf_codes_list = get_casf_complexes(casf_dir)
        print(f"Found {len(casf_codes_list)} valid CASF-2016 complexes")
        casf_vina_results = process_all_complexes(complexes=casf_codes_list)
        print(f"Processed {len(casf_vina_results)} CASF-2016 complexes successfully")
        test_df = prepare_training_data(casf_vina_results)

        # --- Process PDBind_2020 non-CASF complexes (train + val pool) ---
        print(f"\nProcessing PDBind_2020 train/val complexes (CASF excluded)...")
        vina_results = process_all_complexes(num_complexes=args.num_complexes)
        print(f"Processed {len(vina_results)} PDBind_2020 complexes successfully")
        pool_df = prepare_training_data(vina_results)

        if len(pool_df) < 10:
            print("Not enough train/val data for training")
            return

        # Split pool into 90% train / 10% val by complex
        unique_complexes = pool_df['pdb_code'].unique()
        train_complexes, val_complexes = train_test_split(
            unique_complexes, test_size=args.val_frac, random_state=RANDOM_SEED
        )
        train_df_final = pool_df[pool_df['pdb_code'].isin(train_complexes)].copy()
        val_df         = pool_df[pool_df['pdb_code'].isin(val_complexes)].copy()

        print(f"\nDataset split summary:")
        print(f"  Test  (CASF-2016) : {test_df['pdb_code'].nunique()} complexes "
              f"({test_df.shape[0]} poses)")
        print(f"  Train             : {train_df_final['pdb_code'].nunique()} complexes "
              f"({train_df_final.shape[0]} poses)")
        print(f"  Val               : {val_df['pdb_code'].nunique()} complexes "
              f"({val_df.shape[0]} poses)")

        # Save fixed CSVs (reused for all subsequent --load-csv runs)
        train_df_final.to_csv(OUTPUT_DIR / "training_data.csv", index=False)
        val_df.to_csv(OUTPUT_DIR / "val_data.csv", index=False)
        test_df.to_csv(OUTPUT_DIR / "test_data.csv", index=False)
        print(f"Fixed train/val/test CSVs saved to {OUTPUT_DIR}/")

        train_df = train_df_final

    # --- Feature augmentation ---
    if not args.no_augment:
        print("\nAugmenting features (this may take a few minutes)...")

        print("  Adding pose geometry from PDBQT files...")
        train_df = augment_with_pdbqt_features(train_df)
        val_df   = augment_with_pdbqt_features(val_df)
        test_df  = augment_with_pdbqt_features(test_df)

        print("  Adding molecular descriptors from mol2 files...")
        train_df = augment_with_mol2_features(train_df)
        val_df   = augment_with_mol2_features(val_df)
        test_df  = augment_with_mol2_features(test_df)

        if not args.no_contact_features:
            print("  Adding protein-ligand contact features (may take ~5-10 min)...")
            train_df = augment_with_contact_features(train_df)
            val_df   = augment_with_contact_features(val_df)
            test_df  = augment_with_contact_features(test_df)
        else:
            print("  Skipping protein-ligand contact features (--no-contact-features set)")

        print("  Engineering derived features...")
        train_df = engineer_features(train_df)
        val_df   = engineer_features(val_df)
        test_df  = engineer_features(test_df)

        print(f"  Feature count after augmentation: {train_df.shape[1]} columns")

    # --- Load pre-computed hyperparameters if requested ---
    optuna_storage = f"sqlite:///{args.optuna_db}" if args.optuna_db else None
    rf_hyperparams      = None
    gb_hyperparams      = None
    xgb_hyperparams     = None
    rf_affinity_preload    = None
    gb_affinity_preload    = None
    svr_affinity_preload   = None
    xgb_affinity_preload   = None
    ridge_affinity_preload = None
    mlp_affinity_preload   = None
    if args.load_hyperparams:
        hp_path = Path(args.load_hyperparams)
        if not hp_path.exists():
            print(f"Warning: --load-hyperparams file not found: {hp_path}. Using defaults.")
        else:
            loaded_hp = json.loads(hp_path.read_text())
            rf_hyperparams         = loaded_hp.get('rf_ranker',      {}) or None
            gb_hyperparams         = loaded_hp.get('gb_ranker',      {}) or None
            xgb_hyperparams        = loaded_hp.get('xgb_ranker',     {}) or None
            rf_affinity_preload    = loaded_hp.get('rf_affinity',    {}) or None
            gb_affinity_preload    = loaded_hp.get('gb_affinity',    {}) or None
            svr_affinity_preload   = loaded_hp.get('svr_affinity',   {}) or None
            xgb_affinity_preload   = loaded_hp.get('xgb_affinity',   {}) or None
            ridge_affinity_preload = loaded_hp.get('ridge_affinity', {}) or None
            mlp_affinity_preload   = loaded_hp.get('mlp_affinity',   {}) or None
            print(f"Loaded hyperparameters from {hp_path}")
            if rf_hyperparams:
                print(f"  RF ranker       : {rf_hyperparams}")
            if gb_hyperparams:
                print(f"  GB ranker       : {gb_hyperparams}")
            if xgb_hyperparams:
                print(f"  XGB ranker      : {xgb_hyperparams}")
            if rf_affinity_preload:
                print(f"  RF affinity     : {rf_affinity_preload}")
            if gb_affinity_preload:
                print(f"  GB affinity     : {gb_affinity_preload}")
            if svr_affinity_preload:
                print(f"  SVR affinity    : {svr_affinity_preload}")
            if xgb_affinity_preload:
                print(f"  XGB affinity    : {xgb_affinity_preload}")
            if ridge_affinity_preload:
                print(f"  Ridge affinity  : {ridge_affinity_preload}")
            if mlp_affinity_preload:
                print(f"  MLP affinity    : {mlp_affinity_preload}")

    # --- Load fingerprint hyperparameters if requested ---
    rf_affinity_hp_fp    = None
    gb_affinity_hp_fp    = None
    svr_affinity_hp_fp   = None
    xgb_affinity_hp_fp   = None
    ridge_affinity_hp_fp = None
    mlp_affinity_hp_fp   = None
    if args.load_hyperparams_fp:
        hp_fp_path = Path(args.load_hyperparams_fp)
        if not hp_fp_path.exists():
            print(f"Warning: --load-hyperparams-fp file not found: {hp_fp_path}. Using defaults.")
        else:
            loaded_hp_fp = json.loads(hp_fp_path.read_text())
            rf_affinity_hp_fp    = loaded_hp_fp.get('rf_affinity',    {}) or None
            gb_affinity_hp_fp    = loaded_hp_fp.get('gb_affinity',    {}) or None
            svr_affinity_hp_fp   = loaded_hp_fp.get('svr_affinity',   {}) or None
            xgb_affinity_hp_fp   = loaded_hp_fp.get('xgb_affinity',   {}) or None
            ridge_affinity_hp_fp = loaded_hp_fp.get('ridge_affinity', {}) or None
            mlp_affinity_hp_fp   = loaded_hp_fp.get('mlp_affinity',   {}) or None
            print(f"Loaded fingerprint hyperparameters from {hp_fp_path}")

    # --- Optional: Optuna hyperparameter search ---
    # models_to_opt: set of model keys to optimise; None means all
    models_to_opt = set(args.optimize_models) if args.optimize_models else set(_ALL_OPT_MODELS)

    if args.optimize_hyperparams:
        feature_cols_for_opt = [c for c in train_df.columns
                                if c not in ['pdb_code', 'pose_idx', 'is_best_pose', 'rmsd']]
        if 'rf_ranker' in models_to_opt:
            print(f"\nOptimising RF ranker hyperparameters "
                  f"(Optuna, {args.n_trials} trials × 5-fold GroupKFold on train+val)...")
            rf_hyperparams = optimize_ranker_hyperparams(
                train_df, val_df, feature_cols_for_opt, args.n_trials, storage=optuna_storage
            )
            print(f"  Best RF params: {rf_hyperparams}")

        if 'gb_ranker' in models_to_opt:
            print(f"\nOptimising GB ranker hyperparameters "
                  f"(Optuna, {args.n_trials} trials × 5-fold GroupKFold on train+val)...")
            gb_hyperparams = optimize_gb_ranker_hyperparams(
                train_df, val_df, feature_cols_for_opt, args.n_trials, storage=optuna_storage
            )
            print(f"  Best GB params: {gb_hyperparams}")

        if 'xgb_ranker' in models_to_opt:
            print(f"\nOptimising XGB ranker hyperparameters "
                  f"(Optuna, {args.n_trials} trials × 5-fold GroupKFold on train+val)...")
            xgb_hyperparams = optimize_xgb_ranker_hyperparams(
                train_df, val_df, feature_cols_for_opt, args.n_trials, storage=optuna_storage
            )
            print(f"  Best XGB params: {xgb_hyperparams}")

    # --- Train all ranker models ---
    trained = {}
    print(f"\nTraining xgb_ranker...")
    trained['xgb_ranker'] = train_ranker_model(train_df, hyperparams=xgb_hyperparams)
    print(f"Training rf_ranker...")
    trained['rf_ranker'] = train_rf_ranker(train_df, hyperparams=rf_hyperparams)
    print(f"Training gb_ranker...")
    trained['gb_ranker'] = train_gb_ranker(train_df, hyperparams=gb_hyperparams)

    # --- Evaluate all individual models on test and val sets ---
    model_results = {}
    val_results = {}
    for name, (model, feat_cols) in trained.items():
        print(f"Evaluating {name} (test)...")
        model_results[name] = evaluate_ranker(model, test_df, feat_cols)
        print(f"Evaluating {name} (val)...")
        val_results[name] = evaluate_ranker(model, val_df, feat_cols)

    # --- Ensemble: average normalised scores from all three rankers ---
    print("Evaluating ensemble (test)...")
    model_results['ensemble'] = evaluate_ensemble(model_results, test_df)
    print("Evaluating ensemble (val)...")
    val_results['ensemble'] = evaluate_ensemble(val_results, val_df)

    # Choose best model by 2 Å success rate (tie-break: best-pose selection rate)
    best_name = max(
        model_results,
        key=lambda k: (model_results[k]['xgb_success_2A_rate'],
                       model_results[k]['xgb_success_rate'])
    )
    best_metrics = model_results[best_name]
    best_model, _ = trained[best_name]

    # --- Print results ---
    n_test = best_metrics['total_complexes']
    n_val  = val_results[best_name]['total_complexes']
    print(f"\n{'='*60}")
    print("POSE SELECTION PERFORMANCE")
    print(f"{'='*60}")
    print(f"Test complexes : {n_test}   Val complexes : {n_val}")
    print(f"\nAutoDock Vina (baseline):")
    print(f"  Best-pose selection rate : "
          f"val={val_results[best_name]['vina_success_rate']:.3f}  "
          f"test={best_metrics['vina_success_rate']:.3f}")
    print(f"  Poses within 2 Å        : "
          f"val={val_results[best_name]['vina_success_2A_rate']:.3f}  "
          f"test={best_metrics['vina_success_2A_rate']:.3f}")
    print(f"  Mean RMSD of selected   : "
          f"val={val_results[best_name]['mean_vina_rmsd']:.3f}  "
          f"test={best_metrics['mean_vina_rmsd']:.3f} Å")

    for name, metrics in model_results.items():
        vm = val_results[name]
        print(f"\n--- {name} ---")
        print(f"  Best-pose selection rate : "
              f"val={vm['xgb_success_rate']:.3f}  test={metrics['xgb_success_rate']:.3f}")
        print(f"  Poses within 2 Å        : "
              f"val={vm['xgb_success_2A_rate']:.3f}  test={metrics['xgb_success_2A_rate']:.3f}")
        print(f"  Mean/Median RMSD        : "
              f"val={vm['mean_xgb_rmsd']:.3f}/{vm['median_xgb_rmsd']:.3f}  "
              f"test={metrics['mean_xgb_rmsd']:.3f}/{metrics['median_xgb_rmsd']:.3f} Å")
        print(f"  RMSD improvement vs Vina: "
              f"val={vm['mean_rmsd_improvement']:+.3f}  "
              f"test={metrics['mean_rmsd_improvement']:+.3f} Å")
        print(f"  Spearman score–RMSD corr: "
              f"val={vm['spearman_mean']:.3f}  test={metrics['spearman_mean']:.3f}")

    print(f"\n{'='*60}")
    print(f"Best model: {best_name}")

    # --- Save performance metrics ---
    metrics_file = OUTPUT_DIR / 'pose_selection_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("POSE SELECTION PERFORMANCE METRICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Val complexes : {n_val}   Test complexes : {n_test}\n\n")
        f.write("AutoDock Vina (baseline):\n")
        _bv = val_results[best_name]
        f.write(f"  Best-pose selection rate : "
                f"val={_bv['vina_success_rate']:.3f}  "
                f"test={best_metrics['vina_success_rate']:.3f}\n")
        f.write(f"  Poses within 2 Å        : "
                f"val={_bv['vina_success_2A_rate']:.3f}  "
                f"test={best_metrics['vina_success_2A_rate']:.3f}\n")
        f.write(f"  Mean RMSD               : "
                f"val={_bv['mean_vina_rmsd']:.3f}  "
                f"test={best_metrics['mean_vina_rmsd']:.3f} Å\n\n")
        for name, metrics in model_results.items():
            vm = val_results[name]
            f.write(f"{name}{'  <-- best' if name == best_name else ''}:\n")
            f.write(f"  Best-pose selection rate : "
                    f"val={vm['xgb_success_rate']:.3f}  "
                    f"test={metrics['xgb_success_rate']:.3f}\n")
            f.write(f"  Poses within 2 Å        : "
                    f"val={vm['xgb_success_2A_rate']:.3f}  "
                    f"test={metrics['xgb_success_2A_rate']:.3f}\n")
            f.write(f"  Mean/Median RMSD        : "
                    f"val={vm['mean_xgb_rmsd']:.3f}/{vm['median_xgb_rmsd']:.3f}  "
                    f"test={metrics['mean_xgb_rmsd']:.3f}/{metrics['median_xgb_rmsd']:.3f} Å\n")
            f.write(f"  RMSD improvement vs Vina: "
                    f"val={vm['mean_rmsd_improvement']:+.3f}  "
                    f"test={metrics['mean_rmsd_improvement']:+.3f} Å\n")
            f.write(f"  Spearman score-RMSD corr: "
                    f"val={vm['spearman_mean']:.3f}  "
                    f"test={metrics['spearman_mean']:.3f}\n\n")

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

            # New diagnostic plots (all models shown together)
            plot_success_vs_rmsd_threshold(model_results, OUTPUT_DIR)
            plot_rmsd_cdf(model_results, OUTPUT_DIR)
            plot_spearman_per_complex(model_results, OUTPUT_DIR)

            print("\nAll plots saved successfully!")
        except Exception as e:
            print(f"\nWarning: Could not generate some plots: {e}")
            import traceback
            traceback.print_exc()

    # ================================================================
    # --- Binding affinity prediction (rank-1 pose → ΔG regression) ---
    # ================================================================
    if not args.no_affinity:
        print("\n" + "=" * 60)
        print("BINDING AFFINITY PREDICTION")
        print("=" * 60)

        print("\nJoining affinity labels...")
        train_df_aff = join_affinity_labels(train_df)
        val_df_aff   = join_affinity_labels(val_df)
        test_df_aff  = join_affinity_labels(test_df)

        _aff_exclude = {'pdb_code', 'pose_idx', 'is_best_pose', 'rmsd',
                        'exp_affinity_pKd', 'exp_affinity_kcal_mol'}
        feature_cols_aff = [c for c in train_df.columns if c not in _aff_exclude]

        print("Preparing affinity datasets...")
        train_aff = prepare_affinity_data(train_df_aff, feature_cols_aff)
        val_aff   = prepare_affinity_data(val_df_aff,   feature_cols_aff)
        test_aff  = prepare_affinity_data(test_df_aff,  feature_cols_aff)

        if len(train_aff) < 10 or len(test_aff) < 2:
            print("  Warning: Too few complexes with affinity labels. Skipping affinity models.")
        else:
            rf_affinity_hyperparams    = rf_affinity_preload    # None unless --load-hyperparams
            gb_affinity_hyperparams    = gb_affinity_preload
            svr_affinity_hyperparams   = svr_affinity_preload
            xgb_affinity_hyperparams   = xgb_affinity_preload
            ridge_affinity_hyperparams = ridge_affinity_preload
            mlp_affinity_hyperparams   = mlp_affinity_preload
            if args.optimize_hyperparams:
                if 'rf_affinity' in models_to_opt:
                    print(f"Optimising RF affinity hyperparameters "
                          f"(Optuna, {args.n_trials} trials × 5-fold KFold on train+val)...")
                    rf_affinity_hyperparams = optimize_rf_affinity_hyperparams(
                        train_aff, val_aff, feature_cols_aff, args.n_trials, storage=optuna_storage
                    )
                    print(f"  Best RF affinity params: {rf_affinity_hyperparams}")

                if 'gb_affinity' in models_to_opt:
                    print(f"Optimising GB affinity hyperparameters "
                          f"(Optuna, {args.n_trials} trials × 5-fold KFold on train+val)...")
                    gb_affinity_hyperparams = optimize_affinity_hyperparams(
                        train_aff, val_aff, feature_cols_aff, args.n_trials, storage=optuna_storage
                    )
                    print(f"  Best GB affinity params: {gb_affinity_hyperparams}")

                if 'svr_affinity' in models_to_opt:
                    print(f"\nOptimising SVR affinity hyperparameters "
                          f"(Optuna, {args.n_trials} trials × 5-fold KFold on train+val)...")
                    svr_affinity_hyperparams = optimize_svr_affinity_hyperparams(
                        train_aff, val_aff, feature_cols_aff, args.n_trials, storage=optuna_storage
                    )
                    print(f"  Best SVR affinity params: {svr_affinity_hyperparams}")

                if 'xgb_affinity' in models_to_opt:
                    print(f"\nOptimising XGB affinity hyperparameters "
                          f"(Optuna, {args.n_trials} trials × 5-fold KFold on train+val)...")
                    xgb_affinity_hyperparams = optimize_xgb_affinity_hyperparams(
                        train_aff, val_aff, feature_cols_aff, args.n_trials, storage=optuna_storage
                    )
                    print(f"  Best XGB affinity params: {xgb_affinity_hyperparams}")

                if 'ridge_affinity' in models_to_opt:
                    print(f"\nOptimising Ridge affinity hyperparameters "
                          f"(Optuna, {args.n_trials} trials × 5-fold KFold on train+val)...")
                    ridge_affinity_hyperparams = optimize_ridge_affinity_hyperparams(
                        train_aff, val_aff, feature_cols_aff, args.n_trials, storage=optuna_storage
                    )
                    print(f"  Best Ridge affinity params: {ridge_affinity_hyperparams}")

                if 'mlp_affinity' in models_to_opt:
                    print(f"\nOptimising MLP affinity hyperparameters "
                          f"(Optuna, {args.n_trials} trials × 5-fold KFold on train+val)...")
                    mlp_affinity_hyperparams = optimize_mlp_affinity_hyperparams(
                        train_aff, val_aff, feature_cols_aff, args.n_trials, storage=optuna_storage
                    )
                    print(f"  Best MLP affinity params: {mlp_affinity_hyperparams}")

            print("Training affinity models...")
            affinity_models = train_affinity_model(
                train_aff, feature_cols_aff,
                hyperparams_rf=rf_affinity_hyperparams,
                hyperparams_gb=gb_affinity_hyperparams,
                hyperparams_svr=svr_affinity_hyperparams,
                hyperparams_xgb=xgb_affinity_hyperparams,
                hyperparams_ridge=ridge_affinity_hyperparams,
                hyperparams_mlp=mlp_affinity_hyperparams,
                val_affinity_df=val_aff,
            )

            print("Evaluating affinity models (test)...")
            affinity_metrics = evaluate_affinity_model(affinity_models, test_aff, feature_cols_aff)
            print("Evaluating affinity models (val)...")
            val_affinity_metrics = evaluate_affinity_model(affinity_models, val_aff, feature_cols_aff)

            n_aff_test = len(test_aff)
            n_aff_val  = len(val_aff)
            print(f"\n{'='*60}")
            print("BINDING AFFINITY PERFORMANCE")
            print(f"{'='*60}")
            print(f"Val complexes: {n_aff_val}   Test complexes: {n_aff_test}")

            print(f"\nVina baseline (rank-1 score vs exp ΔG):")
            vm_t = affinity_metrics['vina_baseline']
            vm_v = val_affinity_metrics['vina_baseline']
            print(f"  Pearson r  : val={vm_v['pearson_r']:+.3f}  test={vm_t['pearson_r']:+.3f}")
            print(f"  Spearman r : val={vm_v['spearman_r']:+.3f}  test={vm_t['spearman_r']:+.3f}")
            print(f"  RMSE       : val={vm_v['rmse']:.3f}  test={vm_t['rmse']:.3f} kcal/mol")
            print(f"  MAE        : val={vm_v['mae']:.3f}  test={vm_t['mae']:.3f} kcal/mol")

            for name in sorted(k for k in affinity_metrics if k != 'vina_baseline'):
                am_t = affinity_metrics[name]
                am_v = val_affinity_metrics[name]
                print(f"\n--- {name} ---")
                print(f"  Pearson r  : val={am_v['pearson_r']:+.3f}  test={am_t['pearson_r']:+.3f}")
                print(f"  Spearman r : val={am_v['spearman_r']:+.3f}  test={am_t['spearman_r']:+.3f}")
                print(f"  RMSE       : val={am_v['rmse']:.3f}  test={am_t['rmse']:.3f} kcal/mol")
                print(f"  MAE        : val={am_v['mae']:.3f}  test={am_t['mae']:.3f} kcal/mol")

            # Save best hyperparameters if optimisation was run
            if args.optimize_hyperparams:
                best_hp = {
                    'rf_ranker':      rf_hyperparams             or {},
                    'gb_ranker':      gb_hyperparams             or {},
                    'xgb_ranker':     xgb_hyperparams            or {},
                    'rf_affinity':    rf_affinity_hyperparams    or {},
                    'gb_affinity':    gb_affinity_hyperparams    or {},
                    'svr_affinity':   svr_affinity_hyperparams   or {},
                    'xgb_affinity':   xgb_affinity_hyperparams   or {},
                    'ridge_affinity': ridge_affinity_hyperparams or {},
                    'mlp_affinity':   mlp_affinity_hyperparams   or {},
                }
                hp_path = OUTPUT_DIR / 'best_hyperparams.json'
                hp_path.write_text(json.dumps(best_hp, indent=2))
                print(f"\nSaved best hyperparameters to {hp_path}")

            print("\nExporting affinity predictions CSV...")
            export_affinity_predictions_csv(test_aff, affinity_metrics, OUTPUT_DIR)

            if not args.no_plots:
                print("Generating affinity scatter plots (test + val)...")
                try:
                    plot_affinity_predictions(affinity_metrics,     OUTPUT_DIR, split='test')
                    plot_affinity_predictions(val_affinity_metrics, OUTPUT_DIR, split='val')
                except Exception as e:
                    print(f"Warning: Could not generate affinity plot: {e}")

            # ---- Feature-set comparison: current features vs fingerprints only ----
            if args.affinity_compare_features:
                print("\n" + "=" * 60)
                print("AFFINITY FEATURE-SET COMPARISON: current vs fingerprints")
                print("=" * 60)

                print("\nComputing Morgan fingerprints (ECFP4, 2048 bits) for all splits...")
                train_aff_fp = augment_affinity_with_fingerprints(train_aff)
                val_aff_fp   = augment_affinity_with_fingerprints(val_aff)
                test_aff_fp  = augment_affinity_with_fingerprints(test_aff)

                fp_cols = [c for c in train_aff_fp.columns if c.startswith('morgan_fp_')]
                # Keep only fingerprint bits that have at least one non-NaN value
                valid_fp_cols = [c for c in fp_cols
                                 if train_aff_fp[c].notna().any()]
                print(f"  {len(valid_fp_cols)} fingerprint bits with data (out of {len(fp_cols)})")

                if len(valid_fp_cols) < 10:
                    print("  Warning: Too few valid fingerprint bits. Skipping fingerprint models.")
                else:
                    # Optimise fingerprint models if requested (separate studies)
                    if args.optimize_hyperparams:
                        if 'rf_affinity_fp' in models_to_opt:
                            print(f"\nOptimising RF affinity hyperparameters (fingerprints, "
                                  f"{args.n_trials} trials)...")
                            rf_affinity_hp_fp = optimize_rf_affinity_hyperparams(
                                train_aff_fp, val_aff_fp, valid_fp_cols,
                                args.n_trials, storage=optuna_storage,
                                study_name='rf_affinity_fp',
                            )
                            print(f"  Best RF fp params: {rf_affinity_hp_fp}")

                        if 'gb_affinity_fp' in models_to_opt:
                            print(f"\nOptimising GB affinity hyperparameters (fingerprints, "
                                  f"{args.n_trials} trials)...")
                            gb_affinity_hp_fp = optimize_affinity_hyperparams(
                                train_aff_fp, val_aff_fp, valid_fp_cols,
                                args.n_trials, storage=optuna_storage,
                                study_name='gb_affinity_fp',
                            )
                            print(f"  Best GB fp params: {gb_affinity_hp_fp}")

                        if 'svr_affinity_fp' in models_to_opt:
                            print(f"\nOptimising SVR affinity hyperparameters (fingerprints, "
                                  f"{args.n_trials} trials)...")
                            svr_affinity_hp_fp = optimize_svr_affinity_hyperparams(
                                train_aff_fp, val_aff_fp, valid_fp_cols,
                                args.n_trials, storage=optuna_storage,
                                study_name='svr_affinity_fp',
                            )
                            print(f"  Best SVR fp params: {svr_affinity_hp_fp}")

                        if 'xgb_affinity_fp' in models_to_opt:
                            print(f"\nOptimising XGB affinity hyperparameters (fingerprints, "
                                  f"{args.n_trials} trials)...")
                            xgb_affinity_hp_fp = optimize_xgb_affinity_hyperparams(
                                train_aff_fp, val_aff_fp, valid_fp_cols,
                                args.n_trials, storage=optuna_storage,
                                study_name='xgb_affinity_fp',
                            )
                            print(f"  Best XGB fp params: {xgb_affinity_hp_fp}")

                        if 'ridge_affinity_fp' in models_to_opt:
                            print(f"\nOptimising Ridge affinity hyperparameters (fingerprints, "
                                  f"{args.n_trials} trials)...")
                            ridge_affinity_hp_fp = optimize_ridge_affinity_hyperparams(
                                train_aff_fp, val_aff_fp, valid_fp_cols,
                                args.n_trials, storage=optuna_storage,
                                study_name='ridge_affinity_fp',
                            )
                            print(f"  Best Ridge fp params: {ridge_affinity_hp_fp}")

                        if 'mlp_affinity_fp' in models_to_opt:
                            print(f"\nOptimising MLP affinity hyperparameters (fingerprints, "
                                  f"{args.n_trials} trials)...")
                            mlp_affinity_hp_fp = optimize_mlp_affinity_hyperparams(
                                train_aff_fp, val_aff_fp, valid_fp_cols,
                                args.n_trials, storage=optuna_storage,
                                study_name='mlp_affinity_fp',
                            )
                            print(f"  Best MLP fp params: {mlp_affinity_hp_fp}")

                        # Save fingerprint hyperparameters to separate JSON
                        fp_models_optimised = {
                            'rf_affinity_fp', 'gb_affinity_fp', 'svr_affinity_fp',
                            'xgb_affinity_fp', 'ridge_affinity_fp', 'mlp_affinity_fp',
                        }
                        if models_to_opt & fp_models_optimised:
                            best_hp_fp = {
                                'rf_affinity':    rf_affinity_hp_fp    or {},
                                'gb_affinity':    gb_affinity_hp_fp    or {},
                                'svr_affinity':   svr_affinity_hp_fp   or {},
                                'xgb_affinity':   xgb_affinity_hp_fp   or {},
                                'ridge_affinity': ridge_affinity_hp_fp or {},
                                'mlp_affinity':   mlp_affinity_hp_fp   or {},
                            }
                            hp_fp_save = OUTPUT_DIR / 'best_hyperparams_fp.json'
                            hp_fp_save.write_text(json.dumps(best_hp_fp, indent=2))
                            print(f"\nSaved fingerprint hyperparameters to {hp_fp_save}")

                    print("\nTraining affinity models on fingerprints only...")
                    fp_affinity_models = train_affinity_model(
                        train_aff_fp, valid_fp_cols,
                        hyperparams_rf=rf_affinity_hp_fp,
                        hyperparams_gb=gb_affinity_hp_fp,
                        hyperparams_svr=svr_affinity_hp_fp,
                        hyperparams_xgb=xgb_affinity_hp_fp,
                        hyperparams_ridge=ridge_affinity_hp_fp,
                        hyperparams_mlp=mlp_affinity_hp_fp,
                    )

                    print("Evaluating fingerprint affinity models (test)...")
                    fp_affinity_metrics_test = evaluate_affinity_model(
                        fp_affinity_models, test_aff_fp, valid_fp_cols)
                    print("Evaluating fingerprint affinity models (val)...")
                    fp_affinity_metrics_val = evaluate_affinity_model(
                        fp_affinity_models, val_aff_fp, valid_fp_cols)

                    # Side-by-side comparison tables
                    print_affinity_comparison_table(
                        affinity_metrics, fp_affinity_metrics_test,
                        label_a='Current Features', label_b='Fingerprints Only',
                        split='test',
                    )
                    print_affinity_comparison_table(
                        val_affinity_metrics, fp_affinity_metrics_val,
                        label_a='Current Features', label_b='Fingerprints Only',
                        split='val',
                    )

                    if not args.no_plots:
                        print("Generating fingerprint affinity scatter plots...")
                        try:
                            plot_affinity_predictions(
                                affinity_metrics,        OUTPUT_DIR, split='current_test')
                            plot_affinity_predictions(
                                val_affinity_metrics,    OUTPUT_DIR, split='current_val')
                            plot_affinity_predictions(
                                fp_affinity_metrics_test, OUTPUT_DIR, split='fingerprints_test')
                            plot_affinity_predictions(
                                fp_affinity_metrics_val,  OUTPUT_DIR, split='fingerprints_val')
                        except Exception as e:
                            print(f"Warning: Could not generate fingerprint affinity plot: {e}")
    else:
        print("\nSkipping affinity prediction (--no-affinity set).")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
