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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


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


def run_vina(pdb_code: str) -> Optional[Dict]:
    """
    Run AutoDock Vina on a single complex.
    
    Args:
        pdb_code: PDB code of the complex
    
    Returns:
        Dictionary with docking results, or None if failed
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
    # Use -xr to create rigid receptor without torsion tree
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
    
    # Parse Vina output
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
    
    return {
        'pdb_code': pdb_code,
        'scores': scores,
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


def extract_vina_features(vina_result: Dict) -> Dict:
    """
    Extract features from Vina docking results.
    
    Args:
        vina_result: Dictionary from run_vina()
    
    Returns:
        Dictionary of Vina-derived features
    """
    scores = vina_result['scores']
    
    return {
        'vina_top_score': scores[0],
        'vina_mean_score': np.mean(scores),
        'vina_std_score': np.std(scores),
        'vina_score_range': scores[0] - scores[-1],
        'vina_num_poses': len(scores)
    }


def prepare_training_data(binding_data: pd.DataFrame, 
                         vina_results: List[Dict],
                         rdkit_features: Dict[str, Dict],
                         protein_features: Dict[str, Dict]) -> pd.DataFrame:
    """
    Prepare combined feature matrix for training.
    
    Args:
        binding_data: DataFrame with binding affinities
        vina_results: List of Vina docking results
        rdkit_features: Dictionary of RDKit features per PDB code
        protein_features: Dictionary of protein features per PDB code
    
    Returns:
        DataFrame with all features for training
    """
    records = []
    
    for result in vina_results:
        pdb_code = result['pdb_code']
        
        # Get binding affinity
        binding_row = binding_data[binding_data['pdb_code'] == pdb_code]
        if len(binding_row) == 0:
            continue
        
        affinity_nM = binding_row['affinity_nM'].values[0]
        
        # Get features
        rdkit = rdkit_features.get(pdb_code, {})
        vina = extract_vina_features(result)
        protein = protein_features.get(pdb_code, {})
        
        # Combine all features
        record = {
            'pdb_code': pdb_code,
            'affinity_nM': affinity_nM,
            'log_affinity': np.log10(affinity_nM + 1)
        }
        record.update(rdkit)
        record.update(vina)
        record.update(protein)
        
        records.append(record)
    
    return pd.DataFrame(records)


def train_rescoring_model(train_df: pd.DataFrame) -> xgb.XGBRegressor:
    """
    Train XGBoost model for pose rescoring.
    
    Args:
        train_df: DataFrame with features and binding affinities
    
    Returns:
        Trained XGBoost model
    """
    # Define feature columns (exclude identifiers and targets)
    feature_cols = [col for col in train_df.columns 
                   if col not in ['pdb_code', 'affinity_nM', 'log_affinity']]
    
    X = train_df[feature_cols].values
    y = train_df['log_affinity'].values
    
    # Handle missing values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Train model
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    return model


def evaluate_model(model: xgb.XGBRegressor, test_df: pd.DataFrame) -> Dict:
    """
    Evaluate trained model on test set.
    
    Args:
        model: Trained XGBoost model
        test_df: DataFrame with test features
    
    Returns:
        Dictionary of evaluation metrics
    """
    feature_cols = [col for col in test_df.columns 
                   if col not in ['pdb_code', 'affinity_nM', 'log_affinity']]
    
    X_test = test_df[feature_cols].values
    y_test = test_df['log_affinity'].values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'rmse': rmse,
        'r2': r2,
        'y_test': y_test,
        'y_pred': y_pred
    }


def process_all_complexes(num_complexes: Optional[int] = None) -> Tuple[pd.DataFrame, List[Dict], Dict, Dict]:
    """
    Process all complexes and run Vina docking.
    
    Args:
        num_complexes: Limit number of complexes (None for all)
    
    Returns:
        Tuple of (binding_data, vina_results, rdkit_features, protein_features)
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
    rdkit_features = {}
    protein_features = {}
    
    for i, pdb_code in enumerate(complexes):
        print(f"Processing {pdb_code} ({i+1}/{len(complexes)})...")
        
        # Run Vina
        vina_result = run_vina(pdb_code)
        if vina_result:
            vina_results.append(vina_result)
        
        # Extract RDKit features
        try:
            pdb_dir = get_complex_path(pdb_code)
            ligand_file = pdb_dir / f"{pdb_code}_ligand.mol2"
            if ligand_file.exists():
                features = extract_rdkit_features(str(ligand_file))
                if features:
                    rdkit_features[pdb_code] = features
        except FileNotFoundError:
            pass
    
    return binding_data, vina_results, rdkit_features, protein_features


def main():
    """
    Main function to run the complete workflow.
    """
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Process complexes
    num_complexes = 10  # Start with small number for testing
    binding_data, vina_results, rdkit_features, protein_features = \
        process_all_complexes(num_complexes)
    
    print(f"\nProcessed {len(vina_results)} complexes successfully")
    
    # Prepare training data
    print("Preparing training data...")
    train_df = prepare_training_data(
        binding_data, vina_results, rdkit_features, protein_features
    )
    print(f"Training data shape: {train_df.shape}")
    
    if len(train_df) < 10:
        print("Not enough data for training")
        return
    
    # Split data
    train_df, test_df = train_test_split(
        train_df, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Train model
    print("Training XGBoost model...")
    model = train_rescoring_model(train_df)
    
    # Evaluate
    print("Evaluating model...")
    metrics = evaluate_model(model, test_df)
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"R2: {metrics['r2']:.3f}")
    
    # Save results
    train_df.to_csv(OUTPUT_DIR / "training_data.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_data.csv", index=False)
    print(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
