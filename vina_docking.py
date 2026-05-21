"""
vina_docking.py — Standalone AutoDock Vina docking script for HPC SLURM job arrays.

Self-contained: requires only numpy, pandas, subprocess, and standard-library
modules. Does NOT import from process_pdbind.py, avoiding its heavy dependencies
(rdkit, xgboost, seaborn) that may not be available on HPC environments.

Processes a slice of the PDBbind complex list, running AutoDock Vina in parallel
with a fixed number of workers (each Vina call uses exactly 1 CPU, so --cpus N
maps directly to N simultaneous Vina processes on N CPU cores).

Output:
  - PDBQT files in output/ (protein, ligand, docked poses) — shared across all
    SLURM tasks because each task handles a non-overlapping set of complexes.
  - One result CSV per job: output/vina_batch_{start:06d}.csv
    (same column format as prepare_training_data(), ready for --load-csv)

Usage (local single run):
    python vina_docking.py --start 0 --count 100 --cpus 8

SLURM job array (20 jobs × 1000 complexes each):
    #SBATCH --array=0-19
    #SBATCH --cpus-per-task=8
    python vina_docking.py \\
        --start $((SLURM_ARRAY_TASK_ID * 1000)) \\
        --count 1000 \\
        --cpus $SLURM_CPUS_PER_TASK

Post-processing after all jobs complete:
    python - <<'EOF'
    import pandas as pd, glob
    dfs = [pd.read_csv(f) for f in sorted(glob.glob('output/vina_batch_*.csv'))]
    pd.concat(dfs, ignore_index=True).to_csv('output/training_data.csv', index=False)
    print(f"Combined {len(dfs)} batches into output/training_data.csv")
    EOF

    # CASF test set (run once separately)
    python vina_docking.py --casf --cpus 8

    # Feature extraction + ML (after all batches are combined)
    python process_pdbind.py \\
        --load-csv output/training_data.csv output/casf_results.csv
"""

import argparse
import math
import re
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration — mirror process_pdbind.py constants so paths match
# ---------------------------------------------------------------------------

INDEX_DIR       = Path("PDBind_2020_index")
STRUCTURES_DIR  = Path("PDBind_2020")
CASF_DIR        = Path("CASF-2016/coreset")
OUTPUT_DIR      = Path("output")
VINA_EXECUTABLE = Path("/Applications/autodock_vina_1_1_2_mac_catalina_64bit/bin/vina")

NUM_TOP_POSES = 5


# ---------------------------------------------------------------------------
# Binding affinity helpers
# ---------------------------------------------------------------------------

def parse_binding_affinity(binding_str: str) -> Optional[float]:
    """Parse binding affinity string to numeric value in nM."""
    try:
        match = re.search(r'(Kd|Ki|IC50)([<>=])([0-9.]+)([munpM]+)', binding_str)
        if not match:
            return None
        value = float(match.group(3))
        unit = match.group(4).lower()[0]
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
    """Load binding affinity data from PDBbind index file."""
    index_file = INDEX_DIR / "INDEX_general_PL.2020R1.lst"
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")

    records = []
    with open(index_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            pdb_code = parts[0]
            resolution = parts[1]
            year = int(parts[2])
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
                    'binding_data': binding_data,
                })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Complex list helpers
# ---------------------------------------------------------------------------

def load_casf_pdb_codes(casf_dir: Path = CASF_DIR) -> set:
    """Return the set of PDB codes in the CASF-2016 coreset directory."""
    if not casf_dir.exists():
        raise FileNotFoundError(f"CASF directory not found: {casf_dir}")
    return {d.name for d in casf_dir.iterdir() if d.is_dir()}


def get_complexes() -> List[str]:
    """Get sorted list of PDB codes that have both ligand and protein files (non-CASF)."""
    if not STRUCTURES_DIR.exists():
        raise FileNotFoundError(f"Structures directory not found: {STRUCTURES_DIR}")

    casf_codes = load_casf_pdb_codes() if CASF_DIR.exists() else set()
    valid_complexes = []

    for pdb_dir in STRUCTURES_DIR.rglob("*"):
        if not pdb_dir.is_dir():
            continue
        pdb_code = pdb_dir.name
        if pdb_code in casf_codes:
            continue
        ligand_file = pdb_dir / f"{pdb_code}_ligand.mol2"
        protein_file = pdb_dir / f"{pdb_code}_protein.pdb"
        if ligand_file.exists() and protein_file.exists():
            valid_complexes.append(pdb_code)

    valid_complexes.sort()
    return valid_complexes


def get_complex_path(pdb_code: str) -> Path:
    """Find the full path to a complex directory given its PDB code."""
    if CASF_DIR.exists():
        casf_path = CASF_DIR / pdb_code
        if casf_path.is_dir():
            return casf_path
    for pdb_dir in STRUCTURES_DIR.rglob(pdb_code):
        if pdb_dir.is_dir():
            return pdb_dir
    raise FileNotFoundError(f"Complex directory not found for {pdb_code}")


def get_complex_affinity_list(casf: bool = False) -> List[Tuple[str, float]]:
    """
    Return a sorted list of (pdb_code, affinity_nM) for all valid complexes
    that have binding affinity data in the PDBbind index.
    """
    binding_data = load_binding_data()
    affinity_lookup = binding_data.set_index('pdb_code')['affinity_nM'].to_dict()

    if casf:
        if not CASF_DIR.exists():
            raise FileNotFoundError(f"CASF directory not found: {CASF_DIR}")
        codes = sorted(load_casf_pdb_codes())
    else:
        codes = get_complexes()

    result = [(c, affinity_lookup[c]) for c in codes if c in affinity_lookup]
    skipped = len(codes) - len(result)
    if skipped:
        print(f"  Skipping {skipped} complexes with no binding affinity data")
    return result


# ---------------------------------------------------------------------------
# mol2 coordinate parser (replaces RDKit in define_binding_site and
# extract_crystal_ligand_coords)
# ---------------------------------------------------------------------------

def _parse_mol2_coords(mol2_path: str, heavy_only: bool = False) -> Optional[np.ndarray]:
    """
    Parse XYZ coordinates from the @<TRIPOS>ATOM section of a mol2 file.

    mol2 ATOM row format (space-delimited):
        atom_id  atom_name  x  y  z  atom_type  subst_id  subst_name  charge

    heavy_only: when True, skip rows where atom_type (column 5) starts with 'H'.
    """
    coords: List[List[float]] = []
    in_atom_block = False
    try:
        with open(mol2_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                if line.startswith('@<TRIPOS>ATOM'):
                    in_atom_block = True
                    continue
                if line.startswith('@<TRIPOS>'):
                    in_atom_block = False
                    continue
                if in_atom_block and line.strip():
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    if heavy_only and parts[5].startswith('H'):
                        continue
                    try:
                        coords.append([float(parts[2]), float(parts[3]), float(parts[4])])
                    except ValueError:
                        continue
    except (IOError, OSError):
        return None
    return np.array(coords) if coords else None


# ---------------------------------------------------------------------------
# Binding site and RMSD helpers
# ---------------------------------------------------------------------------

def define_binding_site(
    protein_pdb: str,
    ligand_mol2: str,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Define Vina search box centered on the crystal ligand (mol2 text parser)."""
    coords = _parse_mol2_coords(ligand_mol2)
    if coords is None:
        raise ValueError(f"Could not parse ligand: {ligand_mol2}")
    center = coords.mean(axis=0)
    size = np.maximum(coords.max(axis=0) - coords.min(axis=0) + 10, 20)
    return (float(center[0]), float(center[1]), float(center[2])), \
           (float(size[0]), float(size[1]), float(size[2]))


def extract_pose_coordinates(pdbqt_file: Path) -> List[np.ndarray]:
    """Extract heavy-atom coordinates for each pose from a Vina PDBQT output file."""
    poses: List[np.ndarray] = []
    current_pose_atoms: List[List[float]] = []

    with open(pdbqt_file, 'r') as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('MODEL'):
                current_pose_atoms = []
            elif line.startswith('ENDMDL'):
                if current_pose_atoms:
                    poses.append(np.array(current_pose_atoms))
                    current_pose_atoms = []
            elif line.startswith('ATOM') or line.startswith('HETATM'):
                atom_name = line[12:16].strip()
                if not atom_name.startswith('H'):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        current_pose_atoms.append([x, y, z])
                    except (ValueError, IndexError):
                        continue

    if current_pose_atoms:
        poses.append(np.array(current_pose_atoms))
    return poses


def extract_crystal_ligand_coords(pdb_code: str) -> Optional[np.ndarray]:
    """Extract heavy-atom coordinates from the crystal-structure ligand mol2 file."""
    try:
        pdb_dir = get_complex_path(pdb_code)
        ligand_mol2 = pdb_dir / f"{pdb_code}_ligand.mol2"
        if not ligand_mol2.exists():
            return None
        return _parse_mol2_coords(str(ligand_mol2), heavy_only=True)
    except Exception as e:
        print(f"Warning: Could not extract crystal coordinates for {pdb_code}: {e}")
        return None


def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Calculate RMSD between two coordinate arrays using the Kabsch algorithm."""
    if coords1.shape != coords2.shape:
        return float('inf')
    centroid1 = np.mean(coords1, axis=0)
    centroid2 = np.mean(coords2, axis=0)
    centered1 = coords1 - centroid1
    centered2 = coords2 - centroid2
    covariance_matrix = np.dot(centered2.T, centered1)
    V, S, Wt = np.linalg.svd(covariance_matrix)
    d = np.linalg.det(np.dot(V, Wt))
    if d < 0:
        V[:, -1] = -V[:, -1]
    rotation_matrix = np.dot(V, Wt)
    rotated_coords2 = np.dot(centered2, rotation_matrix)
    return float(np.sqrt(np.mean(np.sum((centered1 - rotated_coords2) ** 2, axis=1))))


def calculate_all_pose_rmsds(
    vina_result: Dict,
    pdb_code: str,
) -> Optional[List[float]]:
    """Calculate RMSD between each Vina pose and the crystal structure."""
    crystal_coords = extract_crystal_ligand_coords(pdb_code)
    if crystal_coords is None:
        return None
    pose_coords = vina_result.get('pose_coordinates', [])
    if not pose_coords:
        return None
    if crystal_coords.shape[0] != pose_coords[0].shape[0]:
        print(f"Warning: Atom count mismatch for {pdb_code}: "
              f"crystal={crystal_coords.shape[0]}, pose={pose_coords[0].shape[0]}")
        return None
    return [calculate_rmsd(crystal_coords, pose) for pose in pose_coords]


# ---------------------------------------------------------------------------
# Vina execution
# ---------------------------------------------------------------------------

def run_vina(
    pdb_code: str,
    exhaustiveness: int = 8,
    vina_executable: Optional[Path] = None,
    n_cpu: int = 0,
) -> Optional[Dict]:
    """
    Run AutoDock Vina on a single complex.

    n_cpu: number of CPU cores to pass to Vina (0 = let Vina auto-detect).
    Protein and ligand PDBQT files are skipped if they already exist and are
    non-empty, making restarts cheap.
    """
    exe = vina_executable if vina_executable is not None else VINA_EXECUTABLE
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

    try:
        center, size = define_binding_site(str(protein_pdb), str(ligand_mol2))
    except Exception as e:
        print(f"Warning: Could not define binding site for {pdb_code}: {e}")
        return None

    vina_input  = OUTPUT_DIR / f"{pdb_code}_vina_input.pdbqt"
    vina_ligand = OUTPUT_DIR / f"{pdb_code}_vina_ligand.pdbqt"
    vina_output = OUTPUT_DIR / f"{pdb_code}_vina_output.pdbqt"

    # Prepare receptor — skip if already converted
    if not vina_input.exists() or vina_input.stat().st_size == 0:
        proc = subprocess.run(
            ['obabel', '-ipdb', str(protein_pdb), '-opdbqt', '-O', str(vina_input),
             '-xr', '--partialcharge', 'gasteiger'],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"Warning: Failed to prepare receptor for {pdb_code}: {proc.stderr}")
            return None

    # Prepare ligand — skip if already converted
    if not vina_ligand.exists() or vina_ligand.stat().st_size == 0:
        proc = subprocess.run(
            ['obabel', '-imol2', str(ligand_mol2), '-opdbqt', '-O', str(vina_ligand),
             '--partialcharge', 'gasteiger'],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"Warning: Failed to prepare ligand for {pdb_code}: {proc.stderr}")
            return None

    vina_cmd = [
        str(exe),
        '--receptor', str(vina_input),
        '--ligand',   str(vina_ligand),
        '--center_x', str(center[0]),
        '--center_y', str(center[1]),
        '--center_z', str(center[2]),
        '--size_x',   str(size[0]),
        '--size_y',   str(size[1]),
        '--size_z',   str(size[2]),
        '--out',      str(vina_output),
        '--num_modes', str(NUM_TOP_POSES),
        '--exhaustiveness', str(exhaustiveness),
    ]
    if n_cpu > 0:
        vina_cmd += ['--cpu', str(n_cpu)]

    result = subprocess.run(vina_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: Vina failed for {pdb_code}: {result.stderr}")
        return None

    scores: List[float] = []
    with open(vina_output, 'r') as f:
        for line in f:
            if line.startswith('REMARK VINA RESULT:'):
                parts = line.split()
                if len(parts) >= 5:
                    scores.append(float(parts[3]))

    if not scores:
        print(f"Warning: No poses generated for {pdb_code}")
        return None

    pose_coords = extract_pose_coordinates(vina_output)
    if len(pose_coords) != len(scores):
        print(f"Warning: Mismatch between scores ({len(scores)}) and poses "
              f"({len(pose_coords)}) for {pdb_code}")
        return None

    return {
        'pdb_code': pdb_code,
        'scores': scores,
        'pose_coordinates': pose_coords,
        'top_score': scores[0],
        'center': center,
        'size': size,
    }


def _vina_worker(args: tuple) -> Optional[Dict]:
    """
    Top-level worker for parallel docking (must be module-level for pickling).

    args: (pdb_code, affinity_nM, exhaustiveness, vina_executable[, n_cpu])
    """
    if len(args) == 5:
        pdb_code, affinity_nM, exhaustiveness, vina_executable, n_cpu = args
    else:
        pdb_code, affinity_nM, exhaustiveness, vina_executable = args
        n_cpu = 0

    vina_result = run_vina(
        pdb_code,
        exhaustiveness=exhaustiveness,
        vina_executable=vina_executable,
        n_cpu=n_cpu,
    )
    if vina_result is None:
        return None

    rmsds = calculate_all_pose_rmsds(vina_result, pdb_code)
    if rmsds is None:
        print(f"  Could not calculate RMSDs for {pdb_code}, skipping")
        return None

    vina_result['rmsds'] = rmsds
    vina_result['affinity_nM'] = affinity_nM
    return vina_result


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _has_valid_output_pdbqt(pdb_code: str) -> bool:
    """Return True if a complete Vina output PDBQT already exists for pdb_code."""
    pdbqt = OUTPUT_DIR / f"{pdb_code}_vina_output.pdbqt"
    if not pdbqt.exists() or pdbqt.stat().st_size == 0:
        return False
    with open(pdbqt) as f:
        return any(line.startswith('REMARK VINA RESULT:') for line in f)


def _recover_worker(args: tuple) -> Optional[Dict]:
    """
    Reconstruct a result dict from an existing Vina output PDBQT (no Vina re-run).

    args: (pdb_code, affinity_nM)
    """
    pdb_code, affinity_nM = args
    vina_output = OUTPUT_DIR / f"{pdb_code}_vina_output.pdbqt"
    scores: List[float] = []
    try:
        with open(vina_output) as f:
            for line in f:
                if line.startswith('REMARK VINA RESULT:'):
                    parts = line.split()
                    if len(parts) >= 5:
                        scores.append(float(parts[3]))
    except (IOError, OSError):
        return None
    if not scores:
        return None
    pose_coords = extract_pose_coordinates(vina_output)
    if len(pose_coords) != len(scores):
        return None
    vina_result: Dict = {
        'pdb_code': pdb_code,
        'scores': scores,
        'pose_coordinates': pose_coords,
        'top_score': scores[0],
    }
    rmsds = calculate_all_pose_rmsds(vina_result, pdb_code)
    if rmsds is None:
        print(f"  Could not calculate RMSDs for {pdb_code} (recovery), skipping")
        return None
    vina_result['rmsds'] = rmsds
    vina_result['affinity_nM'] = affinity_nM
    return vina_result


# ---------------------------------------------------------------------------
# Feature extraction and DataFrame assembly
# ---------------------------------------------------------------------------

def extract_pose_features(vina_result: Dict, pose_idx: int, rmsd: float) -> Dict:
    """Extract per-pose Vina score statistics."""
    scores = vina_result['scores']
    num_poses = len(scores)
    pose_score = scores[pose_idx]
    score_rank = pose_idx + 1
    score_mean = float(np.mean(scores))
    score_std  = float(np.std(scores))
    return {
        'vina_score':            pose_score,
        'vina_rank':             score_rank,
        'vina_score_diff_best':  pose_score - scores[0],
        'vina_score_diff_worst': scores[-1] - pose_score,
        'vina_score_percentile': (num_poses - score_rank) / num_poses,
        'vina_score_zscore':     (pose_score - score_mean) / (score_std + 1e-8),
        'vina_num_poses':        num_poses,
        'vina_score_range':      scores[0] - scores[-1],
        'rmsd':                  rmsd,
    }


def prepare_training_data(vina_results: List[Dict]) -> pd.DataFrame:
    """
    Convert a list of _vina_worker result dicts to a per-pose DataFrame.

    Uses already-computed result['rmsds'] to avoid re-parsing mol2 files.
    """
    records = []
    for result in vina_results:
        pdb_code = result['pdb_code']
        scores = result['scores']
        rmsds = result.get('rmsds') or calculate_all_pose_rmsds(result, pdb_code)
        if rmsds is None:
            print(f"Warning: Could not calculate RMSDs for {pdb_code}, skipping")
            continue
        if len(rmsds) != len(scores):
            print(f"Warning: RMSD/score count mismatch for {pdb_code}, skipping")
            continue
        best_pose_idx = int(np.argmin(rmsds))
        for pose_idx in range(len(scores)):
            features = extract_pose_features(result, pose_idx, rmsds[pose_idx])
            record = {
                'pdb_code':    pdb_code,
                'pose_idx':    pose_idx,
                'is_best_pose': 1 if pose_idx == best_pose_idx else 0,
                'rmsd':        rmsds[pose_idx],
            }
            record.update(features)
            records.append(record)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Run AutoDock Vina on a slice of PDBbind complexes for HPC job arrays. '
            'Each Vina call uses 1 CPU; --cpus controls how many run simultaneously.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--start', type=int, default=0,
        help='Index into the sorted complex list to start from '
             '(set to SLURM_ARRAY_TASK_ID * STEP in job arrays)',
    )
    parser.add_argument(
        '--count', type=int, default=1000,
        help='Number of complexes this job should process',
    )
    parser.add_argument(
        '--cpus', type=int, default=8,
        help='Number of parallel Vina workers. Each worker uses exactly 1 CPU '
             '(--cpu 1 is passed to Vina), so this should match --cpus-per-task '
             'in your SLURM script.',
    )
    parser.add_argument(
        '--exhaustiveness', type=int, default=8,
        help='Vina exhaustiveness parameter (higher = more thorough but slower)',
    )
    parser.add_argument(
        '--vina-executable', type=str, default=None,
        help='Override path to the Vina binary (default: VINA_EXECUTABLE constant)',
    )
    parser.add_argument(
        '--casf', action='store_true',
        help='Run on the CASF-2016 test set instead of the training set. '
             'Output is written to output/casf_results.csv.',
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory for PDBQT files and result CSV (default: output/)',
    )
    args = parser.parse_args()

    # Override module-level OUTPUT_DIR if requested
    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    vina_exe = Path(args.vina_executable) if args.vina_executable else VINA_EXECUTABLE

    print("Loading complex list and binding affinity data...")
    try:
        all_complexes = get_complex_affinity_list(casf=args.casf)
    except FileNotFoundError as e:
        sys.exit(f"Error: {e}")

    total = len(all_complexes)
    label = "CASF-2016" if args.casf else "training"
    print(f"Total {label} complexes with affinity data: {total}")

    if args.casf:
        batch = all_complexes
        start = 0
    else:
        start = args.start
        end = min(start + args.count, total)
        batch = all_complexes[start:end]

    if not batch:
        print(f"No complexes in slice [{start}:{start + args.count}] "
              f"(total {total}). Nothing to do.")
        sys.exit(0)

    end_idx = start + len(batch)
    print(f"This job: complexes [{start}:{end_idx}] ({len(batch)} complexes)")

    # Determine output CSV path early — needed to check for existing work
    if args.casf:
        out_csv = out_dir / 'casf_results.csv'
    else:
        out_csv = out_dir / f'vina_batch_{start:06d}.csv'

    # --- Resume: load existing CSV rows and categorize remaining work ---
    existing_df = pd.DataFrame()
    existing_codes: set = set()
    if out_csv.exists():
        try:
            existing_df = pd.read_csv(out_csv)
            existing_codes = set(existing_df['pdb_code'].unique())
            print(f"  Resuming: found existing CSV with {len(existing_codes)} "
                  f"completed complex(es) — skipping those.")
        except Exception as exc:
            print(f"  Warning: could not read existing CSV ({exc}), will reprocess all.")

    to_run: List[Tuple[str, float]] = []      # no output PDBQT → run Vina
    to_recover: List[Tuple[str, float]] = []  # PDBQT exists but not in CSV → recover
    n_already_done = 0

    for pdb_code, affinity_nM in batch:
        if pdb_code in existing_codes:
            n_already_done += 1
        elif _has_valid_output_pdbqt(pdb_code):
            to_recover.append((pdb_code, affinity_nM))
        else:
            to_run.append((pdb_code, affinity_nM))

    print(f"  Already in CSV:        {n_already_done}")
    print(f"  Recover from PDBQT:    {len(to_recover)}")
    print(f"  Need Vina run:         {len(to_run)}")

    if not to_run and not to_recover:
        print("All complexes in this batch already processed. Nothing to do.")
        sys.exit(0)

    # --- Run Vina for new complexes ---
    new_results: List[Optional[Dict]] = []
    if to_run:
        worker_args = [
            (pdb_code, affinity_nM, args.exhaustiveness, vina_exe, 1)
            for pdb_code, affinity_nM in to_run
        ]
        print(f"\nRunning Vina with {args.cpus} parallel workers "
              f"({len(to_run)} complexes)...")
        t0 = time.time()
        if args.cpus == 1:
            for i, wa in enumerate(worker_args, 1):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                remaining = (len(worker_args) - i) / rate if rate > 0 else 0
                eta = f"{int(remaining // 60)}m {int(remaining % 60)}s" if rate > 0 else "--"
                print(f"  [{i}/{len(worker_args)}] {wa[0]}  ETA: {eta}")
                new_results.append(_vina_worker(wa))
        else:
            with Pool(processes=args.cpus) as pool:
                new_results = pool.map(_vina_worker, worker_args)
        elapsed = time.time() - t0
        n_success = sum(1 for r in new_results if r is not None)
        n_failed  = len(new_results) - n_success
        print(f"Vina complete in {elapsed:.1f}s: {n_success} succeeded, {n_failed} failed")

    # --- Recover results from existing PDBQT files ---
    recovered_results: List[Optional[Dict]] = []
    if to_recover:
        print(f"\nRecovering {len(to_recover)} complex(es) from existing PDBQT files...")
        t1 = time.time()
        if args.cpus == 1:
            for item in to_recover:
                recovered_results.append(_recover_worker(item))
        else:
            with Pool(processes=args.cpus) as pool:
                recovered_results = pool.map(_recover_worker, to_recover)
        n_rec = sum(1 for r in recovered_results if r is not None)
        print(f"Recovery complete in {time.time() - t1:.1f}s: "
              f"{n_rec}/{len(to_recover)} succeeded")

    # --- Convert new + recovered results to DataFrame, merge with existing, write ---
    all_new = [r for r in new_results + recovered_results if r is not None]
    if not all_new and existing_df.empty:
        print("No successful results and no existing data. Output CSV not written.")
        sys.exit(1)

    new_df = prepare_training_data(all_new) if all_new else pd.DataFrame()

    if not existing_df.empty and not new_df.empty:
        results_df = pd.concat([existing_df, new_df], ignore_index=True)
    elif not existing_df.empty:
        results_df = existing_df
    else:
        results_df = new_df

    results_df.to_csv(out_csv, index=False)
    n_poses = len(results_df)
    n_complexes = results_df['pdb_code'].nunique()
    print(f"Saved {n_poses} pose rows ({n_complexes} complexes) to {out_csv}")

    print("\nDone.")
    print(f"PDBQT files: {out_dir}/<pdb_code>_vina_{{input,ligand,output}}.pdbqt")
    if not args.casf:
        print(f"\nAfter all SLURM tasks finish, combine batches with:")
        print("  python - <<'EOF'")
        print("  import pandas as pd, glob")
        print("  dfs = [pd.read_csv(f) for f in sorted(glob.glob('output/vina_batch_*.csv'))]")
        print("  pd.concat(dfs, ignore_index=True).to_csv('output/training_data.csv', index=False)")
        print("  EOF")


if __name__ == '__main__':
    main()
