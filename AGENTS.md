# AGENTS.md

This document provides guidelines for working on the PDBind rescoring project.

## Setup and Installation

### Environment
- Python 3.13.5
- Conda environment: `docking_aitenea`

### Install Dependencies
```bash
conda activate docking_aitenea
pip install numpy pandas rdkit xgboost scikit-learn openbabel
# Install AutoDock Vina separately from: https://vina.scripps.edu/download/
```

## Running the Code

### Main Processing Script
```bash
python process_pdbind.py
```

### Quick Test (single complex)
```bash
python -c "
from process_pdbind import load_binding_data, get_complexes
complexes = get_complexes()[:1]
print(f'Test complex: {complexes[0]}')
"
```

### Verify Installation
```bash
python -c "
import numpy, pandas, rdkit, xgboost
print(f'numpy: {numpy.__version__}')
print(f'pandas: {pandas.__version__}')
print(f'rdkit: {rdkit.__version__}')
print(f'xgboost: {xgboost.__version__}')
"
```

## Code Style Guidelines

### Imports
- Standard library first, then third-party, then local
- Group imports by type, alphabetize within groups
- Use explicit imports (no `from X import *`)

```python
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import xgboost as xgb
```

### Formatting
- Line length: 100 characters
- Use 4 spaces for indentation
- Blank lines: 2 between function definitions, 1 within functions
- Spaces around operators: `x = a + b`, not `x=a+b`

### Types
- Use type hints for function signatures
- Simple types: `int`, `float`, `str`, `bool`
- Collections: `List[str]`, `Dict[str, int]`, `Tuple[float, float]`
- Optional: `Optional[str]` instead of `str | None`
- No type hints needed for obvious local variables

### Naming Conventions
- Variables/functions: `snake_case` (e.g., `binding_affinity`, `run_vina`)
- Classes: `CamelCase` (e.g., `ProteinLigandComplex`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_PDB_CODE_LEN`)
- Private attributes: `_private_variable` (single underscore)
- Single-letter names: `i`, `j`, `x`, `y` only in loops/math

### Error Handling
- Use `try/except` for operations that can fail
- Catch specific exceptions: `except FileNotFoundError:`
- Add helpful error messages: `except FileNotFoundError: print("File not found: ...")`
- Let errors propagate for truly unexpected issues
- Validate inputs early with clear error messages

```python
def load_ligand(path: str) -> Optional[Chem.Mol]:
    if not os.path.exists(path):
        print(f"Error: Ligand file not found: {path}")
        return None
    try:
        mol = Chem.MolFromMol2File(path)
        if mol is None:
            print(f"Could not parse ligand: {path}")
        return mol
    except Exception as e:
        print(f"Error loading ligand {path}: {e}")
        return None
```

### Project Structure
- Keep main logic in `process_pdbind.py`
- Helper functions at the top level (no class wrapping needed)
- Simple workflow: data loading → Vina docking → feature extraction → ML training

### Data Paths
- Index directory: `index/` - binding affinity data
- Structures directory: `P-L/` - protein and ligand files
- Complex naming: `{PDB_CODE}_ligand.mol2`, `{PDB_CODE}_protein.pdb`

### Key Functions
- `load_binding_data()` - Read index file, parse binding constants
- `get_complexes()` - Get list of valid PDB codes with structure files
- `run_vina()` - Execute AutoDock Vina docking
- `extract_rdkit_features()` - Calculate molecular descriptors
- `train_rescoring_model()` - Train XGBoost regressor

### Iteration Tips
- Run `python process_pdbind.py 2>&1 | head -50` to see output
- Use `print(f"Debug: x={x}")` for troubleshooting
- Process first 5-10 complexes for quick testing
- Comment your changes with dates and notes
