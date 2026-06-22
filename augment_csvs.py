"""
augment_csvs.py — add protein-ligand contact features to the split CSVs.

The DimeNet++ "with protein features" model (gnn_dimenet_affinity.py) reads the 12
contact_* columns directly from its input CSVs. The base split CSVs produced by
split_data.py contain only the Vina-score columns, so this script adds the contact
features (reusing augment_with_contact_features from process_pdbind, which reads the
already-computed Vina pose PDBQT files in output/ and the protein PDBs — no re-docking)
and writes *_aug.csv variants for DimeNet to load.

Affinity labels and 3D ligand geometry are NOT added here — gnn_dimenet_affinity.py
joins affinity itself and reads coordinates from the mol2 files directly.

Usage:
    python augment_csvs.py
    python augment_csvs.py --splits training_data test_data        # subset
"""

import argparse
from pathlib import Path

import pandas as pd

from process_pdbind import augment_with_contact_features, OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--splits', nargs='+',
        default=['training_data', 'val_data', 'test_data'],
        help='Base CSV stems in output/ to augment (default: train/val/test).',
    )
    parser.add_argument(
        '--output-dir', type=str, default=str(OUTPUT_DIR),
        help=f'Directory holding the CSVs (default: {OUTPUT_DIR}).',
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)

    for name in args.splits:
        src = out_dir / f"{name}.csv"
        if not src.exists():
            print(f"  Skipping {name}: {src} not found")
            continue
        df = pd.read_csv(src)
        print(f"Augmenting {name} ({df.shape[0]} rows, "
              f"{df['pdb_code'].nunique()} complexes) with contact features...")
        aug = augment_with_contact_features(df)
        dst = out_dir / f"{name}_aug.csv"
        aug.to_csv(dst, index=False)
        n_contact = sum(1 for c in aug.columns if c.startswith('contact_'))
        print(f"  -> {dst}  shape={aug.shape}  ({n_contact} contact_* columns)")


if __name__ == "__main__":
    main()
