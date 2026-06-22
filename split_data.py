"""
split_data.py — Build train / val / test CSVs from the combined docking pool.

Splits the full non-CASF PDBbind pool (output/pool_data.csv) into train (90%) and
validation (10%) BY COMPLEX (pdb_code), so no complex's poses straddle the split —
identical logic and seed to process_pdbind.split_val_from_train(). The CASF-2016
docking results (output/casf_results.csv) are used as the fixed test set.

Outputs (same 12-column format as the batch CSVs):
    output/training_data.csv  — train split only (overwrites the combined pool)
    output/val_data.csv       — validation split
    output/test_data.csv      — CASF-2016 test set

The full combined pool is preserved as output/pool_data.csv, so this script is
re-runnable and reproduces the same split every time.

Usage:
    python split_data.py
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42          # matches process_pdbind.RANDOM_SEED
VAL_FRAC = 0.1            # matches process_pdbind --val-frac default

OUTPUT_DIR = Path("output")
POOL_CSV = OUTPUT_DIR / "pool_data.csv"
CASF_CSV = OUTPUT_DIR / "casf_results.csv"


def main() -> None:
    if not POOL_CSV.exists():
        raise FileNotFoundError(
            f"{POOL_CSV} not found. Copy the combined non-CASF CSV there first:\n"
            f"    cp output/training_data.csv {POOL_CSV}"
        )
    if not CASF_CSV.exists():
        raise FileNotFoundError(
            f"{CASF_CSV} not found. Dock the CASF test set first:\n"
            f"    python vina_docking.py --casf --cpus 8"
        )

    pool = pd.read_csv(POOL_CSV)
    test_df = pd.read_csv(CASF_CSV)

    # Split by complex (pdb_code), not by individual pose, to avoid leakage.
    codes = pool["pdb_code"].unique()
    train_codes, val_codes = train_test_split(
        codes, test_size=VAL_FRAC, random_state=RANDOM_SEED
    )
    train_set, val_set, casf_set = set(train_codes), set(val_codes), set(test_df["pdb_code"])
    train_df = pool[pool["pdb_code"].isin(train_set)].copy()
    val_df = pool[pool["pdb_code"].isin(val_set)].copy()

    # Sanity checks — fail loudly rather than write a leaky split.
    assert not (train_set & val_set), "train/val complex overlap"
    assert not (train_set & casf_set), "CASF leaked into train"
    assert not (val_set & casf_set), "CASF leaked into val"
    assert train_set | val_set == set(codes), "train+val do not cover the pool"

    train_df.to_csv(OUTPUT_DIR / "training_data.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val_data.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_data.csv", index=False)

    print("Split complete (split by complex, seed "
          f"{RANDOM_SEED}, val_frac {VAL_FRAC}):")
    print(f"  Train: {train_df['pdb_code'].nunique():>6} complexes / "
          f"{len(train_df):>7} poses  -> output/training_data.csv")
    print(f"  Val  : {val_df['pdb_code'].nunique():>6} complexes / "
          f"{len(val_df):>7} poses  -> output/val_data.csv")
    print(f"  Test : {test_df['pdb_code'].nunique():>6} complexes / "
          f"{len(test_df):>7} poses  -> output/test_data.csv")


if __name__ == "__main__":
    main()
