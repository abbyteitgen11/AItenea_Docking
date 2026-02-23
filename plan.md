# AItenea Docking — Development Plan

**Project:** ML rescoring of AutoDock Vina poses on PDBbind general set (5,300 complexes)
**Script:** `process_pdbind.py`
**Environment:** `docking_aitenea` (Python 3.13, XGBoost 3.2, sklearn 1.8, RDKit, scipy)
**Run command:**
```bash
/opt/anaconda3/envs/docking_aitenea/bin/python process_pdbind.py \
  --load-csv output/training_data.csv output/test_data.csv
```

---

## Overall Progress: ~45%

---

## a) General Work Plan

Goal: Train a machine learning model that selects the best docking pose (closest to the
crystal structure) better than AutoDock Vina's native ranking.

Baseline: Vina rank-1 is correct for **28.2%** of test complexes (1,060 total).
Current best: RF ranker with contact features **32.8%** best-pose selection rate.

Key metric: fraction of test complexes where the model's top-ranked pose has RMSD < 2 Å
to the crystal ligand (standard docking benchmark).

---

## b) Implementation by Stages

### Stage 1 — Basic XGBoost classifier [COMPLETE]
- Binary classification (is_best_pose = 0/1)
- 8 Vina-score-derived features only
- Result: XGB 0.263 < Vina 0.282 (worse than baseline)

### Stage 2 — Ranker models + feature engineering [COMPLETE]
- Switched to XGBoost `rank:pairwise` with relevance = 1/(1+rmsd)
- Added Random Forest and Gradient Boosting rankers
- Added PDBQT pose geometry features (ROG, COM drift, extent, volume)
- Added 13 RDKit molecular descriptors from mol2 files
- Added engineered features (Boltzmann probability, score gaps, log score)
- Added `--load-csv` flag for fast iteration without re-running Vina
- Result: RF 0.334, XGB 0.331, GB 0.333 vs Vina 0.282

### Stage 3 — Contact features + ensemble + extended diagnostics [COMPLETE]
- Added 7 protein-ligand contact features via KDTree (contact_n_3A/4A/5A,
  contact_min_dist, contact_score_lj, contact_buried_frac, contact_n_per_atom)
- Added ensemble ranker (min-max normalised average of 3 models)
- Added `--no-contact-features` flag for A/B comparison
- Extended metrics: Spearman correlation, multi-threshold rates (1–3 Å), median RMSD
- New plots: success_vs_threshold.png, rmsd_cdf.png, spearman_distribution.png
- Result with contacts: RF 0.328, ensemble 0.330 vs Vina 0.282

### Stage 4 — Hyperparameter optimisation [TODO]
- Grid/random search or Optuna for XGB, RF, GB hyperparameters
- Cross-validation on training set (avoid test set contamination)
- Expected gain: +1–3 pp

### Stage 5 — Richer contact features [TODO]
- Atom-type-specific contacts (aromatic–aromatic, H-bond donors/acceptors)
- SASA (solvent-accessible surface area) change on binding
- Pocket volume / shape descriptors
- Expected gain: +2–5 pp

### Stage 6 — Neural network / deep learning [TODO]
- Graph neural network on ligand + contact graph
- Or gradient-boosted trees on expanded feature set
- Expected gain: unknown, high variance

### Stage 7 — Analysis & reporting [TODO]
- Per-protein-family performance breakdown
- Failure mode analysis (what types of ligands fail most)
- Comparison with published rescoring benchmarks

---

## c) Checklist

### Infrastructure
- [x] AutoDock Vina docking pipeline (`run_vina`)
- [x] RMSD calculation (Kabsch algorithm, `calculate_rmsd`)
- [x] Feature matrix assembly (`prepare_training_data`)
- [x] Train/test split by complex (no data leakage)
- [x] `--load-csv` flag (fast model iteration)
- [x] `--no-augment` flag
- [x] `--no-contact-features` flag

### Features
- [x] Base Vina features (8): score, rank, diffs, z-score, num_poses, range
- [x] PDBQT geometry (4): ROG, COM drift, max extent, volume
- [x] RDKit mol2 descriptors (13): MW, logP, TPSA, HBD/A, rings, etc.
- [x] Engineered features (5): Boltzmann prob, score gaps, log score, score/atom
- [x] Protein–ligand contact features (7): KDTree-based, 3/4/5 Å shells
- [ ] Atom-type-specific contacts
- [ ] H-bond geometry features
- [ ] SASA / pocket volume

### Models
- [x] XGBoost ranker (`rank:pairwise`)
- [x] Random Forest regressor (1/(1+rmsd) target)
- [x] Gradient Boosting regressor
- [x] Ensemble (min-max normalised average)
- [ ] Hyperparameter optimisation
- [ ] Neural network / GNN

### Evaluation & Output
- [x] Best-pose selection rate
- [x] Multi-threshold success rates (1.0 / 1.5 / 2.0 / 2.5 / 3.0 Å)
- [x] Mean and median RMSD of selected poses
- [x] RMSD improvement vs Vina
- [x] Per-complex Spearman rank correlation
- [x] `output/score_comparison.csv` (per-pose, all models vs Vina)
- [x] `output/pose_scores_detailed.csv`
- [x] `output/pose_selection_metrics.txt`
- [ ] Per-protein-family breakdown

### Plots
- [x] RMSD distribution comparison (histogram + boxplot)
- [x] Success rate by Vina rank
- [x] Feature importance
- [x] RMSD vs predicted score scatter
- [x] Success rate vs RMSD threshold (all models)
- [x] RMSD CDF (all models)
- [x] Spearman distribution (all models)
- [ ] Learning curves (train/test performance vs n_complexes)
- [ ] Calibration plot

---

## d) Progress Percentage

| Stage | Status | Progress |
|-------|--------|----------|
| 1 — Basic classifier | Complete | 100% |
| 2 — Ranker + feature engineering | Complete | 100% |
| 3 — Contacts + ensemble + diagnostics | Complete | 100% |
| 4 — Hyperparameter optimisation | Not started | 0% |
| 5 — Richer contact features | Not started | 0% |
| 6 — Neural network | Not started | 0% |
| 7 — Analysis & reporting | Not started | 0% |
| **Overall** | | **~45%** |

---

## e) Next Actions (edit this section)

Priority order — edit/reorder as needed:

1. **Hyperparameter search for RF ranker** (highest ROI, ~1 day)
   - Use `RandomizedSearchCV` or Optuna on training set
   - Tune: n_estimators, max_depth, min_samples_leaf, max_features

2. **Atom-type contact features** (~0.5 days)
   - Separate hydrophobic / polar / aromatic contact counts
   - Use atom element from PDB records

3. **H-bond geometry features** (~1 day)
   - Identify donor-acceptor pairs within 3.5 Å with correct angle
   - Requires element assignment from protein PDB

4. **Learning curves** (~0.5 days)
   - Plot best-pose rate vs training set size (100 → 4,240 complexes)
   - Diagnoses whether more data or better features will help more

5. **Per-family performance breakdown** (~0.5 days)
   - Group test complexes by protein family (kinase, GPCR, protease, etc.)
   - Identify where model gains / loses vs Vina

6. **[YOUR NEXT IDEA HERE]**

---

*Last updated: 2026-02-23*
