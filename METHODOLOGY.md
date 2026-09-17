# Methodology — Formative 1 Part 2 (Classification)

## 1. Task

Binary classification, 30,000 labeled rows, 37 numeric + 3 categorical features.
Metric: **ROC AUC** on predicted probabilities; grade set by the **private**
leaderboard. Every design decision below optimizes honest generalization, not
the public score.

## 2. Data understanding (EDA)

Findings that drove the design:

| Finding | Evidence | Decision |
|---|---|---|
| Class imbalance 26.2% positive | `target` value counts | Optimize/threshold on AUC; never report raw accuracy alone |
| 3 exact duplicate columns | element-wise equality incl. NaN | Drop `num_feat_8, 18, 34` (keep `6, 11, 12`) |
| Missing values ~5–12% in 4 numerics + `cat_region` | `isna().sum()` | Let tree models handle NaN natively (no imputation distortion) |
| `cat_channel` is predictive | positive rate A=37%, B=26%, C=15% | Keep; one-hot/native categorical handling |
| `cat_region`, `cat_tier` look like noise | flat positive rates; tiny gain importance | Kept anyway — ablation (§5) showed dropping them hurts slightly |
| All single-feature correlations tiny (max \|r\|=0.059) | corrwith target | Signal must be interactional → feature engineering (§5) |
| Train/test distributions match | per-column mean/std comparison | No adversarial shift; simple CV is trustworthy |

The accuracy trap, demonstrated: a majority-class `DummyClassifier` scores
**73.8% accuracy**; logistic regression scores **73.9%** — nearly identical —
yet their AUCs are ~0.50 vs **0.652**. This is why the competition's metric
(ranking quality), not accuracy, is the right target.

## 3. Models and validation

All model comparisons use **stratified k-fold out-of-fold (OOF) AUC** —
every training row gets an unbiased prediction from a model that never saw it.

- **Baselines:** logistic regression (impute→scale→one-hot pipeline) = 0.652;
  HistGradientBoosting = 0.8215; LightGBM = 0.8287 (5-fold, seed 42).
- **Tuning:** 24-config randomized search over LightGBM
  (num_leaves, min_data_in_leaf, feature/bagging fractions, L1/L2, max_bin,
  min_gain_to_split), 3-fold CV per config, checkpointed to
  `results/tune_results.csv`. Winning region: many leaves (191–255), small
  leaves-size, light regularization.
- **Final:** 10-fold stratified CV. Three LightGBM jobs (2 tuned configs ×
  seeds), two XGBoost jobs, one HGB job — all checkpointed per job to
  `results/*.npy` (OOF + test predictions), so the expensive step runs once.

| Final single models (10-fold, interact features) | OOF AUC |
|---|---|
| lgbm_c20_s42 | 0.84896 |
| lgbm_c10_s42 | 0.84885 |
| lgbm_c10_s43 | 0.84647 |
| xgb_d6_s43   | 0.84119 |
| xgb_d6_s42   | 0.84077 |
| hgb_s42      | 0.83608 |

## 4. Feature engineering (the decisive step)

The brief states the signal is interactional. An ablation lab
(`src/lab_features.py`, `results/results_featlab.txt`) tested variants under
identical LGBM settings:

| Variant | OOF AUC | Δ vs base |
|---|---|---|
| base (37 features) | 0.83215 | — |
| + row statistics (mean/std/extreme/NaN count) | 0.83113 | −0.0010 |
| **+ products & diffs of top-6 numeric features** | **0.84425** | **+0.0121** |
| + channel×tier combo categorical | 0.83461 | +0.0025 |
| all of the above | 0.84198 | +0.0098 |
| interact minus the 2 noise categoricals | 0.84240 | +0.0103 |

Decisions: adopt the `interact` variant (+0.012 AUC); **keep** the noise
categoricals (dropping them cost 0.002 — gradient boosting simply ignores
useless splits, and removing columns is not free); reject row-stat features.

## 5. Ensembling

Candidates were combined three ways, scored on OOF only: best single, best
pair (weight grid), NNLS over all six (non-negative least squares on
probabilities). The **pair `lgbm_c10_s43 + lgbm_c20_s42` at w=0.51** won with
**OOF AUC 0.85310** — the two strongest, most diverse LGBM models. NNLS did
not beat the pair (adding weaker models diluted it), and parsimony protects
the private leaderboard.

## 6. Public/private leaderboard policy

- Model selection, blend weights, and stopping decisions used **OOF AUC only**.
- The public score is treated as a noisy 50% sample of the test set; no
  submission was chosen or rejected because of it.
- Two diverse final submissions are provided: `submission_blend.csv`
  (primary, OOF 0.8531) and `submission_lgbm.csv` (LGBM pair, OOF 0.8531 —
  same selection; see `results/run_subs2.log` for the audit trail).

## 7. Reproducibility

- Fixed seeds everywhere (42/43); stratified CV; no random holdout.
- Every expensive stage is checkpointed (resumable scripts).
- Full run order in [README.md](README.md); all intermediate artifacts under
  `results/` with raw logs.

## 8. Limitations / what I'd try next

- XGBoost/HGB underperformed LGBM here — a deeper XGB search or Dart mode
  might make the 3-model blend competitive; NNLS suggested the marginal
  models aren't pulling weight.
- A second-order search around the winning LGBM region (learning-rate
  annealing, interaction constraints) could add a little more.
- Probability calibration is unnecessary for AUC but would matter for
  thresholded deployment.
