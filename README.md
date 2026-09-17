# Classical ML — Kaggle Classification Competition

Course competition (Formative 1, Part 2): predict the probability that `target = 1`
for each row in the test set. Evaluated on **ROC AUC**, with the final grade based on
the **private** leaderboard.

**Final result: 0.85310 out-of-fold AUC** (see [METHODOLOGY.md](METHODOLOGY.md)).

## Approach

The dataset is synthetic tabular data designed to reward solid ML practice:
mixed numeric/categorical features, missing values, class imbalance (26% positive),
duplicated and noisy columns, and signal that lives in feature *interactions*
(single-feature correlations with the target are all tiny, max |r| ≈ 0.06).

Pipeline (scripts run in numbered order):

1. **EDA** — target balance, missingness, duplicate columns, correlations,
   categorical level rates (`src/eda_report.py`).
2. **Linear baselines** — dummy classifier (accuracy trap) + logistic regression
   in an impute→scale→one-hot pipeline (`src/pipeline_0_baseline.py`).
3. **Gradient-boosting baselines** — sklearn `HistGradientBoostingClassifier` and
   LightGBM, 5-fold stratified CV with out-of-fold (OOF) AUC
   (`src/pipeline_a_hgb.py`, `src/pipeline_b_lgbm.py`).
4. **Feature lab** — ablation over feature variants (row stats, interactions of
   top features, categorical combos, noise-column drop), OOF-scored
   (`src/features.py`, `src/lab_features.py`).
5. **Tuning** — randomized hyperparameter search for LightGBM, 3-fold CV per
   config, checkpointed to CSV so runs are resumable (`src/tune_lgbm.py`).
6. **Final ensemble** — 10-fold CV; 3 LightGBM jobs (2 tuned configs × seeds) +
   2 XGBoost jobs + HGB, checkpointed per job; blend chosen on OOF only
   (`src/final_ensemble.py`, `src/blend_utils.py`).
7. **Submissions** — rebuilt from checkpoints, validated against
   `sample_submission.csv` (`src/make_submissions.py`).

## Results (out-of-fold AUC, stratified CV)

| Step                                        | OOF AUC |
|---------------------------------------------|---------|
| DummyClassifier (majority) — accuracy 0.738 | —       |
| LogisticRegression                          | 0.6517  |
| HistGradientBoosting (baseline)             | 0.8215  |
| LightGBM (baseline)                         | 0.8287  |
| LightGBM (tuned, 5-fold)                    | 0.8430  |
| LightGBM tuned + interaction features, 10-fold | 0.8490 |
| **Final blend (2-model OOF-weighted pair)** | **0.8531** |

Key data findings:

- `num_feat_6 ≡ num_feat_8`, `num_feat_11 ≡ num_feat_18`, `num_feat_12 ≡ num_feat_34`
  (exact duplicates — one of each pair dropped).
- `cat_channel` is the strongest categorical signal (positive rate: A 37% vs C 15%);
  `cat_region` and `cat_tier` carry essentially no signal (kept anyway — an ablation
  showed dropping them slightly *hurts*; the trees simply ignore them).
- Pairwise products/diffs of the top-6 numeric features added **+0.012 AUC**,
  confirming the brief's "signal is in interactions" hint.
- Missing values are left native for the tree models rather than imputed.
- All model/blend selection used OOF AUC only — public-leaderboard feedback was
  deliberately not used for tuning, to protect the private score.

## Repository layout

```
data/           competition files (train/test gitignored; sample_submission committed)
src/            pipeline scripts, run in numbered order
results/        run logs, tuning results, checkpointed OOF/test predictions (.npy)
submissions/    final prediction files
```

## Reproducing

```bash
pip install -U scikit-learn lightgbm xgboost pandas numpy scipy

python src/eda_report.py            # exploratory analysis
python src/pipeline_0_baseline.py   # dummy + logistic regression baselines
python src/pipeline_a_hgb.py        # HGB baseline
python src/pipeline_b_lgbm.py       # LGBM baseline
python src/lab_features.py          # feature engineering ablation (~10 min)
python src/tune_lgbm.py             # randomized search (resumable)
python src/final_ensemble.py        # final ensemble (~25 min, resumable)
python src/make_submissions.py      # rebuild + validate submissions
```
