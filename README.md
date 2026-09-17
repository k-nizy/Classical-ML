# Classical ML — Kaggle Classification Competition

Course competition (Formative 1, Part 2): predict the probability that `target = 1`
for each row in the test set. Evaluated on **ROC AUC**, with the final grade based on
the **private** leaderboard.

## Approach

The dataset is synthetic tabular data designed to reward solid ML practice:
mixed numeric/categorical features, missing values, class imbalance (26% positive),
duplicated and noisy columns, and signal that lives in feature *interactions*
(single-feature correlations with the target are all tiny, max |r| ≈ 0.06).

Pipeline:

1. **EDA** — target balance, missingness, duplicate columns, feature/target and
   feature/feature correlations, categorical level rates (`src/eda_report.py`).
2. **Baselines** — sklearn `HistGradientBoostingClassifier` and LightGBM with
   5-fold stratified CV, out-of-fold (OOF) AUC as the honest estimate
   (`src/pipeline_a_hgb.py`, `src/pipeline_b_lgbm.py`).
3. **Tuning** — randomized hyperparameter search for LightGBM, 3-fold CV per
   config, checkpointed to CSV so runs are resumable (`src/tune_lgbm.py`).
4. **Final model** — two best LightGBM configs × 2 seeds, averaged; blended with
   HGB using an OOF-optimal weight (`src/final_model.py`).
5. **Submissions** — formatted against `sample_submission.csv` with sanity checks
   (`src/make_submissions.py`).

## Results (out-of-fold AUC, 5-fold stratified CV)

| Model                                   | OOF AUC |
|-----------------------------------------|---------|
| HistGradientBoosting (baseline)         | 0.8215  |
| LightGBM (pre-tuning)                   | 0.8287  |
| LightGBM (tuned, 2 configs × 2 seeds)   | **0.8430** |
| Blend (93.5% LGBM + 6.5% HGB)           | **0.8435** |

Key data findings:

- `num_feat_6 ≡ num_feat_8`, `num_feat_11 ≡ num_feat_18`, `num_feat_12 ≡ num_feat_34`
  (exact duplicates — one of each pair dropped).
- `cat_channel` is the strongest categorical signal (positive rate: A 37% vs C 15%);
  `cat_region` and `cat_tier` carry essentially no signal (confirmed by gain importance).
- Missing values are left native for the tree models rather than imputed.
- Public-leaderboard score is treated as noisy feedback; model selection was done
  on OOF AUC only, to protect the private-leaderboard score.

## Repository layout

```
data/           competition files (train/test/sample_submission, gitignored)
src/            pipeline scripts, run in numbered order
results/        run logs, tuning results, OOF metrics
submissions/    final prediction files
```

## Reproducing

```bash
pip install -U scikit-learn lightgbm pandas numpy

python src/eda_report.py          # exploratory analysis
python src/pipeline_a_hgb.py      # HGB baseline
python src/pipeline_b_lgbm.py     # LGBM baseline
python src/tune_lgbm.py           # randomized search (resumable)
python src/final_model.py         # final ensemble (~6 min)
python src/make_submissions.py    # writes submissions/*.csv
```
