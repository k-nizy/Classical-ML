Formative 1 Part 2: Classification
==================================

Name:            Kevin Nizeyimana
Kaggle username: k-nizy
W&B project:     https://wandb.ai/nizykevin98-qeva/formative1-part2-classification

Final Kaggle leaderboard submissions
------------------------------------
1. submission_blend.csv - OOF-optimal blend (lgbm_c10_s43 + lgbm_c20_s42, w=0.51), OOF ROC-AUC 0.85310
2. submission_lgbm.csv  - best single 10-fold model (lgbm_c20_s42), OOF ROC-AUC 0.84896

How to run
----------
1. Open Formative1Part2_Classification.ipynb in Jupyter.
2. train.csv, test.csv and sample_submission.csv are included in data/.
3. Run all cells top to bottom. Requires: pandas, numpy, scikit-learn, lightgbm,
   xgboost, scipy, matplotlib, wandb. Runtime is roughly 50 minutes on 8 CPU cores.
4. W&B logging turns on when WANDB_API_KEY is set (environment variable, or a
   Colab/Kaggle secret). Without a key the notebook still runs end to end;
   the runs just are not uploaded.

Contents
--------
- Formative1Part2_Classification.ipynb (executed, with all outputs)
- README.txt (this file)
- data/ (train.csv, test.csv, sample_submission.csv)
- submissions/ (the two final CSVs listed above)
- results/tune_results.csv (hyperparameter search results referenced in Section 6)

The notebook was executed end to end without errors. Both submission files are
validated inside the notebook against sample_submission.csv (row count,
probability range, exact id order).
