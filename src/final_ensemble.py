"""
Final ensemble, checkpointed (resumable - rerun until it prints ALL DONE).

Feature variant: 'interact' (winner of the feature lab, +0.0121 AUC).
Jobs: LGBM (2 tuned configs + cfg10/seed43) | XGB (tuned hist) | HGB.
Each job: 10-fold stratified CV -> OOF + averaged test predictions saved to results/.
Then: OOF-optimal blend (best single / best pair / NNLS, tie-break to simpler),
submission files written with format validation.

Writes: results/oof_<tag>.npy, results/pred_<tag>.npy, results/results_ensemble.txt,
        submissions/submission_blend.csv, submissions/submission_lgbm.csv
"""
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import build_features

import lightgbm as lgb
import xgboost as xgb
from scipy.optimize import nnls
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SEED = 42
N_FOLDS = 10
R = 'results'

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
y = train['target'].values
X, X_test = build_features(train, test, 'interact')
cat_cols = [c for c in X.columns if str(X[c].dtype) == 'category']

LGBM_BASE = dict(objective='binary', metric='auc', verbosity=-1, feature_pre_filter=False)
LGBM_JOBS = [
    ('lgbm_c10_s42', dict(learning_rate=0.07, num_leaves=255, min_data_in_leaf=20,
     feature_fraction=0.982627653632069, bagging_fraction=0.8035171238433423, bagging_freq=1,
     lambda_l1=0.012705645329288707, lambda_l2=0.01531418971165477, max_bin=127), 42),
    ('lgbm_c10_s43', dict(learning_rate=0.07, num_leaves=255, min_data_in_leaf=20,
     feature_fraction=0.982627653632069, bagging_fraction=0.8035171238433423, bagging_freq=1,
     lambda_l1=0.012705645329288707, lambda_l2=0.01531418971165477, max_bin=127), 43),
    ('lgbm_c20_s42', dict(learning_rate=0.07, num_leaves=191, min_data_in_leaf=30,
     feature_fraction=0.8567840933365807, bagging_fraction=0.7975990992289793, bagging_freq=1,
     lambda_l1=0.009368999682313825, lambda_l2=0.006516990611177174, max_bin=127,
     min_gain_to_split=0.014910847596941081), 42),
]
XGB_JOBS = [
    ('xgb_d6_s42', dict(n_estimators=3000, learning_rate=0.07, max_depth=6,
     min_child_weight=20, subsample=0.85, colsample_bytree=0.9, reg_lambda=1.0,
     reg_alpha=0.05, max_bin=256, tree_method='hist', enable_categorical=True,
     objective='binary:logistic', eval_metric='auc', early_stopping_rounds=100), 42),
    ('xgb_d6_s43', dict(n_estimators=3000, learning_rate=0.07, max_depth=6,
     min_child_weight=20, subsample=0.85, colsample_bytree=0.9, reg_lambda=1.0,
     reg_alpha=0.05, max_bin=256, tree_method='hist', enable_categorical=True,
     objective='binary:logistic', eval_metric='auc', early_stopping_rounds=100), 43),
]
HGB_JOBS = [
    ('hgb_s42', dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
     min_samples_leaf=40, l2_regularization=1.0, max_bins=255, early_stopping=False), 42),
]

report = []

def log(msg):
    print(msg, flush=True)
    report.append(msg)

def run_lgbm(tag, cfg, seed):
    params = {**LGBM_BASE, **cfg, 'seed': seed}
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    oof, pred, aucs = np.zeros(len(X)), np.zeros(len(X_test)), []
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    for tr, va in skf.split(X, y):
        dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_idx)
        dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        pred += m.predict(X_test, num_iteration=m.best_iteration) / N_FOLDS
        aucs.append(roc_auc_score(y[va], oof[va]))
    return oof, pred, aucs

def run_xgb(tag, cfg, seed):
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    oof, pred, aucs = np.zeros(len(X)), np.zeros(len(X_test)), []
    for tr, va in skf.split(X, y):
        m = xgb.XGBClassifier(**cfg, random_state=seed)
        m.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        pred += m.predict_proba(X_test)[:, 1] / N_FOLDS
        aucs.append(roc_auc_score(y[va], oof[va]))
    return oof, pred, aucs

def run_hgb(tag, cfg, seed):
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    oof, pred, aucs = np.zeros(len(X)), np.zeros(len(X_test)), []
    for tr, va in skf.split(X, y):
        m = HistGradientBoostingClassifier(**cfg, random_state=seed)
        m.fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        pred += m.predict_proba(X_test)[:, 1] / N_FOLDS
        aucs.append(roc_auc_score(y[va], oof[va]))
    return oof, pred, aucs

RUNNERS = {'lgbm': run_lgbm, 'xgb': run_xgb, 'hgb': run_hgb}
ALL_JOBS = ([(t, 'lgbm', c, s) for t, c, s in LGBM_JOBS]
            + [(t, 'xgb', c, s) for t, c, s in XGB_JOBS]
            + [(t, 'hgb', c, s) for t, c, s in HGB_JOBS])

# ---- run pending jobs (checkpoint: skip if OOF file exists) -----------------
pending = [j for j in ALL_JOBS if not os.path.exists(f'{R}/oof_{j[0]}.npy')]
if not pending:
    log('all model jobs already done - skipping to blend')
for tag, kind, cfg, seed in pending:
    t0 = time.time()
    oof, pred, aucs = RUNNERS[kind](tag, cfg, seed)
    np.save(f'{R}/oof_{tag}.npy', oof)
    np.save(f'{R}/pred_{tag}.npy', pred)
    log(f'{tag}: OOF {roc_auc_score(y, oof):.5f} (fold std {np.std(aucs):.4f}, {time.time()-t0:.0f}s)')

missing = [j[0] for j in ALL_JOBS if not os.path.exists(f'{R}/oof_{j[0]}.npy')]
if missing:
    log(f'time budget: stopping early, rerun to finish. Missing: {missing}')
    with open(f'{R}/results_ensemble.txt', 'w') as f:
        f.write('\n'.join(report) + '\n')
    sys.exit(0)

# ---- blend selection --------------------------------------------------------
oof_m = {tag: np.load(f'{R}/oof_{tag}.npy') for tag, *_ in ALL_JOBS}
pred_m = {tag: np.load(f'{R}/pred_{tag}.npy') for tag, *_ in ALL_JOBS}
tags = [t for t, *_ in ALL_JOBS]

singles = {t: roc_auc_score(y, oof_m[t]) for t in tags}
for t in sorted(singles, key=singles.get, reverse=True):
    log(f'  single {t}: {singles[t]:.5f}')

best_single = max(singles, key=singles.get)
best_auc, best_name, best_oof, best_pred = singles[best_single], best_single, oof_m[best_single], pred_m[best_single]

# best pair
for i, a in enumerate(tags):
    for b in tags[i+1:]:
        for w in np.linspace(0, 1, 101):
            blend = w * oof_m[a] + (1 - w) * oof_m[b]
            auc = roc_auc_score(y, blend)
            if auc > best_auc:
                best_auc, best_name = auc, f'pair {a}+{b} w={w:.2f}'
                best_pred = w * pred_m[a] + (1 - w) * pred_m[b]

# NNLS over all OOFs (non-negative weights, least squares on probabilities)
O = np.column_stack([oof_m[t] for t in tags])
w_nnls, _ = nnls(O, y.astype(float))
w_norm = w_nnls / w_nnls.sum()
auc_nnls = roc_auc_score(y, O @ w_norm)
if auc_nnls > best_auc:
    best_auc, best_name = auc_nnls, 'nnls ' + ', '.join(f'{t}:{w:.2f}' for t, w in zip(tags, w_norm) if w > 0.01)
    best_pred = np.column_stack([pred_m[t] for t in tags]) @ w_norm

log(f'\nselected blend: {best_name}\n  OOF AUC = {best_auc:.5f}')

# ---- submissions ------------------------------------------------------------
def checks(name, p):
    assert len(p) == len(test), 'row count mismatch'
    assert ((p > 0) & (p < 1)).all(), 'probability out of range'
    assert not np.isnan(p).any(), 'NaN in predictions'
    print(f'checks passed: {name}')

checks('blend', best_pred)
checks('lgbm_best', pred_m[best_single if best_single.startswith('lgbm') else 'lgbm_c10_s42'])
pd.DataFrame({'id': test['id'], 'target': best_pred}).to_csv('submissions/submission_blend.csv', index=False)
pd.DataFrame({'id': test['id'], 'target': pred_m['lgbm_c10_s42']}).to_csv('submissions/submission_lgbm.csv', index=False)

ss = pd.read_csv('data/sample_submission.csv')
for f in ('submissions/submission_blend.csv', 'submissions/submission_lgbm.csv'):
    sub = pd.read_csv(f)
    assert list(sub.columns) == list(ss.columns), f'{f}: column mismatch'
    assert (sub['id'] == ss['id']).all(), f'{f}: id order mismatch'
print('submission format validated against sample_submission')

with open(f'{R}/results_ensemble.txt', 'w') as f:
    f.write('\n'.join(report) + '\n')
print('ALL DONE')
