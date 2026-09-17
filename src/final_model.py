"""
Final model: multi-seed LightGBM (2 tuned configs) + multi-seed HGB,
blended via OOF-optimal weight.

Writes: oof_lgbm_final.npy, pred_lgbm_final.npy,
        oof_hgb_final.npy,  pred_hgb_final.npy, results_final.txt
"""
import time
import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SEEDS = [42, 43]
N_FOLDS = 5

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

DROP = ['num_feat_8', 'num_feat_18', 'num_feat_34']
features = [c for c in train.columns if c not in ('id', 'target') + tuple(DROP)]
for c in ['cat_region', 'cat_channel', 'cat_tier']:
    train[c] = train[c].astype('category')
    test[c] = test[c].astype('category')
    cats = pd.Index(sorted(set(train[c].dropna()) | set(test[c].dropna())))
    train[c] = train[c].cat.set_categories(cats)
    test[c] = test[c].cat.set_categories(cats)

X, X_test, y = train[features], test[features], train['target'].values
cat_idx = [features.index(c) for c in ['cat_region', 'cat_channel', 'cat_tier']]

# best configs from randomized search (3-fold screening)
LGBM_CFGS = [
    dict(learning_rate=0.07, num_leaves=255, min_data_in_leaf=20,
         feature_fraction=0.982627653632069, bagging_fraction=0.8035171238433423,
         bagging_freq=1, lambda_l1=0.012705645329288707, lambda_l2=0.01531418971165477,
         max_bin=127),
    dict(learning_rate=0.07, num_leaves=191, min_data_in_leaf=30,
         feature_fraction=0.8567840933365807, bagging_fraction=0.7975990992289793,
         bagging_freq=1, lambda_l1=0.009368999682313825, lambda_l2=0.006516990611177174,
         max_bin=127, min_gain_to_split=0.014910847596941081),
]
LGBM_BASE = dict(objective='binary', metric='auc', verbosity=-1,
                 feature_pre_filter=False)

HGB_PARAMS = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                  min_samples_leaf=40, l2_regularization=1.0, max_bins=255,
                  early_stopping=False)

report = []

def run_lgbm():
    oof_all, pred_all = [], []
    for ci, cfg in enumerate(LGBM_CFGS):
        for seed in SEEDS:
            params = {**LGBM_BASE, **cfg, 'seed': seed}
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            oof = np.zeros(len(X)); pred = np.zeros(len(X_test)); aucs = []
            t0 = time.time()
            for tr, va in skf.split(X, y):
                dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_idx)
                dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr)
                m = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                              callbacks=[lgb.early_stopping(80, verbose=False)])
                oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
                pred += m.predict(X_test, num_iteration=m.best_iteration) / N_FOLDS
                aucs.append(roc_auc_score(y[va], oof[va]))
            auc = roc_auc_score(y, oof)
            oof_all.append(oof); pred_all.append(pred)
            msg = f'LGBM cfg{ci} seed{seed}: OOF {auc:.5f} (folds {[round(a,5) for a in aucs]}, {time.time()-t0:.0f}s)'
            print(msg, flush=True); report.append(msg)
    return np.mean(oof_all, axis=0), np.mean(pred_all, axis=0)

def run_hgb():
    oof_all, pred_all = [], []
    for seed in SEEDS:
        params = {**HGB_PARAMS, 'random_state': seed}
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros(len(X)); pred = np.zeros(len(X_test)); aucs = []
        t0 = time.time()
        for tr, va in skf.split(X, y):
            m = HistGradientBoostingClassifier(**params)
            m.fit(X.iloc[tr], y[tr])
            oof[va] = m.predict_proba(X.iloc[va])[:, 1]
            pred += m.predict_proba(X_test)[:, 1] / N_FOLDS
            aucs.append(roc_auc_score(y[va], oof[va]))
        auc = roc_auc_score(y, oof)
        oof_all.append(oof); pred_all.append(pred)
        msg = f'HGB seed{seed}: OOF {auc:.5f} (folds {[round(a,5) for a in aucs]}, {time.time()-t0:.0f}s)'
        print(msg, flush=True); report.append(msg)
    return np.mean(oof_all, axis=0), np.mean(pred_all, axis=0)

t0 = time.time()
oof_l, pred_l = run_lgbm()
oof_h, pred_h = run_hgb()

auc_l = roc_auc_score(y, oof_l)
auc_h = roc_auc_score(y, oof_h)
best_w, best_auc = 1.0, auc_l
for w in np.linspace(0, 1, 201):
    a = roc_auc_score(y, w * oof_l + (1 - w) * oof_h)
    if a > best_auc:
        best_w, best_auc = w, a

msg = (f'\nOOF AUC  LGBM: {auc_l:.5f} | HGB: {auc_h:.5f} | '
       f'blend w={best_w:.3f}: {best_auc:.5f}')
print(msg, flush=True); report.append(msg)
report.append(f'total time: {time.time()-t0:.0f}s')

np.save('results/oof_lgbm_final.npy', oof_l); np.save('results/pred_lgbm_final.npy', pred_l)
np.save('results/oof_hgb_final.npy', oof_h);  np.save('results/pred_hgb_final.npy', pred_h)
with open('results/results_final.txt', 'w') as f:
    f.write('\n'.join(report) + '\n')
print('done -> results_final.txt')
