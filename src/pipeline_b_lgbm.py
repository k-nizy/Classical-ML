"""
Pipeline B: LightGBM with native categorical feature support.

Writes: oof_lgbm.npy, pred_lgbm.npy, results_b.txt
"""
import time
import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 42
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

params = dict(
    objective='binary',
    metric='auc',
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=40,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    max_bin=255,
    verbosity=-1,
    seed=SEED,
)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(X))
pred_test = np.zeros(len(X_test))
fold_aucs = []

t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    dtr = lgb.Dataset(X.iloc[tr_idx], y[tr_idx], categorical_feature=cat_idx)
    dva = lgb.Dataset(X.iloc[va_idx], y[va_idx], reference=dtr)

    model = lgb.train(
        params, dtr,
        num_boost_round=2000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    best_it = model.best_iteration

    oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=best_it)
    pred_test += model.predict(X_test, num_iteration=best_it) / N_FOLDS

    auc = roc_auc_score(y[va_idx], oof[va_idx])
    fold_aucs.append(auc)
    print(f'fold {fold}: AUC = {auc:.5f} (trees={best_it})', flush=True)

cv_auc = roc_auc_score(y, oof)
elapsed = time.time() - t0
print(f'\nOOF AUC: {cv_auc:.5f}  (fold std: {np.std(fold_aucs):.5f})')
print(f'total time: {elapsed:.1f}s')

# feature importance from full-data model
full = lgb.train(params, lgb.Dataset(X, y, categorical_feature=cat_idx),
                 num_boost_round=int(best_it * 1.1))
imp = pd.Series(full.feature_importance('gain'), index=features).sort_values(ascending=False)
print('\nTop 15 features (gain):')
print(imp.head(15).round(0).to_string())
print('\nBottom 5 (near-zero = noise candidates):')
print(imp.tail(5).round(1).to_string())

np.save('results/oof_lgbm.npy', oof)
np.save('results/pred_lgbm.npy', pred_test)

with open('results/results_b.txt', 'w') as f:
    f.write(f'pipeline B (LGBM)\nparams: {params}\n')
    f.write(f'fold AUCs: {[round(a,5) for a in fold_aucs]}\n')
    f.write(f'OOF AUC: {cv_auc:.5f} +- {np.std(fold_aucs):.5f}\n')
    f.write(f'time: {elapsed:.1f}s\n')
print('done -> results_b.txt')
