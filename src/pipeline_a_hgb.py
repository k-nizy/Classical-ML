"""
Pipeline A: sklearn HistGradientBoostingClassifier.

Handles: mixed types (ordinal-encode cats), native NaN handling,
         no scaling needed for trees.

CV: StratifiedKFold(5, shuffle, seed 42) -> OOF AUC + test predictions.
Writes: oof_hgb.npy, pred_hgb.npy, results_a.txt
"""
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SEED = 42
N_FOLDS = 5

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# --- preprocessing -----------------------------------------------------------
# Drop exact duplicates (keep one of each pair). Region looks like noise but
# trees handle junk fine; we keep it and let feature importance tell us later.
DROP = ['num_feat_8', 'num_feat_18', 'num_feat_34']
features = [c for c in train.columns if c not in ('id', 'target') + tuple(DROP)]

for c in ['cat_region', 'cat_channel', 'cat_tier']:
    train[c] = train[c].astype('category')
    test[c] = test[c].astype('category')

X = train[features].copy()
X_test = test[features].copy()
# align categories
for c in ['cat_region', 'cat_channel', 'cat_tier']:
    cats = pd.Index(sorted(set(X[c].dropna()) | set(X_test[c].dropna())))
    X[c] = X[c].cat.set_categories(cats)
    X_test[c] = X_test[c].cat.set_categories(cats)

y = train['target'].values

params = dict(
    max_iter=600,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=40,
    l2_regularization=1.0,
    max_bins=255,
    early_stopping=False,
    random_state=SEED,
)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof = np.zeros(len(X))
pred_test = np.zeros(len(X_test))
fold_aucs = []

t0 = time.time()
for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    model = HistGradientBoostingClassifier(**params)
    model.fit(X_tr, y_tr)

    oof[va_idx] = model.predict_proba(X_va)[:, 1]
    pred_test += model.predict_proba(X_test)[:, 1] / N_FOLDS

    auc = roc_auc_score(y_va, oof[va_idx])
    fold_aucs.append(auc)
    print(f'fold {fold}: AUC = {auc:.5f}', flush=True)

cv_auc = roc_auc_score(y, oof)
elapsed = time.time() - t0
print(f'\nOOF AUC: {cv_auc:.5f}  (fold std: {np.std(fold_aucs):.5f})')
print(f'total time: {elapsed:.1f}s')

# feature importances from a full-data model
full = HistGradientBoostingClassifier(**params)
full.fit(X, y)
try:
    imp = pd.Series(full.feature_importances_, index=features).sort_values(ascending=False)
    print('\nTop 15 features:')
    print(imp.head(15).round(4).to_string())
    print('\nBottom 5 features:')
    print(imp.tail(5).round(6).to_string())
except Exception as e:
    print('no feature_importances_:', e)

np.save('results/oof_hgb.npy', oof)
np.save('results/pred_hgb.npy', pred_test)

with open('results/results_a.txt', 'w') as f:
    f.write(f'pipeline A (HGB)\nparams: {params}\n')
    f.write(f'fold AUCs: {[round(a,5) for a in fold_aucs]}\n')
    f.write(f'OOF AUC: {cv_auc:.5f} +- {np.std(fold_aucs):.5f}\n')
    f.write(f'time: {elapsed:.1f}s\n')
print('done -> results_a.txt')
