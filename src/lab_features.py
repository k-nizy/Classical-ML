"""
Feature lab: OOF LGBM comparison of feature variants + noise-column ablation.

Appends one row per variant to results/feature_lab.csv (resumable by variant name).
Writes final comparison to results/results_featlab.txt
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # allow running from project root
import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from features import build_features, CATS

SEED = 42
N_FOLDS = 5
OUT = 'results/feature_lab.csv'
VARIANTS = ['base', 'rowstats', 'interact', 'combo', 'full', 'ablation']

train = pd.read_csv('data/train.csv')
y = train['target'].values

# tuned LGBM params from the search (cfg 10, best screening AUC)
PARAMS = dict(objective='binary', metric='auc', learning_rate=0.07,
              num_leaves=255, min_data_in_leaf=20,
              feature_fraction=0.982627653632069, bagging_fraction=0.8035171238433423,
              bagging_freq=1, lambda_l1=0.012705645329288707,
              lambda_l2=0.01531418971165477, max_bin=127,
              verbosity=-1, seed=SEED, feature_pre_filter=False)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def oof_auc(X, y, drop_cols=()):
    X = X.drop(columns=list(drop_cols)) if drop_cols else X
    cat_cols = [c for c in X.columns if c in CATS or c == 'cat_channel_tier']
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    oof = np.zeros(len(X))
    for tr, va in skf.split(X, y):
        dtr = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=cat_idx)
        dva = lgb.Dataset(X.iloc[va], y[va], reference=dtr)
        m = lgb.train(PARAMS, dtr, num_boost_round=1200, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(60, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
    return roc_auc_score(y, oof)


test = pd.read_csv('data/test.csv')
done = set()
if os.path.exists(OUT):
    done = set(pd.read_csv(OUT)['variant'])
rows = []
t_all = time.time()
for v in VARIANTS:
    if v in done:
        continue
    t0 = time.time()
    # ablation = best variant minus the two noise categoricals
    X, _ = build_features(train, test, 'interact' if v == 'ablation' else v)
    drop = ['cat_region', 'cat_tier'] if v == 'ablation' else ()
    auc = oof_auc(X, y, drop_cols=drop)
    rows.append({'variant': v, 'oof_auc': round(auc, 5), 'n_features': X.shape[1] - len(drop),
                 'secs': round(time.time() - t0)})
    pd.DataFrame(rows).to_csv(OUT, mode='a', header=not os.path.exists(OUT), index=False)
    print(rows[-1], flush=True)

res = pd.read_csv(OUT).drop_duplicates('variant', keep='last').set_index('variant')
res['delta_vs_base'] = (res['oof_auc'] - res.loc['base', 'oof_auc']).round(5)
lines = ['=== Feature lab (OOF LGBM AUC, cfg10 params, 5-fold seed 42) ===',
         res[['oof_auc', 'delta_vs_base', 'n_features', 'secs']].to_string()]
print('\n'.join(lines))
with open('results/results_featlab.txt', 'w') as f:
    f.write('\n'.join(lines) + '\n')
