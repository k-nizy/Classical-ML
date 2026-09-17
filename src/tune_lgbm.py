"""
Randomized search for LightGBM hyperparameters.

- 3-fold CV per config (fast screening)
- appends each finished config to tune_results.csv -> resumable
- rerun this script to continue where it left off
"""
import os
import json
import time
import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 42
N_FOLDS_SEARCH = 3
N_CONFIGS = 26
RESULTS = 'results/tune_results.csv'

train = pd.read_csv('data/train.csv')
DROP = ['num_feat_8', 'num_feat_18', 'num_feat_34']
features = [c for c in train.columns if c not in ('id', 'target') + tuple(DROP)]
for c in ['cat_region', 'cat_channel', 'cat_tier']:
    train[c] = train[c].astype('category')
X, y = train[features], train['target'].values
cat_idx = [features.index(c) for c in ['cat_region', 'cat_channel', 'cat_tier']]

rng = np.random.RandomState(SEED)

def sample_config():
    if cfg_id < 12:  # batch 1: broad exploration
        return dict(
            num_leaves=int(rng.choice([31, 63, 127, 255])),
            min_data_in_leaf=int(rng.choice([10, 20, 40, 80, 150])),
            feature_fraction=float(rng.uniform(0.5, 1.0)),
            bagging_fraction=float(rng.uniform(0.5, 1.0)),
            bagging_freq=1,
            lambda_l1=float(10 ** rng.uniform(-3, 1)),
            lambda_l2=float(10 ** rng.uniform(-3, 1)),
            max_bin=int(rng.choice([127, 255, 511])),
        )
    # batch 2: focused around best region from batch 1
    return dict(
        num_leaves=int(rng.choice([127, 191, 255, 383])),
        min_data_in_leaf=int(rng.choice([10, 15, 20, 30, 40])),
        feature_fraction=float(rng.uniform(0.85, 1.0)),
        bagging_fraction=float(rng.uniform(0.7, 1.0)),
        bagging_freq=1,
        lambda_l1=float(10 ** rng.uniform(-3, -0.5)),
        lambda_l2=float(10 ** rng.uniform(-3, 0.0)),
        max_bin=int(rng.choice([127, 255])),
        min_gain_to_split=float(10 ** rng.uniform(-3, -1)),
    )

base = dict(objective='binary', metric='auc', learning_rate=0.07,
            verbosity=-1, seed=SEED, feature_pre_filter=False)

done = set()
if os.path.exists(RESULTS):
    done = set(pd.read_csv(RESULTS)['status'].index.tolist()) if False else set(
        pd.read_csv(RESULTS).query("done == 1")['cfg_id'])
records = []
skf = StratifiedKFold(n_splits=N_FOLDS_SEARCH, shuffle=True, random_state=SEED)

t_start = time.time()
for cfg_id in range(N_CONFIGS):
    if cfg_id in done:
        continue
    cfg = sample_config()
    params = {**base, **cfg}
    aucs = []
    t0 = time.time()
    for tr_idx, va_idx in skf.split(X, y):
        dtr = lgb.Dataset(X.iloc[tr_idx], y[tr_idx], categorical_feature=cat_idx)
        dva = lgb.Dataset(X.iloc[va_idx], y[va_idx], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=1500,
                      valid_sets=[dva], callbacks=[lgb.early_stopping(60, verbose=False)])
        aucs.append(roc_auc_score(y[va_idx], m.predict(X.iloc[va_idx], num_iteration=m.best_iteration)))
    mean_auc, std = float(np.mean(aucs)), float(np.std(aucs))
    rec = {'cfg_id': cfg_id, 'done': 1, 'auc': round(mean_auc, 5), 'std': round(std, 5),
           'secs': round(time.time() - t0), **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in cfg.items()}}
    records.append(rec)
    pd.DataFrame([rec]).to_csv(RESULTS, mode='a', header=not os.path.exists(RESULTS), index=False)
    print(f"cfg {cfg_id}: AUC {mean_auc:.5f}  {cfg}", flush=True)
    if time.time() - t_start > 420:   # leave time to finish cleanly
        print('time budget reached - rerun script to resume', flush=True)
        break

res = pd.read_csv(RESULTS)
print('\n=== TOP 5 CONFIGS ===')
print(res.sort_values('auc', ascending=False).head(5).to_string())
