"""
Shared feature construction.

build_features(train, test, variant) -> X, X_test

Variants (cumulative):
  base      : drop exact duplicates, category dtype
  rowstats  : + per-row stats over numerics (mean, std, |z|>2 count, NaN count)
  interact  : + products/diffs of top-6 numeric features (signal lives in interactions)
  combo     : + channel x tier combined categorical
  full      : rowstats + interact + combo
"""
import numpy as np
import pandas as pd

DROP = ['num_feat_8', 'num_feat_18', 'num_feat_34']          # exact duplicates
TOP6 = ['num_feat_16', 'num_feat_29', 'num_feat_6', 'num_feat_11',
        'num_feat_24', 'num_feat_15']                        # top-6 by LGBM gain
CATS = ['cat_region', 'cat_channel', 'cat_tier']


def _align_cats(train, test):
    for c in CATS:
        cats = pd.Index(sorted(set(train[c].dropna()) | set(test[c].dropna())))
        train[c] = train[c].cat.set_categories(cats)
        test[c] = test[c].cat.set_categories(cats)


def build_features(train: pd.DataFrame, test: pd.DataFrame, variant: str = 'base'):
    assert variant in ('base', 'rowstats', 'interact', 'combo', 'full')
    train = train.copy()
    test = test.copy()

    for c in CATS:
        train[c] = train[c].astype('category')
        test[c] = test[c].astype('category')
    _align_cats(train, test)

    feats = [c for c in train.columns if c not in ('id', 'target') + tuple(DROP)]
    num_cols = [c for c in feats if c.startswith('num_feat')]

    X, X_test = train[feats].copy(), test[feats].copy()

    want_rows = variant in ('rowstats', 'full')
    want_inter = variant in ('interact', 'full')
    want_combo = variant in ('combo', 'full')

    if want_rows:
        for df in (X, X_test):
            sub = df[num_cols].astype(float)
            z = (sub - sub.mean()) / sub.std()
            df['rs_mean'] = sub.mean(axis=1)
            df['rs_std'] = sub.std(axis=1)
            df['rs_extreme'] = (z.abs() > 2).sum(axis=1)
            df['rs_nan'] = sub.isna().sum(axis=1)

    if want_inter:
        pairs = [(a, b) for i, a in enumerate(TOP6) for b in TOP6[i + 1:]]
        for a, b in pairs:
            for df in (X, X_test):
                df[f'{a}_x_{b}'] = df[a].astype(float) * df[b].astype(float)
                df[f'{a}_m_{b}'] = df[a].astype(float) - df[b].astype(float)

    if want_combo:
        for df in (X, X_test):
            df['cat_channel_tier'] = (
                df['cat_channel'].astype(str) + '_' + df['cat_tier'].astype(str)
            ).replace('nan_nan', np.nan).astype('category')

    return X, X_test
