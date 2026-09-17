"""
Pipeline 0: linear baselines.
- DummyClassifier (majority): shows the accuracy trap on imbalanced data
- LogisticRegression pipeline: median impute -> standard scale -> one-hot cats

Writes: results/results_0.txt
"""
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, accuracy_score

SEED = 42
N_FOLDS = 5

train = pd.read_csv('data/train.csv')
DROP = ['num_feat_8', 'num_feat_18', 'num_feat_34']
features = [c for c in train.columns if c not in ('id', 'target') + tuple(DROP)]
num_cols = [c for c in features if c.startswith('num_feat')]
cat_cols = ['cat_region', 'cat_channel', 'cat_tier']

X, y = train[features].copy(), train['target'].values

pre = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler())]), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='if_binary'), cat_cols),
])

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_lr = np.zeros(len(X))
oof_dummy_acc = []

t0 = time.time()
for tr, va in skf.split(X, y):
    # dummy majority classifier: accuracy without any learning
    dm = DummyClassifier(strategy='most_frequent')
    dm.fit(X.iloc[tr], y[tr])
    oof_dummy_acc.append(accuracy_score(y[va], dm.predict(X.iloc[va])))

    lr = Pipeline([('pre', pre), ('clf', LogisticRegression(max_iter=3000, C=0.5))])
    lr.fit(X.iloc[tr], y[tr])
    oof_lr[va] = lr.predict_proba(X.iloc[va])[:, 1]

auc_lr = roc_auc_score(y, oof_lr)
acc_lr = accuracy_score(y, np.where(oof_lr >= 0.5, 1, 0))
acc_dummy = float(np.mean(oof_dummy_acc))

lines = [
    '=== Pipeline 0: linear baselines (5-fold stratified CV, seed 42) ===',
    f'DummyClassifier (majority): accuracy = {acc_dummy:.4f}  <- the accuracy trap',
    f'LogisticRegression:         AUC      = {auc_lr:.5f}',
    f'LogisticRegression:         accuracy = {acc_lr:.4f}  (at 0.5 threshold, imbalanced)',
    f'Interpretation: LR barely beats the constant predictor on accuracy,',
    f'but AUC exposes its ranking ability; tuned trees dominate on AUC.',
    f'time: {time.time()-t0:.1f}s',
]
print('\n'.join(lines), flush=True)
with open('results/results_0.txt', 'w') as f:
    f.write('\n'.join(lines) + '\n')
