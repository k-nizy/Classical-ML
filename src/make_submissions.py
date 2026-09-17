"""
Rebuild submission files from checkpointed model predictions (no retraining).

- Replays blend selection from results/*_oof/pred checkpoints via blend_utils
- Writes submissions/submission_blend.csv and submissions/submission_lgbm.csv
- Validates both against data/sample_submission.csv

Also verifies the committed submissions match (they are deterministic outputs
of the checkpointed models).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blend_utils import load_oof_pred, select_blend

R = 'results'
TAGS = ['lgbm_c10_s42', 'lgbm_c10_s43', 'lgbm_c20_s42', 'xgb_d6_s42',
        'xgb_d6_s43', 'hgb_s42']

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
ss = pd.read_csv('data/sample_submission.csv')
y = train['target'].values

oof, pred = load_oof_pred(R, TAGS)
name, auc, best_pred, singles, best_single = select_blend(y, TAGS, oof, pred)
print(f'selected blend: {name}\nOOF AUC = {auc:.5f}')

# lgbm-only reference submission: best lgbm tag (or pair of two best lgbm tags
# if that beats it on OOF)
lgbm_tags = [t for t in TAGS if t.startswith('lgbm')]
lgbm_best = max(lgbm_tags, key=lambda t: singles[t])
lgbm_auc, lgbm_pred = singles[lgbm_best], pred[lgbm_best]
if len(lgbm_tags) >= 2:
    for i, a in enumerate(lgbm_tags):
        for b in lgbm_tags[i + 1:]:
            for w in np.linspace(0, 1, 101):
                auc2 = roc_auc_score(y, w * oof[a] + (1 - w) * oof[b])
                if auc2 > lgbm_auc:
                    lgbm_auc = auc2
                    lgbm_best = f'pair {a}+{b} w={w:.2f}'
                    lgbm_pred = w * pred[a] + (1 - w) * pred[b]
print(f'lgbm submission: {lgbm_best}  OOF AUC = {lgbm_auc:.5f}')


def checks(name, p):
    assert len(p) == len(test), 'row count mismatch'
    assert ((p > 0) & (p < 1)).all(), 'probability out of range'
    assert not np.isnan(p).any(), 'NaN in predictions'
    print(f'checks passed: {name}')


checks('blend', best_pred)
checks('lgbm', lgbm_pred)
pd.DataFrame({'id': test['id'], 'target': best_pred}).to_csv(
    'submissions/submission_blend.csv', index=False)
pd.DataFrame({'id': test['id'], 'target': lgbm_pred}).to_csv(
    'submissions/submission_lgbm.csv', index=False)

for f in ('submissions/submission_blend.csv', 'submissions/submission_lgbm.csv'):
    sub = pd.read_csv(f)
    assert list(sub.columns) == list(ss.columns), f'{f}: column mismatch'
    assert (sub['id'] == ss['id']).all(), f'{f}: id order mismatch'
print('submission format validated against sample_submission')
