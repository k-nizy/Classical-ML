"""Create submission files with sanity checks."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
y = train['target'].values

oof_l = np.load('results/oof_lgbm_final.npy'); pred_l = np.load('results/pred_lgbm_final.npy')
oof_h = np.load('results/oof_hgb_final.npy');  pred_h = np.load('results/pred_hgb_final.npy')

W = 0.935
oof_blend = W * oof_l + (1 - W) * oof_h
pred_blend = W * pred_l + (1 - W) * pred_h

checks = []
def check(name, cond):
    checks.append(f'{"PASS" if cond else "FAIL"}: {name}')
    return cond

check('oof blend AUC matches results_final (0.84345)',
      abs(roc_auc_score(y, oof_blend) - 0.84345) < 0.0002)

for name, p in [('blend', pred_blend), ('lgbm', pred_l)]:
    check(f'{name}: {len(p)} rows', len(p) == len(test))
    check(f'{name}: all probs in (0,1)', ((p > 0) & (p < 1)).all())
    check(f'{name}: no NaN', not np.isnan(p).any())
    check(f'{name}: no all-identical values', np.std(p) > 0.01)

print('\n'.join(checks))

sub_blend = pd.DataFrame({'id': test['id'], 'target': pred_blend})
sub_lgbm  = pd.DataFrame({'id': test['id'], 'target': pred_l})
sub_blend.to_csv('submissions/submission_blend.csv', index=False)
sub_lgbm.to_csv('submissions/submission_lgbm.csv', index=False)

ss = pd.read_csv('data/sample_submission.csv')
print('\nformat matches sample_submission:',
      list(sub_blend.columns) == list(ss.columns),
      '| sample ids match order:', (sub_blend['id'] == ss['id']).all())
print('\nsubmission_blend.csv head:')
print(sub_blend.head(3).to_string(index=False))
print('\nmean pred blend:', round(sub_blend['target'].mean(), 4),
      '| mean pred lgbm:', round(sub_lgbm['target'].mean(), 4))
