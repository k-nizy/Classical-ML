"""
Blend selection utilities shared by the final ensemble and the submission tool.

Strategy (parsimony-first, to protect the private leaderboard):
  1. best single model
  2. best pair (grid over mixing weight, step 0.01)
  3. NNLS over all OOFs (non-negative least squares on probabilities)
  The best-scoring option wins; ties effectively favor the simpler option
  because a later option must strictly beat the current best.
"""
import numpy as np
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score


def load_oof_pred(results_dir, tags):
    oof = {t: np.load(f'{results_dir}/oof_{t}.npy') for t in tags}
    pred = {t: np.load(f'{results_dir}/pred_{t}.npy') for t in tags}
    return oof, pred


def select_blend(y, tags, oof, pred, log=print):
    """Return (name, oof_auc, test_pred, singles, best_single_tag)."""
    singles = {t: roc_auc_score(y, oof[t]) for t in tags}
    for t in sorted(singles, key=singles.get, reverse=True):
        log(f'  single {t}: {singles[t]:.5f}')

    best_single = max(singles, key=singles.get)
    best_auc, best_name, best_pred = singles[best_single], best_single, pred[best_single]

    # best pair
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            for w in np.linspace(0, 1, 101):
                auc = roc_auc_score(y, w * oof[a] + (1 - w) * oof[b])
                if auc > best_auc:
                    best_auc = auc
                    best_name = f'pair {a}+{b} w={w:.2f}'
                    best_pred = w * pred[a] + (1 - w) * pred[b]

    # NNLS over all OOFs
    O = np.column_stack([oof[t] for t in tags])
    w_nnls, _ = nnls(O, y.astype(float))
    w_norm = w_nnls / w_nnls.sum()
    auc_nnls = roc_auc_score(y, O @ w_norm)
    if auc_nnls > best_auc:
        best_auc = auc_nnls
        best_name = 'nnls ' + ', '.join(
            f'{t}:{w:.2f}' for t, w in zip(tags, w_norm) if w > 0.01)
        best_pred = np.column_stack([pred[t] for t in tags]) @ w_norm

    return best_name, best_auc, best_pred, singles, best_single
