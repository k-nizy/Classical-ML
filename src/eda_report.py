"""EDA for the classification competition - prints to eda_output.txt."""
import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
num_cols = [c for c in train.columns if c.startswith('num_feat')]
cat_cols = ['cat_region', 'cat_channel', 'cat_tier']

lines = []
def p(*args):
    lines.append(' '.join(str(a) for a in args))

p('=== NUMERIC SUMMARY (train) ===')
desc = train[num_cols].describe().T[['mean', 'std', 'min', 'max']]
p(desc.round(3).to_string())

p('')
p('=== TEST vs TRAIN NUMERIC DISTRIBUTIONS (mean/std) ===')
for c in num_cols:
    p(f"{c:14s} train: {train[c].mean():9.3f}±{train[c].std():8.3f}   test: {test[c].mean():9.3f}±{test[c].std():8.3f}")

p('')
p('=== CAT VALUE DISTRIBUTION train vs test ===')
for c in cat_cols:
    tr = train[c].value_counts(dropna=False, normalize=True).round(3)
    te = test[c].value_counts(dropna=False, normalize=True).round(3)
    p(c)
    p('  train:', tr.to_dict())
    p('  test :', te.to_dict())

p('')
p('=== ID FORMAT CHECK ===')
p('train id sample:', train['id'].head(2).tolist())
p('test id sample:', test['id'].head(2).tolist())
p('id overlap train/test:', len(set(train['id']) & set(test['id'])))

p('')
p('=== CONSTANT / NEAR-CONSTANT NUMERIC FEATURES ===')
for c in num_cols:
    nunique = train[c].nunique(dropna=True)
    if nunique < 10:
        p(c, 'nunique:', nunique, train[c].value_counts(dropna=True).head(5).to_dict())

with open('results/eda_output.txt', 'w') as f:
    f.write('\n'.join(str(x) for x in lines))
print('done, wrote eda_output.txt')
