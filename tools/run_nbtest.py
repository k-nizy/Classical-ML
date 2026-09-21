"""Execute the coursework notebook end-to-end, cell by cell with progress output.

SMOKE=1  -> fast pass: truncates train to 4000 rows in the executed copy only
            and forces wandb into no-network 'disabled' mode
SMOKE=0  -> full-size run (the deliverable verification)
"""
import os
import time

SMOKE = os.environ.get('SMOKE') == '1'

import nbformat
from nbclient import NotebookClient

nb = nbformat.read('Formative1Part2_Classification.ipynb', as_version=4)

if SMOKE:
    for c in nb.cells:
        if c.cell_type != 'code':
            continue
        if 'train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))' in c.source:
            c.source += ("\n\n# SMOKE-TEST override (executed copy only, not in the deliverable)\n"
                         "train = train.sample(min(4000, len(train)), random_state=0).reset_index(drop=True)")
        if 'WANDB_MODE' in c.source and 'wandb' in c.source:
            c.source += "\n\n# SMOKE-TEST: keep wandb fully offline/no-network\nos.environ['WANDB_MODE'] = 'disabled'"

t_start = time.time()

def _log(msg):
    print(msg.encode('ascii', 'replace').decode('ascii'), flush=True)

def on_cell_start(cell, cell_index):
    preview = cell.source[:60].replace('\n', ' ')
    _log(f'[cell {cell_index:2d} {cell.cell_type:8s}] {preview}...')

def on_cell_complete(cell, cell_index, execute_reply=None):
    _log(f'[cell {cell_index:2d}] done  (+{time.time()-t_start:6.0f}s total)')

def on_cell_error(cell, cell_index, execute_reply=None):
    _log(f'[cell {cell_index:2d}] ERROR')
    if execute_reply:
        _log(str(execute_reply.get('content', {}).get('ename', '')))

client = NotebookClient(nb, timeout=7200, kernel_name='python3',
                        resources={'metadata': {'path': '.'}},
                        allow_errors=False)
client.on_cell_start = on_cell_start
client.on_cell_complete = on_cell_complete
client.on_cell_error = on_cell_error

client.execute()

out = 'results/executed_smoke.ipynb' if SMOKE else 'results/executed_full.ipynb'
nbformat.write(nb, out)
print(f'EXECUTION OK ({time.time()-t_start:.0f}s) -> {out}')
