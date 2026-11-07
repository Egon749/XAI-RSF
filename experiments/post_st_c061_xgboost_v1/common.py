import json, os, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
YEARS = (2023, 2024, 2025)
SEEDS = (17, 155, 8679)
def read(p): return json.loads(Path(p).read_text())
def write(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False,
        default=lambda z: z.item() if isinstance(z, np.generic) else str(z))+'\n')
def arrays(p):
    with np.load(p, allow_pickle=False) as f: return dict(f)
def sha(p):
    with open(p, 'rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def location(): return read(HERE/'run_location.json')
def run(): return Path(os.environ.get('C061_XGB_RUN') or location()['private_run'])
def source(): return Path(location()['source_run'])

def recensor(train, metadata, inner_year):
    """Rebuild an earlier training snapshot, never just filter entry dates."""
    import pandas as pd
    from experiments.post_st_prognosis_v1.common import calendar_elapsed
    cutoff = pd.Timestamp(inner_year, 1, 1)
    keep = (metadata.origin < cutoff).to_numpy()
    stop = metadata.stop.where(metadata.stop < cutoff, cutoff-pd.Timedelta(days=1))
    result = {k: v[keep].copy() for k, v in train.items()}
    result['time'] = (stop[keep]-metadata.origin[keep]).dt.days.to_numpy()
    result['event'] = np.where((metadata.stop[keep] < cutoff), train['event'][keep], 0)
    result['analysis_time'] = np.array([calendar_elapsed(o,s) for o,s in zip(metadata.origin[keep],stop[keep])])
    assert (result['time'] > 0).all()
    return result
