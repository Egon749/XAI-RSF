"""Re-read both authoritative 2025 workbooks and verify every cached daily row."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

if '--runtime' not in sys.argv:
    exe='/home/oliver/.venvs/xai-baselines-20261027/bin/python' if socket.gethostname().startswith('oliver') else '/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python'
    os.execv(exe,[exe,__file__,'--runtime',*sys.argv[1:]])
sys.argv.remove('--runtime')
root=Path('/home/oliver/MainDiskData/early_warning_12m_server_isolated_20261025/code')
sys.path[:0]=[str(root/'registry/src'),str(root/'vendor')]
from research_data import Registry
import numpy as np
import pandas as pd


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def run(index):
    start=time.perf_counter()
    path=Path('/home/oliver/MainDiskData/early_warning_24m_2012plus_20261025/results/market_2025')/f'source-{index:02d}'/'2462'/f'task-{index}'/'output'
    meta=json.loads((path/'summary.json').read_text());r=Registry(root/'registry');desc=r.describe(meta['dataset_id'])
    assert desc['version']==meta['version'] and desc['storage_path']==meta['source']
    cols=['Stkcd','Trddt','Clsprc','Dretwd','Dsmvtll','Dsmvosd','Dnshrtrd','Dnvaltrd','Hiprc','Loprc','Trdsta','Markettype']
    cached=pd.read_parquet(path/'daily.parquet',columns=cols);at=0;rows=0
    source_hash=sha(meta['source']);cache_hash=sha(path/'daily.parquet')
    for b in r.iter_batches(meta['dataset_id'],columns=cols,max_rows=10000000,batch_size=8192,timeout_seconds=3600,scan_budget_bytes=4000000000):
        f=b.to_pandas();rows+=len(f);f=f[(f.Trddt>='2024-01-01')&(f.Trddt<='2025-12-31')].copy()
        if not len(f):continue
        expected=cached.iloc[at:at+len(f)]
        np.testing.assert_array_equal(f.Stkcd.astype(str),expected.Stkcd.astype(str))
        np.testing.assert_array_equal(pd.to_datetime(f.Trddt),pd.to_datetime(expected.Trddt))
        for c in cols[2:]:
            np.testing.assert_allclose(pd.to_numeric(f[c],errors='coerce'),pd.to_numeric(expected[c],errors='coerce'),rtol=0,atol=0,equal_nan=True)
        at+=len(f)
    assert rows==meta['input_rows'] and at==len(cached)==meta['kept_rows']
    assert source_hash==sha(meta['source']) and cache_hash==sha(path/'daily.parquet')
    out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
    result=dict(status='PASS',source=index,raw_rows=rows,verified_daily_rows=at,host=socket.gethostname(),seconds=time.perf_counter()-start,
        raw_source_sha256=source_hash,cache_sha256=cache_hash,dataset_id=meta['dataset_id'],dataset_version=meta['version'],
        code_sha256=sha(__file__),new_model_fits=0)
    with (out/'verification.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=int,choices=[0,1],required=True);run(p.parse_args().source)
