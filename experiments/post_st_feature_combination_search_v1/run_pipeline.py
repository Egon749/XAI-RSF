"""Reproduce the frozen study into a new private directory."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from experiments.post_st_dropcolumn_selection_v1.common import load, write, sha


def execute(run):
    repo=Path(__file__).resolve().parents[2]
    public=Path(__file__).resolve().parent
    if run.exists() or not run.is_absolute() or not run.is_relative_to(Path('/home/oliver/MainDiskData')):
        raise ValueError('Use a new absolute private directory under shared MainDiskData')
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip():
        raise ValueError('Commit the intended version before reproduction')
    protocol=load(public/'frozen_protocol.json')
    original=Path(protocol['private_run'])
    sources=load(original/'audit/source_hashes.json')
    for path,expected in sources.items():
        assert sha(Path(path))==expected,path
    subprocess.run(['ssh','-o','BatchMode=yes','lemon','mkdir -- '+shlex.quote(str(run))],check=True)
    for name in ['input','audit','reports','results','private']:(run/name).mkdir()
    protocol['private_run']=str(run)
    write(run/'input/protocol.json',protocol)
    write(run/'audit/source_hashes.json',sources)
    write(run/'audit/reproduction.json',dict(original_run=str(original),original_protocol_sha256=sha(public/'frozen_protocol.json')))
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        CRFOREST_RSCRIPT='/home/oliver/MainDiskData/financial_diagnosis/csf_joint_sensitivity_v1/environment/r/bin/Rscript',
        EW_RLIB='/home/oliver/MainDiskData/ernet_original_tau0_full_20261015/dynamic_competing_h24_20261015/rsf_er_competing_20261016/environment/rlib')
    with (run/'audit/tests_before_fitting.log').open('x') as log:
        subprocess.run([sys.executable,'-m','pytest','-q'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    for action in ['freeze','pilot','run']:
        with (run/f'audit/entry_{action}.log').open('x') as log:
            subprocess.run([sys.executable,'-m','experiments.post_st_feature_combination_search_v1.pipeline',
                            '--run',str(run),'--action',action],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f'Fitting completed and preserved at {run}; run analyze/validate/deliver for tables and report.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    execute(parser.parse_args().run)
