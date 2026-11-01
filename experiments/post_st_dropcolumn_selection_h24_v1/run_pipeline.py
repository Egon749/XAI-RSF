"""Reproduce privately in a new directory, preserving published results."""
from pathlib import Path
import argparse,os,shlex,subprocess,sys

def execute(run):
    repo=Path(__file__).resolve().parents[2]
    if run.exists() or not run.is_absolute() or not run.is_relative_to(Path('/home/oliver/MainDiskData')):raise ValueError('A new absolute private run under shared MainDiskData is required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip():raise ValueError('Commit the intended version before reproduction')
    subprocess.run(['ssh','-o','BatchMode=yes','lemon','mkdir -- '+shlex.quote(str(run))],check=True)
    for name in ['input','audit','reports','results','private']:(run/name).mkdir()
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        CRFOREST_RSCRIPT='/home/oliver/MainDiskData/financial_diagnosis/csf_joint_sensitivity_v1/environment/r/bin/Rscript',
        EW_RLIB='/home/oliver/MainDiskData/ernet_original_tau0_full_20261015/dynamic_competing_h24_20261015/rsf_er_competing_20261016/environment/rlib')
    def call(module,*args):
        with (run/f'audit/entry_{module}_{"_".join(args).replace("-","") or "default"}.log').open('x') as log:
            subprocess.run([sys.executable,'-m','experiments.post_st_dropcolumn_selection_h24_v1.'+module,'--run',str(run),*args],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (run/'audit/tests_before_fitting.log').open('x') as log:subprocess.run([sys.executable,'-m','pytest','-q'],cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    call('pipeline','--action','freeze');call('initial')
    for action in ['pilot','main','bootstrap','final']:call('pipeline','--action',action)
    call('analyze','--stage','prediction');call('analyze','--stage','stability');call('validate');call('deliver','--action','figures')
    print(f'Private reproduction completed: {run}. Original reports and current model preserved.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();execute(a.run)
