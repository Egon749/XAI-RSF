"""Reproduce into a NEW private directory; never overwrite published evidence."""
from pathlib import Path
import argparse,json,os,shlex,subprocess,sys
from .common import NAME

def execute(run):
    repo=Path(__file__).resolve().parents[2]
    if run.exists():raise ValueError('A new, nonexistent private run directory is required; never silently rerun a fit')
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip():raise ValueError('Commit the intended code before reproduction')
    if not run.is_absolute() or not run.is_relative_to(Path('/home/oliver/MainDiskData')):raise ValueError('Use an absolute new directory under the verified shared research store')
    # Creating through the existing NFS client also avoids the server-side
    # directory-delegation wait observed during the original preparation.
    subprocess.run(['ssh','-o','BatchMode=yes','lemon','mkdir -- '+shlex.quote(str(run))],check=True)
    for n in ['input','evaluation_input','audit','reports','private','results']:(run/n).mkdir()
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        CRFOREST_RSCRIPT='/home/oliver/MainDiskData/financial_diagnosis/csf_joint_sensitivity_v1/environment/r/bin/Rscript',
        EW_RLIB='/home/oliver/MainDiskData/ernet_original_tau0_full_20261015/dynamic_competing_h24_20261015/rsf_er_competing_20261016/environment/rlib')
    def call(module,*args):subprocess.run([sys.executable,'-m',f'experiments.{NAME}.{module}','--run',str(run),*map(str,args)],cwd=repo,env=env,check=True)
    def tests(name):
        with (run/'audit'/name).open('x') as out:subprocess.run([sys.executable,'-m','pytest','-q'],cwd=repo,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
    def pool(stage):
        receipt=None
        with (run/f'audit/launcher_{stage}.log').open('x') as log:
            p=subprocess.Popen(['/home/oliver/.local/bin/hetero-run',str(run/f'{stage}.py')],cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in p.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
                try:
                    value=json.loads(line)
                    if isinstance(value,dict) and 'job' in value and 'result' in value:receipt=value
                except json.JSONDecodeError:pass
            result=p.wait()
        if result or receipt is None or not receipt['success']:raise RuntimeError(f'{stage} failed; inspect preserved logs/publication receipts, do not automatically retrain')
        if stage.startswith('aggregate_'):call('aggregate','--action','gate','--kind',stage.removeprefix('aggregate_'),'--job',receipt['job'],'--report',receipt['result'])
        else:call('stages','--action','gate','--stage',stage,'--job',receipt['job'],'--report',receipt['result'])
    call('prepare');tests('tests_before_freeze.log');call('stages','--action','freeze');pool('pilot')
    for stage in ['inner','bootstrap']:
        call('stages','--action','manifest','--stage',stage);pool(stage)
    for kind in ['inner','bootstrap']:
        call('aggregate','--action','prepare','--kind',kind);pool('aggregate_'+kind)
    call('analyze','--stage','inner');call('stages','--action','manifest','--stage','outer');pool('outer')
    call('aggregate','--action','prepare','--kind','outer');pool('aggregate_outer')
    call('analyze','--stage','outer');call('stability');call('validate');call('plots');tests('tests_final.log')
    print(f'Completed private reproduction: {run}. Existing public reports and current reference were not overwritten.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();execute(a.run)
