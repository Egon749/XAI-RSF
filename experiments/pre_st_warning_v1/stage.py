"""Stage immutable inputs and disjoint tasks for the existing two-node pool."""
from pathlib import Path
import argparse,json,shutil,subprocess
from .raw_scan import sha

def successful(run,stage,expected):
    paths=sorted((run/f'results/{stage}').glob('*/task-*/output/summary.json'))
    assert len(paths)==expected,(stage,len(paths),expected)
    values=[json.loads(p.read_text()) for p in paths]
    assert all(v['status']=='COMPLETE' for v in values)
    return [(p.parent,v) for p,v in zip(paths,values)]

def stage(run,which):
    repo=Path(__file__).resolve().parents[2];exp=Path(__file__).resolve().parent
    run.mkdir(exist_ok=True)
    for n in ['audit','input','prepared','evaluation_input','private','reports','results']:(run/n).mkdir(exist_ok=True)
    inp=run/'input'/which;inp.mkdir()
    if which=='raw':
        shutil.copy2(exp/'raw_scan.py',inp/'raw_scan.py')
        raw=Path('/home/oliver/pylib/csmar');files=sorted(p for p in raw.rglob('TRD_Dalyr*') if p.suffix in ['.xlsx','.csv'])+[raw/'盈利能力/FI_T5.xlsx']
        args=[['--source',str(p),'--kind','industry' if p.name=='FI_T5.xlsx' else 'market'] for p in files]
        command='exec "$py" "$DATA_DIR/raw_scan.py" "$@"'
        result_stage='raw_scan'
    else:
        if which=='formal':successful(run,'pilot',1)
        if which=='ensemble':
            members={str(y):[] for y in [2023,2024,2025]}
            found=successful(run,'formal',9)
            assert {(v['year'],v['seed']) for _,v in found}=={(y,s) for y in [2023,2024,2025] for s in [17,503,22]}
            for p,v in sorted(found,key=lambda z:(z[1]['year'],z[1]['seed'])):members[str(v['year'])].append(str(p))
            (inp/'members.json').write_text(json.dumps(members,indent=2))
        code=inp/'code';code.mkdir()
        files=[p for parent in [repo/'crforest',repo/'experiments'] for p in parent.rglob('*.py') if '__pycache__' not in p.parts]
        for p in files:
            target=code/p.relative_to(repo);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        shutil.copytree(run/'prepared/folds',inp/'folds')
        shutil.copy2(exp/'frozen_protocol.json',inp/'frozen_protocol.json')
        shutil.copy2(exp/'feature_schema.json',inp/'feature_schema.json')
        if which=='pilot':args=[['--year','2023','--seed','17','--trees','4']]
        elif which=='formal':args=[['--year',str(y),'--seed',str(s)] for y in [2023,2024,2025] for s in [17,503,22]]
        else:args=[['--ensemble','--year',str(y)] for y in [2023,2024,2025]]
        command='cd "$DATA_DIR/code"\nexec "$py" -m experiments.pre_st_warning_v1.worker "$@"'
        result_stage=which
    lemon_python='/home/lemon/miniforge3/bin/python' if which=='raw' else '/home/lemon/.venvs/xai-warning-20261105/bin/python'
    launcher=inp/'launch.sh';launcher.write_text('#!/bin/bash\nset -euo pipefail\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1\nif [ "$(hostname -s)" = "lemon" ]; then py='+lemon_python+'; else py=/home/oliver/miniforge3/bin/python; fi\n'+command+'\n');launcher.chmod(0o755)
    tasks=[dict(kind='cpu',argv=['bash','{data}/launch.sh']+a,timeout=28800,storage=dict(source=str(inp),mode='auto',results=str(run/'results'/result_stage))) for a in args]
    manifest=run/f'{which}_tasks.jsonl';manifest.write_text(''.join(json.dumps(x)+'\n' for x in tasks))
    (run/f'{which}.py.hetero.json').write_text(json.dumps({'independent':True,'manifest':str(manifest)}))
    hashes={str(p.relative_to(inp)):sha(p) for p in inp.rglob('*') if p.is_file()}
    (run/'audit'/f'{which}_input_hashes.json').write_text(json.dumps(hashes,indent=2))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    (run/'audit'/f'{which}_stage.json').write_text(json.dumps(dict(git_commit=commit,tasks=len(tasks),manifest_sha256=sha(manifest),new_fits=(1 if which=='pilot' else 9 if which=='formal' else 0)),indent=2))
    print(manifest)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['raw','pilot','formal','ensemble'],required=True);a=p.parse_args();stage(a.run,a.stage)
