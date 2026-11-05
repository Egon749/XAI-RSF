"""Freeze immutable inputs; independent tasks use the existing two-node pool."""
import argparse,shutil,subprocess
from .common import *

def stage_code(inp):
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    code=inp/'code';code.mkdir()
    files=subprocess.check_output(['git','ls-files','*.py'],cwd=ROOT,text=True).splitlines()
    for name in files:
        dst=code/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dst)
    write(inp/'code_identity.json',dict(commit=commit,python_files_sha256={n:sha(code/n) for n in files},
        protocol_sha256=sha(D/'FROZEN_PROTOCOL.json')))

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','derive']);a=p.parse_args();r=run()
    if a.stage=='fit':
        inp=r/'input';stage_code(inp);items=[(y,s) for y in HISTORICAL for s in SEEDS]
    else:
        paths=list((r/'fit_published').glob('*/task-*/output'));book={}
        for pp in paths:
            s=load(pp/'summary.json');assert s['mode']=='fit' and s['status']=='COMPLETE' and s['forest_fits']==1
            for name,digest in load(pp/'hashes.json').items():assert sha(pp/name)==digest
            key=(s['year'],s['seed']);assert key not in book;book[key]=pp
        assert set(book)=={(y,s) for y in HISTORICAL for s in SEEDS}
        inp=r/'derive_input';shutil.copytree(r/'input',inp)
        for (y,s),pp in book.items():
            dst=inp/f'raw/{y}/{s}';dst.mkdir(parents=True);shutil.copyfile(pp/'forest_PRIVATE.json',dst/'forest_PRIVATE.json')
        items=[(y,s) for y in ALL_YEARS for s in STRENGTHS]
    tasks=[dict(kind='cpu',argv=['python3',str(inp/'code/experiments/post_st_shrinkage_strength_v1/launch.py'),a.stage,str(y),str(v)],
        cwd=str(inp/'code'),timeout=2400,storage=dict(source=str(inp),mode='nfs',results=str(r/f'{a.stage}_published'))) for y,v in items]
    (r/f'{a.stage}.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
    (r/f'{a.stage}.py').write_text('# Independent task manifest; use hetero-run.\n')
    write(r/f'{a.stage}.py.hetero.json',dict(independent=True,manifest=f'{a.stage}.jsonl'))
    write(r/f'{a.stage}_input_hashes.json',{str(p.relative_to(inp)):sha(p) for p in inp.rglob('*') if p.is_file()})
    print(json.dumps(dict(stage=a.stage,tasks=len(tasks),manifest=str(r/f'{a.stage}.jsonl'))))

if __name__=='__main__':main()
