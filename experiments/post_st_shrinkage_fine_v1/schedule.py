"""Freeze code; schedule only the missing integer strengths, without fitting."""
import shutil,subprocess
from .common import *

def main():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    r=run();inp=r/'input';code=inp/'code';code.mkdir()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    files=subprocess.check_output(['git','ls-files','-z','*.py'],cwd=ROOT,text=True).rstrip('\0').split('\0')
    for name in files:
        dst=code/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dst)
    write(inp/'code_identity.json',dict(commit=commit,protocol_sha256=sha(D/'FROZEN_PROTOCOL.json'),
        python_files_sha256={n:sha(code/n) for n in files}))
    tasks=[dict(kind='cpu',argv=['python3',str(code/'experiments/post_st_shrinkage_fine_v1/launch.py'),'derive',str(y),str(s)],
        cwd=str(code),timeout=2400,storage=dict(source=str(inp),mode='nfs',results=str(r/'published'))) for y in ALL_YEARS for s in NEW]
    (r/'derive.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
    (r/'derive.py').write_text('# Independent saved-forest derivations; use hetero-run.\n')
    write(r/'derive.py.hetero.json',dict(independent=True,manifest='derive.jsonl'))
    write(r/'input_hashes.json',{str(p.relative_to(inp)):sha(p) for p in inp.rglob('*') if p.is_file()})
    print(json.dumps(dict(tasks=len(tasks),manifest=str(r/'derive.jsonl'))))

if __name__=='__main__':main()
