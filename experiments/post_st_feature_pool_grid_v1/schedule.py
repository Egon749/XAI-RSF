"""Stage immutable committed code, one official pilot, then remaining grid."""
import shutil,subprocess
from .common import *
def main():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    r=run();inp=r/'input';code=inp/'code';code.mkdir()
    files=subprocess.check_output(['git','ls-files','-z','*.py'],cwd=ROOT,text=True).rstrip('\0').split('\0')
    for n in files:
        dst=code/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,dst)
    write(inp/'code_identity.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protocol_sha256=sha(D/'FROZEN_PROTOCOL.json'),python_files_sha256={n:sha(code/n) for n in files}))
    tasks=[]
    for y in ALL_YEARS:
        for c in COUNTS:
            tasks.append(dict(kind='cpu',argv=['python3',str(code/'experiments/post_st_feature_pool_grid_v1/launch.py'),str(y),str(c)],
                cwd=str(code),timeout=2400,storage=dict(source=str(inp),mode='nfs',results=str(r/'published'))))
    pilot=[t for t in tasks if t['argv'][-2:]==['2024','25']];assert len(pilot)==1
    batches={'pilot':pilot,'formal':[t for t in tasks if t not in pilot]}
    for n,ts in batches.items():
        (r/f'{n}.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in ts))
        (r/f'{n}.py').write_text('# Independent forest tasks; use hetero-run.\n')
        write(r/f'{n}.py.hetero.json',dict(independent=True,manifest=f'{n}.jsonl'))
    write(r/'input_hashes.json',{str(p.relative_to(inp)):sha(p) for p in inp.rglob('*') if p.is_file()})
    print(json.dumps(dict(pilot=1,remaining=155,forest_fits=468,run=str(r))))
if __name__=='__main__':main()
