"""Stage immutable committed Python source and a finite independent manifest."""
import io,subprocess,tarfile
from .common import *

def main():
    r=run();inp=r/'input';code=inp/'code';code.mkdir()
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    paths=subprocess.check_output(['git','ls-files','crforest/*.py','experiments/**/*.py'],cwd=ROOT,text=True).splitlines()
    archive=subprocess.check_output(['git','archive',rev,'--',*paths],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:tf.extractall(code,filter='data')
    tasks=[]
    for start in STARTS:
        for year in YEARS:
            if (start,year)==(2012,2024):continue  # full-budget pilot retained, never repeated
            tasks.append(dict(argv=['python3',str(code/'experiments'/D.name/'launch.py'),str(year),str(start)],
                cwd=str(code),kind='cpu',timeout=2400,storage=dict(source=str(inp),mode='nfs',results=str(r/'published'))))
    manifest=r/'tasks.jsonl';manifest.write_text(''.join(json.dumps(x)+'\n' for x in tasks))
    write(r/'snapshot.json',dict(worker_commit=rev,tasks=len(tasks),forest_fits_scheduled=len(tasks)*3,
        manifest_sha256=sha(manifest),code_hashes={str(p.relative_to(code)):sha(p) for p in code.rglob('*.py')},
        protocol_sha256=sha(D/'FROZEN_PROTOCOL.json')))
    write(r/'run.py.hetero.json',dict(independent=True,manifest=str(manifest)))
    print(str(manifest),flush=True)

if __name__=='__main__':main()
