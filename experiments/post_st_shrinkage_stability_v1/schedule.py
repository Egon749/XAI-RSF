"""Immutable source, one finite paired draw per Slurm task."""
import io,subprocess,tarfile
from .common import *
def main():
    r=run();inp=r/'input';code=inp/'code';code.mkdir()
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    paths=subprocess.check_output(['git','ls-files','crforest/*.py','experiments/**/*.py'],cwd=ROOT,text=True).splitlines()
    data=subprocess.check_output(['git','archive',rev,'--',*paths],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:tf.extractall(code,filter='data')
    tasks=[]
    for year in YEARS:
        for repeat in range(1,REPEATS+1):
            tasks.append(dict(argv=['python3',str(code/'experiments'/D.name/'launch.py'),str(year),str(repeat)],
              cwd=str(code),kind='cpu',timeout=3600,storage=dict(source=str(inp),mode='nfs',results=str(r/'published'))))
    # The full-budget first draw doubles as the numerical pilot and is never rerun.
    pilot=r/'pilot.jsonl';formal=r/'formal.jsonl'
    pilot.write_text(json.dumps(tasks[0])+'\n')
    formal.write_text(''.join(json.dumps(t)+'\n' for t in tasks[1:]))
    write(r/'run.py.hetero.json',dict(independent=True,manifest=str(formal)))
    write(r/'snapshot.json',dict(worker_commit=rev,pilot_tasks=1,formal_tasks=149,forest_fits=450,
      protocol_sha256=sha(D/'FROZEN_PROTOCOL.json'),
      manifests_sha256={p.name:sha(p) for p in [pilot,formal]},
      input_sha256={str(p.relative_to(inp)):sha(p) for p in inp.rglob('*') if p.is_file()}))
    print(pilot)
if __name__=='__main__':main()

