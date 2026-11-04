import os,sys,io,tarfile,subprocess,shutil,json
from pathlib import Path
from experiments.post_st_eswa_baselines_v1.common import sha,load,dump
D=Path(__file__).resolve().parent;ROOT=D.parents[1]
def run():return Path(os.environ['STABILITY_RUN'])
def schedule():
    r=run();source=Path(os.environ['ESWA_RUN']);inp=r/'input';inp.mkdir();code=inp/'code';code.mkdir()
    audit=load(ROOT/'experiments/post_st_eswa_baselines_v1/DATA_ALIGNMENT_AUDIT.json')['hashes']
    for y in [2023,2024,2025]:
        (inp/str(y)).mkdir()
        for role in ('train','eval'):
            p=source/'input'/str(y)/(role+'.npz');assert sha(p)==audit[f'{y}/{role}'];shutil.copyfile(p,inp/str(y)/p.name)
    (inp/'deps_location.txt').write_text(str(source/'deps'))
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    files=[p for p in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines() if p.endswith('.py') and p.startswith(('crforest/','experiments/'))]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',rev,'--',*files],cwd=ROOT))) as f:f.extractall(code,filter='data')
    items=[]
    # Interleave heavy and light annual tasks across the heterogeneous worker queue.
    for rep in range(1,51):
        for y in (2025,2023,2024):
            for h in (24,12):
                for model in ('Stacking','PROBoost'):
                    items.append(dict(kind='cpu',argv=['python3',str(code/'experiments'/D.name/'launch.py'),str(y),str(h),str(rep),model],cwd=str(code),timeout=3600,storage=dict(source=str(inp),mode='nfs',results=str(r/'published'))))
    assert len(items)==600; (r/'tasks.jsonl').write_text(''.join(json.dumps(i)+'\n' for i in items))
    (r/'replicates.py').write_text('# independent model, training year, horizon and replicate\n')
    dump(r/'replicates.py.hetero.json',dict(independent=True,manifest='tasks.jsonl'));dump(r/'snapshot.json',dict(commit=rev,inputs=audit,manifest_sha256=sha(r/'tasks.jsonl')))
    subprocess.run([str(Path.home()/'.local/bin/hetero-run'),str(r/'replicates.py')],check=True)
if __name__=='__main__':schedule()
