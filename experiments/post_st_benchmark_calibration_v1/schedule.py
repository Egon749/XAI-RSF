"""Stage an immutable worker code snapshot, schedule disjoint jobs, verify publication."""
import subprocess,shutil
from .common import *
def main():
    r=run();root=r/'input';code=root/'code';code.mkdir()
    # Depend only on the committed snapshot (includes the R reference entry point).
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    import io,tarfile
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',rev,'--','crforest','experiments',':(exclude,glob)**/*.csv',':(exclude,glob)**/*.png',':(exclude,glob)**/*.pdf'],cwd=ROOT))) as a:a.extractall(code,filter='data')
    tasks=[(a,y,0) for y in YEARS for a in ('RF','FG','CSC','AJ','KNN_AJ')]+[('unrestricted',y,s) for y in YEARS for s in (*PICK,*ANCHOR)]
    items=[dict(argv=['python3',str(code/'experiments'/D.name/'launch.py'),a,str(y),str(s)],kind='cpu',cwd=str(code),timeout=1800,storage=dict(source=str(root),mode='nfs',results=str(r/'published'))) for a,y,s in tasks]
    (r/'tasks.jsonl').write_text(''.join(json.dumps(i)+'\n' for i in items))
    (r/'benchmark.py').write_text('# independent units\n');write(r/'benchmark.py.hetero.json',dict(independent=True,manifest='tasks.jsonl'))
    write(r/'worker_snapshot.json',dict(commit=rev,tasks=tasks))
    subprocess.run([str(Path.home()/'.local/bin/hetero-run'),str(r/'benchmark.py')],check=True)
if __name__=='__main__':main()
