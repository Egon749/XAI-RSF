import subprocess,tarfile,io
from .common import *
def main():
    r=run();root=r/'input';code=root/'code';code.mkdir()
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    files=[p for p in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines() if p.endswith('.py') and (p.startswith('crforest/') or p.startswith('experiments/'))]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',rev,'--',*files],cwd=ROOT))) as t:t.extractall(code,filter='data')
    tasks=[(a,y,h,m) for y in YEARS for h in (12,24) for m in ('IPCW','complete') for a in ('Stacking','PROBoost')]
    argv=lambda item:['python3',str(code/'experiments'/D.name/'launch.py'),*map(str,item)]
    items=[dict(argv=argv(item),kind='cpu',cwd=str(code),timeout=3600,storage=dict(source=str(root),mode='nfs',results=str(r/'published'))) for item in tasks]
    (r/'tasks.jsonl').write_text(''.join(json.dumps(i)+'\n' for i in items));(r/'eswa.py').write_text('# independent fixed horizon jobs\n');dump(r/'eswa.py.hetero.json',dict(independent=True,manifest='tasks.jsonl'));dump(r/'snapshot.json',dict(commit=rev,tasks=tasks))
    subprocess.run([str(Path.home()/'.local/bin/hetero-run'),str(r/'eswa.py')],check=True)
if __name__=='__main__':main()
