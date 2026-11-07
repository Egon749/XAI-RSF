import json,subprocess,tarfile,io
from .common import *

def main():
    r=run();root=r/'input';code=root/'code';code.mkdir()
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    files=[p for p in subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines() if (p.endswith('.py') or p.endswith('FROZEN_PROTOCOL.json') or p.endswith('run_location.json')) and (p.startswith('crforest/') or p.startswith('experiments/'))]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',rev,'--',*files],cwd=ROOT))) as t:t.extractall(code,filter='data')
    tasks=[dict(argv=['python3',str(code/'experiments'/HERE.name/'launch.py'),str(y),str(h)],kind='cpu',cwd=str(code),timeout=600,storage=dict(source=str(root),mode='nfs',results=str(r/'published'))) for y in YEARS for h in (12,24)]
    (r/'tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks));(r/'experiment.py').write_text('# independent fixed-parameter seed reruns\n')
    write(r/'experiment.py.hetero.json',dict(independent=True,manifest='tasks.jsonl'));write(r/'snapshot.json',dict(commit=rev,tasks=6))
    subprocess.run([str(Path.home()/'.local/bin/hetero-run'),str(r/'experiment.py')],check=True)

if __name__=='__main__':main()
