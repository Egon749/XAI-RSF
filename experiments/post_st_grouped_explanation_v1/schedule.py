import shutil,subprocess,sys
from .common import *
def submit(stage):
    r=run();root=r/('input' if stage=='verify' else 'analysis_input')
    snapshot=root/'code/experiments'/D.name;snapshot.mkdir(parents=True,exist_ok=False)
    for pattern in ['*.py','*.cpp','FROZEN_PROTOCOL.json','FEATURE_GROUP_MAPPING.csv','RANDOM_MAPS.csv']:
        for file in D.glob(pattern):shutil.copy2(file,snapshot/file.name)
    shutil.copytree(REPO/'crforest',root/'code/crforest',ignore=shutil.ignore_patterns('__pycache__'))
    if stage=='analysis':
        subprocess.run(['g++','-std=c++17','-O3','-mpopcnt','-ffp-contract=off','-shared','-fPIC',str(snapshot/'kernel.cpp'),'-o',str(snapshot/'kernel.so')],check=True)
    dump(r/(stage+'_code_hashes.json'),{str(p.relative_to(root)):sha(p) for p in (root/'code').rglob('*') if p.is_file()})
    groups=([(i,min(i+3,12)) for i in range(0,12,3)]+[(i,min(i+6,132)) for i in range(12,132,6)]) if stage=='verify' else [(0,1)]+[(i,min(i+5,201)) for i in range(1,201,5)]
    args=[[stage,str(y),str(a),str(b)] for y in YEARS for a,b in groups]
    target=r/stage;target.mkdir();manifest=target/'tasks.jsonl'
    manifest.write_text(''.join(json.dumps(dict(kind='cpu',argv=['python3',str(snapshot/'launch.py'),*a],cwd=str(root/'code'),timeout=10800,storage=dict(source=str(root),results=str(target/'results'),mode='nfs')))+'\n' for a in args))
    with (target/'launcher.log').open('x') as log:
        subprocess.run(['/home/oliver/.local/bin/hetero-run',str(manifest)],check=True,stdout=log,stderr=subprocess.STDOUT)
    receipt=[json.loads(l) for l in (target/'launcher.log').read_text().splitlines() if l.startswith('{') and '"result"' in l][-1]
    job=read(receipt['result']);assert receipt['success'] and job['expected']==len(args) and not job['pending']
    outputs=[]
    for item in job['completed']:
        i=int(item['id']);p=target/f'results/{receipt["job"]}/task-{i}/output'
        info=read(p/'done.json');assert args[i]==info['args'] and item['compute_completed'] and item['publication_completed']
        outputs.append(info|dict(path=str(p)))
    assert len(outputs)==len(args);dump(target/'published.json',outputs)
if __name__=='__main__':submit(sys.argv[1])
