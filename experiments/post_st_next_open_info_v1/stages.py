from pathlib import Path
import argparse, json, subprocess, tarfile
from experiments.post_st_prognosis_v1.common import sha, write
from experiments.post_st_cause_weights_h24_v1.stages import gate


def freeze(run, repo):
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'], text=True)
    commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
    for p in (run/'input/folds').glob('*/*/plan.json'):
        data = json.loads(p.read_text()); data['code_commit'] = commit; p.write_text(json.dumps(data, indent=2)+'\n')
    with (run/'audit/code.tar').open('xb') as f: subprocess.run(['git','-C',str(repo),'archive',commit], stdout=f, check=True)
    with tarfile.open(run/'audit/code.tar') as f: f.extractall(run/'input/code', filter='data')
    protocol = json.loads((run/'audit/protocol.json').read_text())
    for stage in ['pilot','formal']:
        items = [(y,v,m,s) for y in protocol['test_years']
            for v in (['announcement_fields'] if stage=='pilot' else protocol['variants'])
            for m in protocol['models'] for s in ([42] if stage=='pilot' else protocol['seeds'])]
        tasks = [dict(kind='cpu', argv=['python3',str(run/'input/code/experiments/post_st_next_open_info_v1/launch.py'),
            '--year',str(y),'--variant',v,'--model',m,'--seed',str(s),*(['--pilot'] if stage=='pilot' else [])],
            cwd=str(run/'input/code'), timeout=3600,
            storage=dict(source=str(run/'input'), mode='nfs', results=str(run/'results'/stage))) for y,v,m,s in items]
        (run/f'{stage}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in tasks))
        (run/f'{stage}.py').write_text('# Independent tasks are declared by the hetero sidecar.\n')
        write(run/f'{stage}.py.hetero.json', dict(independent=True, manifest=f'{stage}.jsonl'))
    write(run/'audit/immutable_input_hashes.json', {str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()})
    print(commit)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); p.add_argument('--repo', type=Path)
    p.add_argument('--stage', choices=['freeze','pilot','formal'], required=True); p.add_argument('--job'); a=p.parse_args()
    if a.stage=='freeze': freeze(a.run,a.repo)
    else: gate(a.run,a.stage,a.job)
