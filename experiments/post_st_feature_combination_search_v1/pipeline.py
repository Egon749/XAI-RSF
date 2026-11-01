"""Execute immutable, dependency-ordered search stages on the two-node pool."""
import argparse
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tarfile

import pandas as pd

from experiments.post_st_dropcolumn_selection_v1.common import load, sha, write, subset_hash
from experiments.post_st_dropcolumn_selection_h24_v1.selection import Evaluator, gate24, initial_units
from experiments.post_st_dropcolumn_selection_h24_v1.fitting import historical_support
from .search import FULL, anchors, lattice, legal, neighbourhood, centers, decide
from .store import Store

NAME = 'post_st_feature_combination_search_v1'


def snapshot(run, repo):
    if subprocess.check_output(['git','status','--porcelain'], cwd=repo).strip():
        raise ValueError('Commit the tested code and frozen protocol before execution')
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()
    tracked = subprocess.check_output(['git','ls-files'], cwd=repo, text=True).splitlines()
    files = [p for p in tracked if p.endswith('.py')]
    code = run / 'input/code'
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'--',*files], cwd=repo))) as f:
        f.extractall(code, filter='data')
    actual=load(run / 'input/protocol.json')
    expected=load(repo / f'experiments/{NAME}/frozen_protocol.json')
    assert {k:v for k,v in actual.items() if k!='private_run'} == {k:v for k,v in expected.items() if k!='private_run'}
    write(run / 'audit/code_snapshot.json', dict(commit=commit, protocol_sha256=sha(run / 'input/protocol.json'),
                                               code_hashes={str(p.relative_to(code)):sha(p) for p in code.rglob('*.py')}))


def pool(run, stage, items, batch_size=4):
    if not items:
        write(run / f'audit/registry_{stage}.json', [])
        return []
    meta = load(run / 'audit/code_snapshot.json')
    protocol = load(run / 'input/protocol.json')
    root = run / f'input/{stage}'
    root.mkdir()
    batches = [items[i:i+batch_size] for i in range(0,len(items),batch_size)]
    write(root / 'plan.json', dict(parent=protocol['parent_private_run'], code_commit=meta['commit'], batches=batches))
    code = run / 'input/code'
    tasks = [dict(kind='cpu', argv=['python3',str(code / f'experiments/{NAME}/launch.py'),'--index',str(i)],
                  cwd=str(code), timeout=3600,
                  storage=dict(source=str(root), mode='nfs', results=str(run / f'results/{stage}')))
             for i in range(len(batches))]
    with (run / f'{stage}.jsonl').open('x') as f:
        f.write(''.join(json.dumps(t)+'\n' for t in tasks))
    (run / f'{stage}.py').write_text('# Independent bundle batches; see frozen sidecar.\n')
    write(run / f'{stage}.py.hetero.json', dict(independent=True,manifest=f'{stage}.jsonl'))
    receipt = None
    print(json.dumps(dict(stage=stage, tasks=len(tasks), new_forests=3*len(items))), flush=True)
    with (run / f'audit/launcher_{stage}.log').open('x') as log:
        proc = subprocess.Popen(['/home/oliver/.local/bin/hetero-run',str(run / f'{stage}.py')],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line); log.flush()
            print(line, end='', flush=True)
            try:
                obj = json.loads(line)
                if isinstance(obj,dict) and 'job' in obj and 'result' in obj:
                    receipt = obj
            except json.JSONDecodeError:
                pass
        code_return = proc.wait()
    if code_return or receipt is None or not receipt['success']:
        raise RuntimeError(f'{stage}: retain failed evidence; recover publication without rerunning completed fits')
    report = load(Path(receipt['result']))
    assert report['expected'] == len(tasks) and not report['pending']
    records = []
    for state in report['completed']:
        assert state['returncode'] == 0 and state['compute_completed'] and state['publication_completed']
        i = int(state['id'])
        out = run / f'results/{stage}/{receipt["job"]}/task-{i}/output'
        for rel, expected in load(out / 'hashes.json').items():
            assert sha(out / rel) == expected, (out,rel)
        summary = load(out / 'summary.json')
        assert len(summary['records']) == len(batches[i])
        for record in summary['records']:
            records.append(dict(**record, path=str(out / record['relative_path']), output=str(out),
                                host=summary['host'], job=receipt['job'], task=i,
                                task_seconds=summary['seconds'], peak_rss_MiB=summary['peak_rss_MiB']))
    assert len(records) == len(items)
    write(run / f'audit/registry_{stage}.json', records)
    write(run / f'audit/gate_{stage}.json', dict(status='PASS',stage=stage,job=receipt['job'],
                                               tasks=len(tasks),new_forests=3*len(items),
                                               seconds=receipt['execution_seconds'], report=receipt['result'],
                                               report_sha256=sha(Path(receipt['result']))))
    return records


def initial_candidates(parent):
    result = {}
    for year in [2023,2024,2025]:
        keep = set(lattice(year))
        keep.update(tuple(j for j in FULL if j not in u) for u in initial_units(parent,year))
        result[year] = sorted(k for k in keep if legal(k))
    return result


def rows_to_frame(rows):
    data = []
    for row in rows:
        r = dict(row)
        for key in ['keep','failed_guards']:
            if key in r:
                r[key] = json.dumps(r[key])
        data.append(r)
    return pd.DataFrame(data)


def run_search(run):
    protocol = load(run / 'input/protocol.json')
    parent, previous = Path(protocol['parent_private_run']), Path(protocol['previous_h24_run'])
    candidates = initial_candidates(parent)
    groups = load(parent / 'input/group_specs.json')
    histories = {y:historical_support(parent,y)[0] for y in candidates}
    scores = {y:{} for y in candidates}
    provenance = {y:{k:'initial_lattice' if k in set(lattice(y)) else 'existing_diagnostic' for k in v}
                  for y,v in candidates.items()}
    for step in range(3):
        stage = f'search_{step}'
        store = Store(parent,previous,run)
        requests = {}
        for year,keeps in candidates.items():
            ev = Evaluator(parent,year,store,histories[year])
            for r in store.missing(keeps,ev.cohorts[12]+ev.cohorts[24]):
                requests[(r['year'],tuple(r['keep']))] = r
        items = [requests[k] for k in sorted(requests)]
        write(run / f'audit/candidates_{step}.json', {str(y):[list(k) for k in ks] for y,ks in candidates.items()})
        if not (run / f'audit/registry_{stage}.json').exists():
            pool(run,stage,items)
        store = Store(parent,previous,run)
        for year,keeps in candidates.items():
            ev = Evaluator(parent,year,store,histories[year])
            for keep in keeps:
                if keep in scores[year]:
                    continue
                d = ev.contrast(FULL,keep)
                summary = ev.summary(keep)
                d.update(year=year,first_stage=step,provenance=provenance[year][keep],
                         mean_AP24=summary['metrics'][(24,1,'ap')],
                         mean_AP12=summary['metrics'][(12,1,'ap')],
                         mean_AUC24=summary['metrics'][(24,1,'auc')],
                         mean_AUC12=summary['metrics'][(12,1,'auc')],
                         historical_cells=summary['cells'], support_passed=gate24(histories[year])['passed'])
                scores[year][keep] = d
            print(json.dumps(dict(stage=step,year=year,scored=len(scores[year]),
                                  feasible=sum(r['approved'] for r in scores[year].values()))),flush=True)
        rows = [r for ss in scores.values() for r in ss.values()]
        rows_to_frame(rows).to_csv(run / f'reports/CANDIDATE_SCORES_STAGE_{step}.csv',index=False)
        write(run / f'audit/reuse_stage_{step}.json', list(store.used.values()))
        if step < 2:
            chosen = {}
            for year in candidates:
                cc = centers(year,list(scores[year].values()))
                chosen[year] = [list(k) for k in cc]
                new = set()
                for keep in cc:
                    new.update(neighbourhood(keep,[g for g in groups if g['outer']==year]))
                for keep in new:
                    provenance[year].setdefault(keep,f'general_neighbourhood_round_{step+1}')
                candidates[year] = sorted(set(candidates[year]) | new)
            write(run / f'audit/centers_{step+1}.json',chosen)
    decisions = {}
    for year,ss in scores.items():
        d = decide(list(ss.values()),gate24(histories[year]))
        d.update(gate=gate24(histories[year]),candidates_evaluated=len(ss),
                 anchors=[dict(keep=list(k),features=len(k),**{a:ss[k][a] for a in ['mean_AP24','mean_AP12','approved','failed_guards']}) for k in anchors(year)])
        decisions[str(year)] = d
    write(run / 'audit/selection_decisions.json',decisions)
    rows_to_frame([r for ss in scores.values() for r in ss.values()]).to_csv(run / 'reports/CANDIDATE_SCORES.csv',index=False)
    print(json.dumps(dict(selection_complete=True,features={y:len(d['keep']) for y,d in decisions.items()})),flush=True)


def outer_registry(parent,previous,run):
    registry = {}
    for r in load(parent / 'audit/registry_aggregate_outer.json'):
        if r['mode']=='count':
            registry[(r['year'],tuple(r['keep']))] = dict(style='aggregate',path=r['path'],paths=r['paths'],reused=True)
    for base in [previous,run]:
        for file in sorted((base/'audit').glob('registry_*.json')):
            for r in load(file):
                if r.get('kind')=='bundle' and r.get('outer_final') and 'repeat' not in r:
                    location=r['path'] if 'path' in r else str(Path(r['output'])/'bundle')
                    registry[(r['year'],tuple(r['keep']))] = dict(style='bundle',path=location,reused=base!=run)
    return registry


def final(run):
    protocol=load(run/'input/protocol.json');parent=Path(protocol['parent_private_run']);previous=Path(protocol['previous_h24_run'])
    decisions=load(run/'audit/selection_decisions.json')
    registry=outer_registry(parent,previous,run)
    wanted={(int(y),tuple(d['keep'])) for y,d in decisions.items()}
    wanted.add((2023,anchors(2023)[1]))
    items=[dict(kind='bundle',year=y,keep=list(k),outer_final=True) for y,k in sorted(wanted) if (y,k) not in registry]
    pool(run,'final',items)
    registry=outer_registry(parent,previous,run)
    write(run/'audit/final_models.json',{str(y):dict(keep=d['keep'],**registry[(int(y),tuple(d['keep']))]) for y,d in decisions.items()})
    write(run/'audit/historical75_model.json',dict(year=2023,keep=list(anchors(2023)[1]),**registry[(2023,anchors(2023)[1])]))


def bootstrap(run):
    protocol=load(run/'input/protocol.json');decisions=load(run/'audit/selection_decisions.json')
    items=[dict(kind='bundle',year=int(y),keep=d['keep'],outer_final=True,repeat=r)
           for y,d in decisions.items() if tuple(d['keep'])!=FULL for r in range(50)]
    pool(run,'bootstrap',items,batch_size=2)


def main(run,action):
    repo=Path(__file__).resolve().parents[2]
    if action=='freeze':snapshot(run,repo)
    elif action=='pilot':
        protocol=load(run/'input/protocol.json');parent=Path(protocol['parent_private_run']);previous=Path(protocol['previous_h24_run'])
        store=Store(parent,previous,run);keeps=lattice(2025)
        items=[store.missing(keeps,[y])[0] for y in [2020,2021,2022,2023]]
        pool(run,'pilot',items,batch_size=1)
    elif action=='search':run_search(run)
    elif action=='final':final(run)
    elif action=='bootstrap':bootstrap(run)
    elif action=='run':
        run_search(run);final(run);bootstrap(run)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--action',choices=['freeze','pilot','search','final','bootstrap','run'],required=True)
    args=parser.parse_args();main(args.run,args.action)
