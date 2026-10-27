"""Freeze inputs, 2x2 configuration contrasts, seeds and practical acceptance gates."""
from pathlib import Path
from dataclasses import asdict,replace
import argparse,hashlib,json,shutil,subprocess,tarfile
from crforest import ForestConfig


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)
def main(run,repo,original):
    run,repo,original=map(Path,(run,repo,original));run.mkdir(exist_ok=False)
    inp=run/'input';inp.mkdir();(run/'audit').mkdir();(run/'reports').mkdir();(run/'private').mkdir()
    old=original/'input';prior=json.loads((old/'real_config.json').read_text());expected=json.loads((original/'audit/input_hashes.json').read_text())
    for folder in ['V19_train','V19_eval','V20_train','V20_eval','legacy']:
        shutil.copytree(old/folder,inp/folder)
    for fold in ['V19','V20']:
        for name in [f'{fold}_train_PRIVATE.npz',f'{fold}_eval_PRIVATE.npz',f'{fold}_M2_PRIVATE.npz']:
            shutil.copy2(old/name,inp/name)
    hashes={str(p.relative_to(inp)):sha(p) for p in inp.rglob('*') if p.is_file()}
    assert all(expected[k]==v for k,v in hashes.items())
    configs={};variants=['unrestricted','local_q2','shared_q2','local_budget','shared_budget'];seeds=[42,101,202]
    for fold in ['V19','V20']:
        for seed in seeds:
            base=ForestConfig(**prior['real_forest'],seed=seed)
            for v in variants:
                c=base if v=='unrestricted' else replace(base,path_budget=2)
                if v.startswith('shared'):c=replace(c,cut_strategy='shared_quantile')
                if v.endswith('budget'):c=replace(c,global_cell_budget=1000)
                configs[f'{fold}/{seed}/{v}']=asdict(c)
    plan=dict(status='FROZEN_PROVISIONAL',original_run=str(original),source_counts=prior['source_counts'],
              folds=['V19','V20'],seeds=seeds,variants=variants,configs=configs,formal_fits=30,pilot_fits=5,
              training_tasks=156,assembly_tasks=18,tree_shard_size=4,tuning_trials=0,
              cut_rule='5 observed-value empirical quantiles j/6, j=1..5, on each entire training fold; exclude maximum; filter to node observed range; separate native missing-only candidate',
              partition_rule='whole forest unmerged main/pair spatial cells including missing state; exact monotone ledger; 1000 cap; zero-cost reuse remains admissible',
              traversal='tree index then recursive left-before-right; budget couples structures and may privilege earlier trees; no tree-order optimization',
              contrasts=[['shared_only','shared_q2','local_q2'],['budget_local','local_budget','local_q2'],['budget_shared','shared_budget','shared_q2'],['combined','shared_budget','local_q2']],
              acceptance_gates=dict(scope='descriptive paired means over 3 seeds, each V19/V20/pooled; not a statistical noninferiority claim',
                  ap_retention_vs_unrestricted=.95,brier_ratio_vs_unrestricted=1.05,max_mean_top20_events_lost_vs_unrestricted=3,
                  spatial_cells_max=1000,reconstruction_tolerance=1e-10,all_failures_reported=True),
              selection='No winner-driven extra fits, budgets, thresholds or seeds. Report every fixed contrast.',
              resources='existing two-node Slurm pool in physical mode; 24 distinct oliver physical cores and 8 lemon cores if scheduler allows; one thread per task, 6GiB process cap; tree shards for uncoupled fits',
              stopping_rule='one frozen round; no second-stage search',
              data_status='Same PROVISIONAL PIT/quality/origin/follow-up limitations; 2022-2024 unused',
              source_hashes=hashes,code_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip())
    write(inp/'plan.json',plan);write(run/'audit/frozen_protocol.json',plan)
    public=repo/'experiments/shared_partition_v1/frozen_protocol.json'
    if not public.exists():write(public,plan)
    with (run/'audit/code.tar').open('xb') as f:subprocess.run(['git','-C',str(repo),'archive',plan['code_commit']],stdout=f,check=True)
    with tarfile.open(run/'audit/code.tar') as t:t.extractall(inp/'code',filter='data')
    def task(fold,seed,v,mode,stage,extra=(),estimate=0):
        return dict(kind='cpu',argv=['python3',str(inp/'code/experiments/shared_partition_v1/launch.py'),'--fold',fold,'--seed',str(seed),'--variant',v,'--mode',mode,*extra],
                    cwd=str(inp/'code'),timeout=7200,estimated_seconds=estimate,storage=dict(source=str(inp),mode='auto',results=str(run/'results'/stage)))
    training=[]
    for fold in plan['folds']:
        for seed in seeds:
            for v in variants:
                if v.endswith('budget'):training.append(task(fold,seed,v,'fit','training',estimate=600))
                else:
                    for i in range(0,32,4):training.append(task(fold,seed,v,'shard','training',['--tree-start',str(i)],70 if v=='unrestricted' else 35))
    pilot=[task('V19',42,v,'fit','pilot',['--pilot']) for v in variants]
    for name,tasks in [('training',training),('pilot',pilot)]:
        (run/f'{name}.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
        (run/f'{name}.py').write_text('# Independent task manifest; submit using physical two-node pool.\n')
        write(run/f'{name}.py.hetero.json',dict(independent=True,manifest=f'{name}.jsonl'))
    write(run/'audit/input_hashes.json',{str(p.relative_to(inp)):sha(p) for p in inp.rglob('*') if p.is_file()})
    print(json.dumps(dict(run=str(run),formal_fits=30,training_tasks=len(training),pilot_fits=len(pilot),code=plan['code_commit'])))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--repo',required=True);p.add_argument('--original',required=True);a=p.parse_args();main(a.run,a.repo,a.original)
