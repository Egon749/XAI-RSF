"""Freeze a single 2x2 search/cut experiment; never access later-year outcomes."""
from pathlib import Path
from dataclasses import asdict,replace
import argparse,hashlib,json,shutil,subprocess,tarfile
from crforest import ForestConfig


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False)
def main(run,repo,original):
    run,repo,original=map(Path,(run,repo,original));run.mkdir(exist_ok=False)
    inp=run/'input';inp.mkdir()
    for d in ['audit','reports','private']:(run/d).mkdir()
    prior=json.loads((original/'input/plan.json').read_text())
    for name,h in prior['source_hashes'].items():
        source=original/'input'/name;assert sha(source)==h
        dest=inp/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
    variants=['unrestricted','baseline','coverage','refined','combined'];seeds=[42,101,202];configs={}
    for fold in ['V19','V20']:
        for seed in seeds:
            for v in variants:
                c=ForestConfig(**prior['configs'][f'{fold}/{seed}/unrestricted'])
                if v!='unrestricted':c=replace(c,path_budget=2,cut_strategy='shared_quantile')
                if v in ['coverage','combined']:c=replace(c,tree_feature_count=25)
                if v in ['refined','combined']:c=replace(c,cut_strategy='shared_refined')
                configs[f'{fold}/{seed}/{v}']=asdict(c)
    plan=dict(status='FROZEN_PROVISIONAL',original_run=str(original),source_counts=prior['source_counts'],folds=['V19','V20'],seeds=seeds,variants=variants,configs=configs,
        formal_fits=30,pilot_fits=5,training_tasks=240,assembly_tasks=30,tree_shard_size=4,tuning_trials=0,
        coverage_rule='25/75 input coordinates per tree; seed-specific shuffled disjoint blocks of three trees, renewed every block; same company bootstrap and node RNG; every input exposed in 10 or 11 of 32 trees; all nodes restricted to tree subset; path order remains two',
        cut_rule='Training observed quantiles: original j/6 anchors UNION j/20 j=1..19 UNION .01 .025 .975 .99 (27 quantile positions before value dedup). At node, restrict dictionary to observed range; if >5 cuts snap five node empirical quantiles j/6 to nearest cut (lower tie), dedup without refill. Uses bootstrap record multiplicities, no validation information.',
        fairness='Same 32 trees, depth6, node support, bootstrap, seeds and <=5 finite candidates per eligible feature. Subspace changes feature eligibility; refined dictionary changes cut locations and deep-node availability. Actual candidate counts and CPU time reported; not equal FLOPs.',
        complexity='No hard forest budget in any arm: prior shared 1000-cell constraint inactive. Record all cells; 1000 is frozen readability evaluation threshold only. No hidden residual/pruning/merging.',
        contrasts=[['coverage_only','coverage','baseline'],['cuts_only','refined','baseline'],['coverage_with_refined','combined','refined'],['cuts_with_coverage','combined','coverage'],['combined','combined','baseline']],
        acceptance_gates=dict(scope='descriptive three paired seed means, separately V19 V20 pooled; no statistical noninferiority claim',auc_gap_vs_unrestricted_max=.02,ap_retention_vs_unrestricted=.95,brier_ratio_vs_unrestricted=1.05,max_mean_top20_events_lost_vs_unrestricted=3,readability_cells_max=1000,reconstruction_tolerance=1e-10),
        resources='physical two-node Slurm pool, oliver24 + lemon8 independent single-thread items, 6GiB per-process limit, NFS, 4-tree shards with fixed global tree IDs',
        stopping_rule='one frozen round; pilot numerical/resource checks only; no outcome-based new configurations, seeds or tree budgets',
        data_status='Inherited PROVISIONAL PIT/quality/origin/followup issues; no 2022-2024 outcomes; original M2 reused only',
        source_hashes=prior['source_hashes'],code_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip())
    write(inp/'plan.json',plan);write(run/'audit/frozen_protocol.json',plan)
    with (run/'audit/code.tar').open('xb') as f:subprocess.run(['git','-C',str(repo),'archive',plan['code_commit']],stdout=f,check=True)
    with tarfile.open(run/'audit/code.tar') as t:t.extractall(inp/'code',filter='data')
    def task(f,s,v,mode,stage,extra=()):
        return dict(kind='cpu',argv=['python3',str(inp/'code/experiments/search_coverage_v1/launch.py'),'--fold',f,'--seed',str(s),'--variant',v,'--mode',mode,*extra],cwd=str(inp/'code'),timeout=7200,storage=dict(source=str(inp),mode='auto',results=str(run/'results'/stage)))
    training=[task(f,s,v,'shard','training',['--tree-start',str(i)]) for f in plan['folds'] for s in seeds for v in variants for i in range(0,32,4)]
    pilot=[task('V19',42,v,'fit','pilot',['--pilot']) for v in variants]
    for name,tasks in [('training',training),('pilot',pilot)]:
        (run/f'{name}.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
        (run/f'{name}.py').write_text('# Run with hetero-run; independent frozen commands.\n')
        write(run/f'{name}.py.hetero.json',dict(independent=True,manifest=f'{name}.jsonl'))
    write(run/'audit/input_hashes.json',{str(p.relative_to(inp)):sha(p) for p in inp.rglob('*') if p.is_file()})
    print(json.dumps(dict(run=str(run),code=plan['code_commit'],formal_fits=30,training_tasks=len(training))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--repo',required=True);p.add_argument('--original',required=True);a=p.parse_args();main(a.run,a.repo,a.original)
