"""Freeze the finite protocol and run independent complete pipelines."""
import argparse,datetime,io,subprocess,tarfile,shutil
from pathlib import Path
from dataclasses import asdict
from experiments.post_st_group_selection_v1.run import PARENT,PREVIOUS,SEARCH,REPO
from experiments.post_st_group_selection_v2.run import RUN as V2
from experiments.post_st_dropcolumn_selection_v1.common import load,write,sha,configuration
from experiments.post_st_dropcolumn_selection_h24_v1 import pipeline as pool_api
NAME='post_st_group_selection_v3'
RUN=Path('/home/oliver/MainDiskData/xai_rsf_group_selection_v3_20261101_DfdNL6cc')
PUBLIC=REPO/'experiments'/NAME


def prepare():
    for d in ['input','audit','reports','results']:(RUN/d).mkdir(exist_ok=True)
    sources=load(V2/'audit/source_hashes.json')
    for p,h in sources.items():assert sha(Path(p))==h
    for y in [2023,2024,2025]:shutil.copy2(V2/f'input/{y}_metadata_PRIVATE.parquet',RUN/f'input/{y}_metadata_PRIVATE.parquet')
    write(RUN/'audit/source_hashes.json',sources)
    protocol=dict(name=NAME,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_private_run=str(PARENT),private_run=str(RUN),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        fixed_configuration=asdict(configuration(42,83)),seeds=[42,101,202],parent_AJ_strength=10,
        data='Unchanged conservativeC83; Jan1 information cutoffs2023/24/25; next trading-day pre-open origin; first final termination=1 vs first recovery=2; calendar24mo fitting cap. Original financial-vintage PIT remains PROVISIONAL. All discovery labels are censored before each cutoff; outer labels read only after decisions sealed.',
        candidates='Same finite v2 scheme: latest3 pre-cutoff historical training windows,complete-link stable |Spearman|>=.95,>=30 unique overlaps,audit76:82 protected; full83 plus each group reduced to one/two representatives. Recompute for every company draw. No recursion,Cartesian extension or external-result tuning.',
        coverage='Anchor balanced25 pools on ORIGINAL83IDs for every seed/tree. Keep every retained eligible variable; fill ONLY removed slots,prefer retained group representatives,then independent SeedSequence([seed,tree,2026110210]) permutation. Exactly25/tree,mtry75 unchanged. No bootstrap/split RNG consumption. Unaffected trees must exactly match saved83 raw trees. This is anchored repair,not balanced sampling over82.',
        discovery='>=2 mature H24 forward cohorts,total>=10 unique terminations; each train>=1termination,val>=3targets,>=5controls. Insufficient discovery returns83. No independent confirmation requirement for exploration.',
        periods='Report ALL mature/supported historical cohorts with fittingtrain>=1termination. For EACH cohort,horizon12/24,cause1/2: AUC/AP loss<=.02 and Top20 countloss<=1 when >=3 unique positives,>=5controls; otherwise rank metrics descriptive/NA. Every estimable period Brier increase<=.002 termination or .005 recovery. Require strictly fewer meanH24 spatial cells. No annual averaging of these guards.',
        protected_group='Expand affected redundancy group by one-hop same-feature_group proxies |Spearman|>=.8,>=30 unique co-observed companies using only that inner training sample. No transitive expansion. Aggregate mains+ALL incident pairs,internal pairs once. Same original fold training product-marginal reference; empirical-joint contribution centered for comparison. Constant offsets allowed; calibration protected by per-period Brier.',
        qualified_anchor='For each historicalH24 fold,bothcauses,12/24: baseline group centeredRMS>=.001probability(0.1pp),group mean absolute contribution / sum all component mean absolute contributions>=.01,median3 pairwise seed correlations>=.7(undefined=0),>=5unique fitting cause events. All required. Three seeds measure algorithm stability only.',
        group_guard='Only qualified anchors veto: centered group RMS difference<=max(.5*baselineRMS,baseline between-seed RMS dispersion,.001). Other anchors recorded as insufficient stable-important evidence,not as proof of no information. Allow main/pair and correlated-variable redistribution; all component diagnostics retained. No per-pair veto in primary policy.',
        selection='Choose passing candidate with fewest meanH24cells,thenfewestfields,thenIDs;else83. Thresholds engineering safeguards,not statistical equivalence or causal truth.',
        matched_comparators='Point2024/25: same candidates,per-period evaluator and fixed data with uncontrolled subset-balanced coverage(reuse verifiedv2 forests) versus anchored coverage(newfits); apply old pair guard and new group guard to each. Primary newgroup+anchored final forests; other policy choices are diagnostic. Historical83/74 fixed outer references unchanged. Oldv2 mean-guard results separately contextual,not same-policy comparator.',
        stability='50 company resamples per cutoff stratified entry year,SeedSequence([2026110191,year,repeat]);same draws asv2. Recompute groups,support,all candidate comparisons,groupqualification,selection and actual3seed final refit. Reuse hash-verified full83 discovery controls where available; missing controls fitted. Anchored candidates newly fit. Report conditional search and unconditional fallback separately; do not relabel reuse as fit.',
        budget='153 pipelines=3point+150resampled. Pilotpoint2024 andbootstrap2025repeat0,then151remaining. No early candidate truncation or automatic search expansion. 4GiB singlethread/task on existing oliver/lemon Slurm NFS. Pointcoverage controls reused when valid; count all extra/missing-control fits. Computational conditions differ from old runs.',
        verification='Every raw/shrunk/96tree model all train/eval rows/native times/twocauses,fullq2,reload<=1e-10,nonnegative/nondecreasing,sum<=1,no clipping. Pair bootstrap hashes; unchanged eligible trees exact; output/publication SHA256. Keep failures.',
        evaluation='After fixed decisions:2023/24/25,12/24,cause1/2,mean3seedbeforemetrics,pooled recomputed,Top20 annualceil then sum.2025H24 unsupportedNA.1000 common year-stratified fixed-model bootstrap(seed2026110192);descriptive conditional intervals. Exposed outer cohorts exploratory. Original83/74/currentpointer remain unchanged.',
        nonclaims='No global optimum,no safe deletion proof from weak support,no independent test,no identification/causal claim; exact reconstruction describes fitted model,not true economic interaction recovery.')
    write(PUBLIC/'frozen_protocol.json',protocol);write(RUN/'input/protocol.json',protocol);write(PUBLIC/'run_location.json',dict(private_run=str(RUN)))


def freeze():
    assert not subprocess.check_output(['git','status','--porcelain']).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();files=[p for p in subprocess.check_output(['git','ls-files'],text=True).splitlines() if p.endswith('.py')]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'--',*files]))) as f:f.extractall(RUN/'input/code',filter='data')
    write(RUN/'audit/code_snapshot.json',dict(commit=commit,code_dir='code',protocol_sha256=sha(RUN/'input/protocol.json')))


def execute(stage):
    pool_api.NAME=NAME
    allitems=[dict(kind='pipeline',year=y,repeat=i) for y in [2023,2024,2025] for i in [-1]+list(range(50))]
    pilots=[dict(kind='pipeline',year=2024,repeat=-1),dict(kind='pipeline',year=2025,repeat=0)]
    pool_api.pool(RUN,stage,pilots if stage=='pilot' else [r for r in allitems if r not in pilots])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze','pilot','batch']);a=p.parse_args()
    {'prepare':prepare,'freeze':freeze,'pilot':lambda:execute('pilot'),'batch':lambda:execute('batch')}[a.action]()
