"""Freeze, pilot, and run the actual finite selection pipelines on both nodes."""
import argparse, datetime, io, json, shutil, subprocess, tarfile
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import asdict
from .selection import *
from experiments.post_st_group_selection_v1.run import PARENT, PREVIOUS, SEARCH, REPO
from experiments.post_st_dropcolumn_selection_v1.common import arrays, load, write, sha, configuration
from experiments.post_st_dropcolumn_selection_h24_v1 import pipeline as pool_api
NAME='post_st_group_selection_v2'
RUN=Path('/home/oliver/MainDiskData/xai_rsf_group_selection_v2_20261101_G4bKZ12C')
V1=Path('/home/oliver/MainDiskData/xai_rsf_group_selection_20261101_OTpJKej9')
PUBLIC=REPO/'experiments'/NAME


def prepare():
    for d in ['input','audit','reports','results']:(RUN/d).mkdir(exist_ok=True)
    sources={};gates={};support=[]
    for y in [2023,2024,2025]:
        mp=V1/f'input/{y}_metadata_PRIVATE.parquet';dp=PARENT/f'input/outer/{y}/train_PRIVATE.npz'
        data=arrays(dp);meta=pd.read_parquet(mp);np.testing.assert_array_equal(data['sample_ids'],meta.sample_id.to_numpy(str))
        assert meta.group.is_unique and (meta.stop<pd.Timestamp(f'{y}-01-01')).all()
        shutil.copy2(mp,RUN/f'input/{y}_metadata_PRIVATE.parquet')
        rows=support_at(data,meta,y);support+=rows
        groups,candidates=grouped_candidates(data,meta,y)
        gates[y]=dict(discovery=discovery_gate(rows),confirmation_available=time_gate(data,meta,y)['passed'],groups=groups,candidates=len(candidates))
        for p in [dp,mp,PARENT/f'input/outer/{y}/eval_PRIVATE.npz']:sources[str(p)]=sha(p)
    for p in (PARENT/'input/folds').rglob('*PRIVATE.npz'):sources[str(p)]=sha(p)
    ptr=REPO/'CURRENT_POST_ST_MODEL.json';sources[str(ptr)]=sha(ptr)
    assert sources[str(ptr)]=='aa12ff6d41a515625566cf94f20faa4435d8f9c8e71fdeef8d8faba77fc57c65'
    write(RUN/'audit/source_hashes.json',sources);write(RUN/'audit/point_support.json',gates)
    pd.DataFrame(support).to_csv(PUBLIC/'HISTORICAL_SUPPORT.csv',index=False)
    protocol=dict(name=NAME,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        parent_private_run=str(PARENT),private_run=str(RUN),prior_incomplete_run=str(V1),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        fixed_configuration=asdict(configuration(42,83)),seeds=[42,101,202],parent_AJ_strength=10,
        input='Unchanged old conservativeC83,next exchange trading-day pre-open,first final termination vs first recovery,24-calendar-month fitting cap; PROVISIONAL original financial-vintage evidence.',
        discovery_gate='>=2 mature forward H24 cohorts with total>=10 UNIQUE termination companies; each train>=1target, validation>=3targets and>=5controls. Operational finite-search safeguard, not universal power threshold. Failed discovery returns83.',
        confirmation='No independent confirmation is required to execute exploratory search. Absence limits claims only. No exposed outer labels used to select, adjust guards or reject candidates.',
        candidates='Recompute stable |Spearman|>=.95 complete-link groups in latest3 pre-cutoff historical training windows,>=30 UNIQUE overlapping companies. Fixed one/two representatives chosen by missing coverage and complementarity; audit76:82 protected. Full83 plus one group replacement at a time, no Cartesian or recursive expansion. Historical74 is evaluation reference only.',
        scoring='Actually refit all candidates on eligible chronological training folds; average three seeds before metrics. Mature/supported cohorts with train>=1target,validation>=3targets,controls>=5. Use only labels available strictly before the outer Jan1. Discovery scores are selected-on evidence, not unbiased validation of selection.',
        performance_guards='Annual-mean target AUC/AP losses<=.02 at BOTH12/24; summed annualTop20 captured loss<=1 each; Brier increases<=.002; recoveryBrier<=.005. All required metrics finite; strictly fewer mean spatial cells. Frozen existing thresholds, no equivalence claim.',
        interaction_protection='On each eligibleH24 fold training reference, aggregate affected group/external pair functions and within-group pair block. Both causes and12/24calendar endpoints; baseline RMS>=1e-8 requires ratio[.5,2],correlation>=.7,relativeL2<=.5; negligible base accepted separately; no affected base pairs passes vacuously. Every non-negligible check must pass. This preserves model-function evidence, not causal truth.',
        selection='Among candidates passing all guards select fewest mean cells,then fewest inputs,then stable inputID order; else83. Confirmation and post-hoc resampling frequency do NOT veto the actual exploratory branch. No thresholds changed after outcomes.',
        resampling='50 company resamples stratified by entry year per cutoff,SeedSequence([2026110191,year,repeat]); repeat=-1 is point fit. Recompute correlations/support/groups,fit ALL discovery candidates,score/check interactions,select,then fit final3seed forest for EACH task. Count support on unique companies; duplicate rows retain bootstrap weights in fitting/scoring. Same reference is original historical fold training X for normalized functions.',
        stability='Unconditional retained frequency includes fallback; report conditional selection frequency among actual searches separately. Function moments use common outer TRAIN reference with absent functions zero and original feature IDs. Three forest seeds are algorithm replicates, not separate cohorts.',
        budget='153tasks:3point+150complete company pipelines; each has3final rawforests plus3rawforests per finite candidate/cohort. No candidate skip on unfavorable score. Pilotpoint2024 andbootstrap2025repeat0 reused in153. 4GiB singlethread per task,existing two-node Slurm,NFS. All raw/shrunk/merged functions retained privately.',
        evaluation='After all decisions sealed:2023–2025 exposed exploratory annual12/24bothcauses,3seedmean,pooled recomputed with annualceil20 quota; unsupported2025H24NA. Original83/74 unchanged. 1000paired fixed-prediction company draws stratified by year for descriptive differences,seed2026110192.',
        numerical='Every fitted raw/shrunk/merged model:all train/eval rows,all native times,bothcauses;fullq2reconstruction andsave/reload<=1e-10,CIFmonotonicity/nonnegativity/sum<=1,no clipping. Preserve failures, verify publication hashes. Never replace current model.',
        limits='Exploratory correlated candidate search; no independent confirmation, small historical events, no global optimum claim; both accepted and rejected candidates reported. Added within-group interaction block explicitly extends previous external-partner guard, not result-tuned.')
    write(PUBLIC/'frozen_protocol.json',protocol);write(RUN/'input/protocol.json',protocol)
    write(PUBLIC/'run_location.json',dict(private_run=str(RUN)))
    print(json.dumps(gates))


def freeze():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    files=[p for p in subprocess.check_output(['git','ls-files'],cwd=REPO,text=True).splitlines() if p.endswith('.py')]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'--',*files],cwd=REPO))) as f:f.extractall(RUN/'input/code',filter='data')
    write(RUN/'audit/code_snapshot.json',dict(commit=commit,code_dir='code',protocol_sha256=sha(RUN/'input/protocol.json')))


def execute(stage):
    pool_api.NAME=NAME
    allitems=[dict(kind='full_pipeline',year=y,repeat=i) for y in [2023,2024,2025] for i in [-1]+list(range(50))]
    pilots=[dict(kind='full_pipeline',year=2024,repeat=-1),dict(kind='full_pipeline',year=2025,repeat=0)]
    items=pilots if stage=='pilot' else [r for r in allitems if r not in pilots]
    pool_api.pool(RUN,stage,items)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze','pilot','batch']);a=p.parse_args()
    {'prepare':prepare,'freeze':freeze,'pilot':lambda:execute('pilot'),'batch':lambda:execute('batch')}[a.action]()
