"""Reproducible finite protocol; no changes to historical experiments or forest core."""
import argparse,datetime,io,json,os,shutil,subprocess,tarfile
from pathlib import Path
import numpy as np
import pandas as pd
from .selection import *
from experiments.post_st_dropcolumn_selection_v1.common import load,write,arrays,sha,history,configuration
from experiments.post_st_dropcolumn_selection_h24_v1 import pipeline as pool_api
from experiments.post_st_feature_combination_search_v1.store import Store
from experiments.post_st_dropcolumn_selection_h24_v1.selection import Evaluator
from experiments.post_st_dropcolumn_selection_h24_v1.fitting import historical_support
from experiments.post_st_feature_combination_search_v1.analyze import model_specs
from dataclasses import asdict
NAME='post_st_group_selection_v1'
REPO=Path(__file__).resolve().parents[2]
PARENT=Path('/home/oliver/MainDiskData/xai_rsf_dropcolumn_selection_20261101_EzUsFtiO')
PREVIOUS=Path('/home/oliver/MainDiskData/xai_rsf_dropcolumn_h24_20261101_GdmMpMdx')
SEARCH=Path('/home/oliver/MainDiskData/xai_rsf_feature_combinations_20261101_uido4igz')
RUN=Path('/home/oliver/MainDiskData/xai_rsf_group_selection_20261101_OTpJKej9')
PUBLIC=REPO/'experiments'/NAME

def prepare():
    if (RUN/'input/protocol.json').exists():
        raise FileExistsError('Frozen run exists; use a new unique run directory')
    for folder in ['input','audit','reports','results']:(RUN/folder).mkdir(exist_ok=True)
    sources={};gates={};candidate_sets={};rows=[]
    for y in [2023,2024,2025]:
        dp=PARENT/f'input/outer/{y}/train_PRIVATE.npz';mp=history(y)/f'evaluation_input/{y}_fit_metadata_PRIVATE.parquet'
        data=arrays(dp);meta=pd.read_parquet(mp)
        np.testing.assert_array_equal(data['sample_ids'],meta.sample_id.to_numpy(str))
        assert meta.group.is_unique and (meta.stop<pd.Timestamp(f'{y}-01-01')).all()
        shutil.copy2(mp,RUN/f'input/{y}_metadata_PRIVATE.parquet')
        gates[str(y)]=time_gate(data,meta,y)
        groups,candidates=grouped_candidates(data,meta,y)
        candidate_sets[str(y)]=dict(groups=groups,candidates=[list(k) for k in candidates])
        rows+=support_at(data,meta,y)
        for p in [dp,mp,PARENT/f'input/outer/{y}/eval_PRIVATE.npz']:sources[str(p)]=sha(p)
    for p in (PARENT/'input/folds').glob('*/train_PRIVATE.npz'):sources[str(p)]=sha(p)
    pointer=REPO/'CURRENT_POST_ST_MODEL.json';sources[str(pointer)]=sha(pointer)
    assert sources[str(pointer)]=='aa12ff6d41a515625566cf94f20faa4435d8f9c8e71fdeef8d8faba77fc57c65'
    write(RUN/'audit/source_hashes.json',sources)
    protocol=dict(name=NAME,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=REPO).strip(),
        parent_private_run=str(PARENT),private_run=str(RUN),years=[2023,2024,2025],
        reference=asdict(configuration(42,83)),seeds=[42,101,202],parent_AJ_strength=10,
        input='Old conservative C83, original names/order/NaNs, PROVISIONAL financial-vintage PIT; next trading-day pre-open; announcement first final termination versus first recovery;24-calendar-month training cap.',
        candidates='One-group-at-a-time only; full83 plus one/two representatives in stable Spearman .95 complete-link groups, minimum30 unique paired companies in latest3 historical training windows. No Cartesian search, no supervised representative choice. Audit76:82 never split/deleted. Existing74 is a comparison only, no future-derived 2023 anchor.',
        discovery_gate='At least2 mature H24 forward cohorts, >=10 unique targets total; each train>=1 target, validation>=3 targets and>=5 controls. Operational safeguard, not statistical power.',
        confirmation='At least1 later H24 cohort satisfying the same per-cohort support, with its full selector discovery gate evaluated strictly BEFORE that confirmation cohort starts. Labels mature after this date cannot choose its features. If no honest confirmation, fallback83 BEFORE supervised search; historical candidate scoring remains diagnostic only.',
        protections='For supported future use:12/24 target AUC/AP losses<=.02, captured loss<=1, Brier increase<=.002; recovery Brier<=.005; spatial cells strictly decrease; independent confirmation required. No equivalence claim from non-significance. Never adjust thresholds from exposed outer outcomes.',
        interaction='Diagnostic and tested guard on group-aggregated native pair functions on each cohort training reference, both horizons/causes: amplitude ratio [.5,2], correlation>=.7 and relativeL2<=.5 for non-negligible base components. No deletion from main importance alone; unknown evidence retains full83. Not claimed as verified true interactions.',
        bootstrap='50 independent company resamples stratified by historical entry year per cutoff, seed SeedSequence([2026110191,year,repeat]); rerun group identification and chronological support/confirmation gate, then fit final selected/fallback forest. Gates count UNIQUE companies/events, not duplicates. No supported gate is expected after full-sample audit; unexpected supported branch stops for explicit review, never silently assumes deletion. No claim that blocked searches establish selection stability.',
        budget='150 pipeline tasks,450 new32-tree raw forests plus450 direct-parent derivatives; first task2023 and2025 are pilots reused in150, no discarded seeds. Point83/74 forecasts and historical finite group candidates reused by hash where available; no new unsupported candidate search.',
        evaluation='Only after decisions sealed: same annual3-seed averaged company predictions,12/24 both causes and pooled; annualceil20 quota;2025H24 NA; exposed2023-2025 exploratory. No reference replacement.',
        numerical='Every raw/shrunk/96-tree ensemble: all train/eval rows, all native times,both causes; exact functions and reload<=1e-10; legal CIF; no clipping.',
        unsupported='Actual supervised resampled search, confirmed deletion gains and selection frequencies remain NOT_ESTIMABLE if gate closes; fitting fallback is real computation, not successful feature selection.',
        resources='Two-node existing Slurm NFS, independent single-thread jobs4GiB, affinity managed by pool.',
        preparation_incident='One local mktemp blocked on shared-root lease; stopped only own preparer before fits and created unique directory through lemon. No service/data change.')
    write(PUBLIC/'frozen_protocol.json',protocol);write(RUN/'input/protocol.json',protocol)
    write(RUN/'audit/point_gates.json',gates);write(RUN/'input/candidates.json',candidate_sets)
    pd.DataFrame(rows).to_csv(PUBLIC/'HISTORICAL_SUPPORT.csv',index=False)
    write(PUBLIC/'run_location.json',dict(private_run=str(RUN)))
    print(json.dumps(dict(gates=gates,candidate_counts={y:len(d['candidates']) for y,d in candidate_sets.items()})))

def freeze():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    files=[p for p in subprocess.check_output(['git','ls-files'],cwd=REPO,text=True).splitlines() if p.endswith('.py')]
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'--',*files],cwd=REPO))) as f:f.extractall(RUN/'input/code',filter='data')
    write(RUN/'audit/code_snapshot.json',dict(commit=commit,code_dir='code',protocol_sha256=sha(RUN/'input/protocol.json')))

def execute(stage):
    pool_api.NAME=NAME
    items=[dict(kind='full_pipeline',year=y,repeat=i) for y in [2023,2024,2025] for i in range(50)]
    pilots=[items[0],items[100]]
    selected=pilots if stage=='pilot' else [x for x in items if x not in pilots]
    pool_api.pool(RUN,stage,selected)

def diagnostics():
    # Small predeclared single-group comparisons, never choose from outer results.
    from crforest import CIFExpansion
    from experiments.post_st_dropcolumn_selection_v1.common import TIMES
    source=load(RUN/'input/candidates.json');store=Store(PARENT,PREVIOUS,SEARCH)
    rows=[];interaction=[];reused=[]
    for year in [2023,2024,2025]:
        support,_=historical_support(PARENT,year);ev=Evaluator(PARENT,year,store,support)
        for keep in source[str(year)]['candidates']:
            if tuple(keep)==FULL:continue
            missing=store.missing([keep],set(ev.cohorts[12]+ev.cohorts[24]))
            if missing:rows.append(dict(year=year,keep=json.dumps(keep),status='MISSING_DIAGNOSTIC_ARTIFACT_NO_FIT',missing=len(missing)));continue
            d=ev.contrast(FULL,keep)
            rows.append(dict(year=year,keep=json.dumps(keep),features=len(keep),status='DISCOVERY_ONLY_NOT_CONFIRMED',
                **{k:v for k,v in d.items() if k.startswith('delta_') or k=='spatial_cell_change'},
                failed_guards=json.dumps(performance_guards(d))))
            for cohort in ev.cohorts[24]:
                data=arrays(PARENT/f'input/folds/{cohort}/train_PRIVATE.npz')
                oldreg=store.registry[(cohort,FULL)];newreg=store.registry[(cohort,tuple(keep))]
                old=CIFExpansion.load(Path(oldreg['path'])/'ensemble/functions_PRIVATE.npz')
                new=CIFExpansion.load(Path(newreg['path'])/'ensemble/functions_PRIVATE.npz')
                removed=set(FULL)-set(keep)
                group=next(g['members'] for g in source[str(year)]['groups'] if removed<=set(g['members']))
                # Aggregate all pair terms connecting this group with a given outside variable.
                for partner in range(83):
                    if partner in group:continue
                    pairs=[k for k in old.pairs if partner in k and any(j in group for j in k)]
                    if not pairs:continue
                    a=sum((old.component(data['X'],k,TIMES) for k in pairs))
                    newpairs=[k for k in new.pairs if partner in [keep[j] for j in k] and any(keep[j] in group for j in k)]
                    b=sum((new.component(data['X'][:,keep],k,TIMES) for k in newpairs),np.zeros_like(a))
                    for t,h in [(1,12),(3,24)]:
                        for cause in [1,2]:
                            ix=np.searchsorted(TIMES,data[f'h{h}']); rr=np.arange(len(ix))
                            guard=interaction_guard(a[rr,ix,cause-1],b[rr,ix,cause-1])
                            interaction.append(dict(year=year,cohort=cohort,keep=json.dumps(keep),group=json.dumps(group),partner=partner,horizon=h,cause=cause,**guard,
                                status='MODEL_FUNCTION_DIAGNOSTIC_NOT_TRUE_INTERACTION_PROOF'))
    pd.DataFrame(rows).to_csv(RUN/'reports/GROUP_DISCOVERY_DIAGNOSTICS.csv',index=False)
    pd.DataFrame(interaction).to_csv(RUN/'reports/INTERACTION_PROTECTION_DIAGNOSTICS.csv',index=False)
    write(RUN/'audit/reused_diagnostic_bundles.json',list(store.used.values()))
    decisions={y:dict(keep=list(FULL),status=g['reason'],gate=g) for y,g in load(RUN/'audit/point_gates.json').items()}
    assert all(not d['gate']['passed'] for d in decisions.values())
    write(RUN/'audit/selection_decisions.json',decisions)
    print('Point decisions sealed; discovery rows',len(rows),'interaction checks',len(interaction))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze','pilot','batch','diagnostics']);a=p.parse_args()
    {'prepare':prepare,'freeze':freeze,'pilot':lambda:execute('pilot'),'batch':lambda:execute('batch'),'diagnostics':diagnostics}[a.action]()
