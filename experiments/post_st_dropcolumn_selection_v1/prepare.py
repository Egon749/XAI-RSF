"""Read original snapshots, audit clocks, define all experiments before fitting."""
from pathlib import Path
from dataclasses import asdict
import argparse,datetime,json,shutil,subprocess
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from experiments.post_st_leaf_shallow_v1.explain_company import LABELS
from experiments.post_st_feature_permutation_v1.common import groups_from_training
from .common import *

def prepare(run,repo):
    sources={};support=[];quality=[];cohorts=[];reuse=[];folds={};outerdata={};metadata_differences=[]
    def source(p):sources[str(p)]=sha(p);return p
    def copy(p,d):source(p);d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d)
    assert sha(source(repo/'CURRENT_POST_ST_MODEL.json'))==REFERENCE_SHA
    names=load(source(BASE/'input/folds/2025/plan.json'))['feature_names'];assert len(names)==83
    write(run/'input/features.json',[dict(feature_id=j,feature_name=n,label=LABELS[n],feature_group=feature_group(j),dtype='float64_with_NaN') for j,n in enumerate(names)])
    metaall=pd.read_parquet(source(BASE/'private/shifted_cohort_PRIVATE.parquet'))
    clock=arrays(source(BASE/'private/financial_availability_PRIVATE.npz'));info=pd.read_parquet(source(BASE/'private/information_alignment_PRIVATE.parquet'))
    parts=[arrays(source(BASE/f'input/folds/2025/{role}_PRIVATE.npz')) for role in ['train','eval']]
    ix=pd.Index(np.concatenate([d['sample_ids'] for d in parts])).get_indexer(metaall.sample_id)
    assert (ix>=0).all() and metaall.group.is_unique
    Xall=np.vstack([d['X'] for d in parts])[ix]
    np.testing.assert_array_equal(clock['sample_ids'],metaall.sample_id);np.testing.assert_array_equal(info.sample_id,metaall.sample_id)
    assert (np.isfinite(Xall[:,:48])<=~np.isnat(clock['dates'])).all()
    assert not ((~np.isnat(clock['dates']))&(clock['dates']>=metaall.origin.to_numpy(dtype='datetime64[D]')[:,None])).any()
    for field in ['financial_available','market_available']:assert (info[field].isna()|(info[field]<metaall.origin)).all()
    assert (metaall.entry_announcement<metaall.origin).all()
    for outer in YEARS:
        root=history(outer);d=arrays(source(root/f'input/folds/{outer}/train_PRIVATE.npz'))
        m=pd.read_parquet(source(root/f'evaluation_input/{outer}_fit_metadata_PRIVATE.parquet'));np.testing.assert_array_equal(m.sample_id,d['sample_ids'])
        assert (m.stop<pd.Timestamp(f'{outer}-01-01')).all() and (m.origin<pd.Timestamp(f'{outer}-01-01')).all()
        np.testing.assert_array_equal(d['X'],Xall[pd.Index(metaall.sample_id).get_indexer(d['sample_ids'])])
        outerdata[outer]=(d,m)
        dest=run/f'input/outer/{outer}';dest.mkdir(parents=True)
        copy(root/f'input/folds/{outer}/train_PRIVATE.npz',dest/'train_PRIVATE.npz')
        copy(root/f'input/folds/{outer}/eval_PRIVATE.npz',dest/'eval_PRIVATE.npz')
        for f in ['test_PRIVATE.npz','test_metadata_PRIVATE.parquet']:copy(root/f'evaluation_input/{outer}_{f}',run/f'evaluation_input/{outer}_{f}')
        ev=arrays(run/f'evaluation_input/{outer}_test_PRIVATE.npz')
        assert not set(d['groups'])&set(ev['groups'])
        cohorts.append(dict(year=outer,training_companies=len(d['X']),training_termination=int((d['event']==1).sum()),training_recovery=int((d['event']==2).sum()),evaluation_companies=len(ev['X']),train_test_group_overlap=0))
        for arm in ['reference83','drop9']:
            for seed,p in references(outer,arm,repo).items():
                for fn in ['forest_PRIVATE.json','predictions_PRIVATE.npz','functions_PRIVATE.npz']:source(p/fn)
                reuse.append(dict(year=outer,arm=arm,seed=seed,path=str(p),forest_sha256=sha(p/'forest_PRIVATE.json')))
        for j,n in enumerate(names):
            z=d['X'][:,j];quality.append(dict(outer=outer,feature_id=j,feature_name=n,feature_group=feature_group(j),n=len(z),missing_fraction=float(np.isnan(z).mean()),finite_unique=len(np.unique(z[np.isfinite(z)])),duplicate_of=','.join(str(k) for k in range(j) if np.array_equal(z,d['X'][:,k],equal_nan=True))))
        for v in range(2019,outer):
            fit=m.origin.dt.year.lt(v).to_numpy();val=m.origin.dt.year.eq(v).to_numpy();cap=pd.Timestamp(f'{v}-01-01')-pd.Timedelta(days=1)
            stop=m.stop.where(m.stop<=cap,cap);events=d['event'].copy();events[(m.stop>cap).to_numpy()]=0
            tr={k:a[fit].copy() for k,a in d.items()};tr['time']=(stop[fit]-m.origin[fit]).dt.days.to_numpy();tr['event']=events[fit]
            tr['analysis_time']=np.array([calendar_elapsed(o,s) for o,s in zip(m.origin[fit],stop[fit])])
            evx=dict(X=d['X'][val],sample_ids=d['sample_ids'][val]);assert tr['time'].min()>0
            assert not set(tr['groups'])&set(d['groups'][val])
            folder=run/f'input/folds/{v}'
            if v not in folds:
                folder.mkdir(parents=True);np.savez_compressed(folder/'train_PRIVATE.npz',**tr);np.savez_compressed(folder/'eval_PRIVATE.npz',**evx)
                folds[v]=(tr,evx)
            else:
                for k,a in tr.items():
                    if k=='chain_gap_present' and not np.array_equal(a,folds[v][0][k]):
                        # A later source reports a subsequent chain gap. Keep the
                        # earliest historical flag; neither fitting nor metrics
                        # consumes this diagnostic. All eligibility/outcomes
                        # and predictors must still match exactly.
                        metadata_differences.append(dict(cohort=v,later_outer=outer,field=k,changed_rows=int((a!=folds[v][0][k]).sum()),action='retain_earliest_snapshot_diagnostic; not used in fit or scoring'))
                    else:np.testing.assert_array_equal(a,folds[v][0][k])
                for k,a in evx.items():np.testing.assert_array_equal(a,folds[v][1][k])
            lab={k:a[val].copy() for k,a in d.items() if k!='X'}
            np.savez_compressed(folder/f'labels_{outer}_PRIVATE.npz',**lab)
            for h in [12,24]:
                y,w,known,ok,diag=evaluation_context(lab,h,1)
                support.append(dict(outer_cutoff=outer,cohort=v,horizon=h,train_companies=len(tr['X']),train_termination=int((tr['event']==1).sum()),train_recovery=int((tr['event']==2).sum()),
                    validation_companies=len(y),termination=int((y&known).sum()),recovery=int(((lab['event']==2)&(lab['analysis_time']<=h)&known).sum()),controls=int((known&~y).sum()),unknown=int((~known).sum()),at_horizon=int((lab['analysis_time']>=h).sum()),
                    mature=bool(((m.origin[val]+pd.DateOffset(months=h))<pd.Timestamp(f'{outer}-01-01')).all()),supported=ok))
    # Stable connected groups are defined from historical TRAINING X only.
    group_specs=[];edges=[];group_sets={};sens=[]
    for outer in YEARS:
        available=[v for v in FOLDS if v<outer]
        X=outerdata[outer][0]['X']
        for threshold in [.90,.95,.99]:
            seen={}
            for v in available:
                _,ee,_=groups_from_training(folds[v][0]['X'],threshold)
                for j,k,r,n in ee:seen.setdefault((j,k),[]).append((v,r,n))
            stable={pair:values for pair,values in seen.items() if len(values)>=3 and all(v in [z[0] for z in values] for v in available[-3:])}
            # Transitive components are reported, never called pairwise cliques.
            parent=list(range(83))
            def find(j):
                while parent[j]!=j:parent[j]=parent[parent[j]];j=parent[j]
                return j
            for (j,k),values in stable.items():
                parent[find(k)]=find(j)
                edges.append(dict(outer=outer,threshold=threshold,j=j,k=k,name_j=names[j],name_k=names[k],windows=len(values),min_abs_rho=min(abs(x[1]) for x in values),min_overlap=min(x[2] for x in values)))
            groups={}
            for j in range(83):groups.setdefault(find(j),[]).append(j)
            gg=[g for g in groups.values() if len(g)>1]
            sens.append(dict(outer=outer,threshold=threshold,groups=len(gg),linked_variables=sum(map(len,gg)),stable_edges=len(stable)))
            if threshold==.95:group_sets[outer]=gg+[list(range(76,83))]
        for members in group_sets[outer]:
            coverage=np.isfinite(X[:,members]).mean(0)
            first=members[int(np.argmax(coverage))]
            c=pd.DataFrame(X[:,members]).corr(method='spearman',min_periods=30).to_numpy()
            remain=[j for j in members if j!=first]
            # Complementarity here is unsupervised, not proven outcome synergy.
            second=min(remain,key=lambda j:(abs(c[members.index(first),members.index(j)]) if np.isfinite(c[members.index(first),members.index(j)]) else -1,-float(np.isfinite(X[:,j]).mean()),j))
            group_specs.append(dict(outer=outer,group_id='g'+subset_hash(members)[:10],members=members,representative_one=[first],representative_two=sorted([first,second]),audit_block=members==list(range(76,83))))
    sets={};arms=[]
    def add(label,keep,outers,mode='count',**tags):
        keep=sorted(keep);sid='s'+subset_hash(keep)[:12]+'_'+mode
        if sid not in sets:sets[sid]=dict(subset_id=sid,keep=keep,mode=mode,subset_sha256=subset_hash(keep),outers=[])
        sets[sid]['outers']=sorted(set(sets[sid]['outers'])|set(outers))
        arms.append(dict(label=label,subset_id=sid,outers=outers,**tags));return sid
    full=add('full83',range(83),YEARS,kind='baseline')
    drop9=[j for j,n in enumerate(names) if n not in DROP_NAMES]
    for mode in ['count','ratio']:add('legacy74',drop9,YEARS,mode,kind='reference')
    for j in range(83):add(f'drop_{j:02d}',[k for k in range(83) if k!=j],YEARS,kind='single',feature=j)
    for repeat in range(5):
        removed=np.random.default_rng(np.random.SeedSequence([ANALYSIS_SEED,repeat,9])).choice(83,9,replace=False)
        for mode in ['count','ratio']:add(f'random9_{repeat}',[j for j in range(83) if j not in removed],YEARS,mode,kind='random9',repeat=repeat)
    for g in group_specs:
        members=g['members'];outer=g['outer']
        choices={'B_drop_all':[], 'C_one':g['representative_one'],'D_two':g['representative_two']}
        # All single representatives are fitted for the training-only E choice.
        choices.update({f'E_candidate_{j}':[j] for j in members})
        for policy,retained in choices.items():
            keep=[j for j in range(83) if j not in members or j in retained]
            for mode in (['count','ratio'] if policy in ['B_drop_all','C_one','D_two'] else ['count']):
                add(g['group_id']+'_'+policy,keep,[outer],mode,kind='group',group_id=g['group_id'],policy=policy,outer=outer)
    gates={str(y):support_gate([r for r in support if r['outer_cutoff']==y]) for y in YEARS}
    for v in FOLDS:
        for seed in SEEDS:
            p=PERM/f'input/folds/{v}/models/{seed}' if v<=2023 else references(2024,'reference83',repo)[seed]
            copy(p/'forest_PRIVATE.json',run/f'input/folds/{v}/reference/{seed}/forest_PRIVATE.json')
            copy(p/'predictions_PRIVATE.npz',run/f'input/folds/{v}/reference/{seed}/predictions_PRIVATE.npz')
    protocol=dict(name=NAME,source_reference_commit=SOURCE_COMMIT,execution_parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        status='FROZEN_BEFORE_NEW_FITS',years=YEARS,inner_years=FOLDS,seeds=SEEDS,reference=asdict(configuration(42,83)),shrinkage=10,
        target='first final termination cause1 versus first full recovery cause2, next-trading-day pre-open, announcement clocks; old conservative C83; train cap24calendar months',
        outer_use='Previously exposed2023-2025 outcomes are descriptive only, not used by selector. 2025 24months unsupported remainsNA.',
        internal_validation='Fit before each inner Jan1; labels censored strictly before outer Jan1. Mature complete cohorts select; partial unsupported endpoints NA. All four horizon/cause combinations reported.',
        support_gate='At EACH12/24-month termination endpoint: >=2 mature cohorts with train target>=1, validation target>=3 and controls>=5; >=10 target events across informative cohorts. Operational conservative minimum, not validated power/equivalence.',
        deletion_rule=dict(ap12_gain_min=.005,capture12_loss_max=0,ap12_lower90_min=-.01,auc24_loss_max=.02,ap24_loss_max=.02,capture24_loss_max=1,brier_loss_max=.002,recovery_brier_loss_max=.005,seed_direction_min=2/3,pair_jaccard_loss_max=.10,spatial_cell_increase_max=0),
        recursive='Full83 scan -> pool up to12 candidates ranked by eligible historical AP12 delta, protected linked groups -> rescore ALL remaining pool deletions after each accepted step, max12 rounds -> local restore/swap of last deleted stable-group members. Stop at support/constraints/no valid gain. Never force checkpoints65/55/45/35.',
        group_rule='Stable |Spearman|>=.95 in >=3 windows including latest3, pairwise finite overlap>=30. Sensitivity .90/.99. Connected components may not be cliques. C=max observed coverage; D=least correlated remaining representative; E supervised choice allowed only after support gate.',
        grouping_semantics='Column subsets preserve all observed values and NaNs; audit states never independently permuted/relabelled. Audit block separately reported.',
        ratio='round(p*25/83), nearest integer half-up; count=min(25,p). For82 columns both equal25, therefore deduplicate identical fits.',
        uncertainty=dict(repeats=500,seed=ANALYSIS_SEED,unit='company within entry-year',fixed_predictions=True,fixed_IPCW=True,limitations='No complete selection/training uncertainty; mature primary weights1; no multiplicity guarantee'),
        resampling=dict(initial=50,expand_to=100,rule='Expand only if support gate passes and last25-vs-first25 selection frequencies differ>0.10; otherwise no expansion. If support fails, run50 company-bootstrap refits of the conservative fallback per outer cutoff, explicitly not evidence of feature-selection stability.',seed=ANALYSIS_SEED+50),
        verification='q<=2; all training/eval rows, all native times, both causes, full exact sum<=1e-10; valid CIFs and save/reload. No clipping, no deleting terms from fitted models.',
        budget='Finite all83 scans, stable groups,5 random9 sets, count/ratio controls; no outcome-driven extra search;32trees/seed; independent single-thread Slurm/NFS tasks4GiB;50fallback bootstrap refits if selection unsupported.',
        current_reference_preserved=True,inherited_status='PROVISIONAL original-vintage/PIT limits; no new vintage certification')
    for n,obj in [('protocol',protocol),('subsets',sets),('arms',arms),('group_specs',group_specs),('support_gates',gates)]:write(run/f'input/{n}.json',obj)
    public_protocol=repo/'experiments'/NAME/'frozen_protocol.json'
    if public_protocol.exists():
        previous=load(public_protocol)
        assert {k:v for k,v in previous.items() if k!='execution_parent_commit'}=={k:v for k,v in protocol.items() if k!='execution_parent_commit'},'Existing scientific protocol differs; use a new experiment version'
    else:write(public_protocol,protocol)
    for n,rows in [('cohort_audit',cohorts),('historical_support',support),('feature_schema_quality',quality),('group_edges',edges),('group_sensitivity',sens),('reused_artifacts',reuse)]:pd.DataFrame(rows).to_csv(run/f'reports/{n}.csv',index=False)
    write(run/'audit/source_hashes.json',sources)
    write(run/'audit/historical_metadata_differences.json',metadata_differences)
    write(run/'audit/preparation.json',dict(status='AUDITED_PROVISIONAL',reference_preserved=True,reference_sha256=REFERENCE_SHA,gates=gates,unique_subsets=len(sets),bootstrap_training_pairable=True,
        environment_incident='Local creation in shared parent waited on NFS directory delegation; created unique directory via authorized lemon NFS client. No data/fit existed in interrupted directory probes.'))
    print(json.dumps(dict(cohorts=cohorts,gates=gates,unique_subsets=len(sets),group_specs=len(group_specs))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();prepare(a.run,Path(__file__).resolve().parents[2])
