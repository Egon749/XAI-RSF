"""Audit/copy only historical inputs, then freeze before permutation outcomes."""
from pathlib import Path
from dataclasses import asdict
import argparse, datetime, io, json, shutil, subprocess, tarfile
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest, CIFExpansion
from experiments.post_st_ablation_v1.common import configuration
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_leaf_shallow_v1.explain_company import LABELS
from experiments.post_st_prognosis_v1.common import calendar_elapsed
from .common import *


def protocol():
    return dict(name=NAME, status='FROZEN_BEFORE_PERMUTATION', outer_cutoffs=[2023,2024],
        rolling_entry_years=[2019,2020,2021,2022,2023], seeds=SEEDS,
        model='unchanged q2/depth4/shared_quantile/balanced25/direct-raw-parent-AJ10',
        forest_fits=0, calibration_fits=0, repeats=128, company_bootstrap=1000,
        random_seed=20261031, shards_per_cohort=8, main_horizon_months=24, secondary_horizon_months=12,
        labels='Only saved training labels stopped strictly before each outer January 1. No 2023/2024 test labels; 2023 entries added from pre-2024 TRAINING snapshot only.',
        prediction='For cohort v reuse forests fitted before v. Three seed forecasts averaged before metrics; no in-sample feature screening.',
        time_support='Endpoint unsupported if nobody has follow-up >=H and some outcomes unknown; report NA, never zero loss. Mature entire entry-year cohorts primary; partial cohorts descriptive only.',
        permutation='128 common within-entry-year company donor permutations; move NaN with value; audit76:82 indicators move together. Same draws for single/group/pair; labels/times never shuffled. Direct forest predictions score all permutations (preserve exact score ties); independently check algebraic lookup to1e-10.',
        correlation='Training-only pairwise-complete Spearman abs(rho)>=.95, >=30 overlap; connected groups plus audit-source block. Incomplete correlation does not certify conditional independence.',
        interaction='All native structural pairs: RMS/main effects and joint empirical support; joint pair permutation moves pair with SAME donor. Exact component removal is a SCORE-only diagnostic, not a valid CIF or retrained ablation.',
        bootstrap='1000 common multinomial company counts within entry year; all128 fixed permutations averaged analytically for AUC/Brier effects. Fixed IPCW weights; primary complete horizons weights1. No training/selection uncertainty or multiplicity guarantee.',
        classifications=dict(retain='Provisional: >=2 distinct mature informative cohorts (training cause>=1, validation cause>=3, controls>=5), positive delta_AUC>=.005 in >=2, equal-cohort mean delta_AUC>=.01 and paired-bootstrap 95% lower>0. Target first, recovery safeguard explicitly labeled; linked audit block evidence cannot identify individual contributions. Native interaction score check may support retention separately.',
            redundant='Suspected only: training abs Spearman>=.95 with same partner in >=3 available rolling folds, >=30 finite overlap each; keep higher observed-coverage representative (tie original index); do not mark a retained variable redundant. Exact constants/duplicates separately flagged. No automatic deletion.',
            noise='Requires >=2 mature informative target cohorts with >=10 training target events, upper95 delta_AUC<.01 and upper95 delta_Brier<.001, no retained group/interaction protection. Low power never counts as noise.',
            insufficient='All other variables; absence of use/effect is not absence of a true signal.'),
        current_reference='preserved; no variable deletion, no policy adoption',
        publish='Names, aggregate scores, classifications, hashes, code only. No company IDs, forecasts, true cuts or model coefficients.',
        evidence='PROVISIONAL historical diagnostic after prior research exposure; inherited annual-vintage PIT limitations remain.')


def prepare(run, repo):
    for n in ['input','audit','reports','results','private']:(run/n).mkdir(exist_ok=False)
    p=protocol();write(run/'input/protocol.json',p)
    public=repo/'experiments'/NAME;write(public/'frozen_protocol.json',p)
    sources={}; counts=[]; checks=[]; reused=[]; feature_quality=[]; edges_all=[]
    def copy(src,dst):
        sources[str(src)]=sha(src);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    metas={}; outer_train={}
    for outer in p['outer_cutoffs']:
        src=HISTORY/'input/folds'/str(outer)/'train_PRIVATE.npz';sources[str(src)]=sha(src);outer_train[outer]=arrays(src)
        src=HISTORY/'evaluation_input'/f'{outer}_fit_metadata_PRIVATE.parquet';sources[str(src)]=sha(src)
        metas[outer]=pd.read_parquet(src)
        np.testing.assert_array_equal(metas[outer].sample_id,outer_train[outer]['sample_ids'])
        assert (metas[outer].stop<pd.Timestamp(f'{outer}-01-01')).all()
    src=HISTORY/'input/folds/2023/plan.json';sources[str(src)]=sha(src)
    names=json.loads(src.read_text())['feature_names'];assert len(names)==83
    write(run/'input/features.json',[dict(index=j,name=n,label=LABELS[n]) for j,n in enumerate(names)])
    CompetingRiskForest.fit=prohibit_fit
    for year in p['rolling_entry_years']:
        folder=run/'input/folds'/str(year);folder.mkdir(parents=True)
        src=(OLD/f'input/validation/{year}' if year<2023 else HISTORY/'input/folds/2023')
        for name in ['train_PRIVATE.npz','eval_PRIVATE.npz']:copy(src/name,folder/name)
        tr,ev=arrays(folder/'train_PRIVATE.npz'),arrays(folder/'eval_PRIVATE.npz')
        assert tr['X'].shape[1]==83 and len(np.unique(tr['groups']))==len(tr['X'])
        members=[]
        for seed in SEEDS:
            src=ROOT/'input'/('rolling' if year<2023 else 'reference')/str(year)/str(seed)
            dest=folder/'models'/str(seed)
            for n in ['forest_PRIVATE.json','predictions_PRIVATE.npz']:copy(src/n,dest/n)
            f=CompetingRiskForest.load(dest/'forest_PRIVATE.json')
            assert f.config==configuration(seed,'shallow_parent')
            for tree in f.trees_:
                for node in tree:
                    if 'cif' in node:
                        assert node['leaf_shrinkage']['strength']==10
            saved=arrays(dest/'predictions_PRIVATE.npz');np.testing.assert_array_equal(saved['sample_ids'],ev['sample_ids'])
            np.testing.assert_allclose(f.predict(ev['X'],saved['times']),saved['cif'],atol=1e-13,rtol=0)
            reused.append(dict(cohort=year,seed=seed,forest_sha256=sha(dest/'forest_PRIVATE.json'),input_sha256=sha(folder/'train_PRIVATE.npz')))
            members.append(f)
        f=merge(members);ex=CIFExpansion.from_forest(f,tr['X']);ex.save(folder/'functions_PRIVATE.npz')
        X=np.vstack([tr['X'],ev['X']]);err=0.;component_error=0.;minval=0.;maxsum=0.;mindiff=0.
        for start in range(0,len(X),64):
            xx=X[start:start+64];native=f.predict(xx);represented=ex.predict(xx)
            err=max(err,float(abs(native-represented).max()));minval=min(minval,float(native.min()));maxsum=max(maxsum,float(native.sum(-1).max()));mindiff=min(mindiff,float(np.diff(native,axis=1).min(initial=0)))
            total=np.broadcast_to(ex.mu,native.shape).copy()
            for key in [(j,) for j in ex.mains]+ex.pairs:total+=ex.component(xx,key)
            component_error=max(component_error,float(abs(total-native).max()))
        assert max(err,component_error)<1e-10 and minval>=-1e-12 and maxsum<=1+1e-12 and mindiff>=-1e-12
        base=f.predict(ev['X'],TIMES);np.savez_compressed(folder/'baseline_PRIVATE.npz',pred=base)
        corr,edges,groups=groups_from_training(tr['X']);np.save(folder/'correlation_PRIVATE.npy',corr)
        for j,k,r,n in edges:edges_all.append(dict(cohort=year,j=j,k=k,name_j=names[j],name_k=names[k],rho=r,overlap=n))
        for j,n in enumerate(names):
            z=tr['X'][:,j];feature_quality.append(dict(cohort=year,feature=j,name=n,label=LABELS[n],missing_rate=float(np.isnan(z).mean()),distinct_states=len(np.unique(z)),
                constant=bool(len(np.unique(z))==1),duplicates=','.join(str(k) for k in range(j) if np.array_equal(z,tr['X'][:,k],equal_nan=True)),
                split_trees=sum(any(node.get('feature')==j for node in tree) for tree in f.trees_)))
        sets={tuple(logical_closure(g)) for g in groups};sets.add(tuple(range(76,83)))
        items=[dict(kind='single',key=[j],moved=logical_closure([j])) for j in range(83)]
        items += [dict(kind='group',key=list(g),moved=list(g)) for g in sorted(sets)]
        items += [dict(kind='pair',key=list(g),moved=logical_closure(g)) for g in ex.pairs]
        write(folder/'items.json',items)
        permutations=np.stack([np.random.default_rng(np.random.SeedSequence([p['random_seed'],year,r,11])).permutation(len(ev['X'])) for r in range(p['repeats'])])
        bootstrap=np.random.default_rng(np.random.SeedSequence([p['random_seed'],year,22])).multinomial(len(ev['X']),np.full(len(ev['X']),1/len(ev['X'])),size=p['company_bootstrap'])
        np.savez_compressed(folder/'draws_PRIVATE.npz',permutations=permutations,bootstrap=bootstrap)
        for outer in p['outer_cutoffs']:
            if year>=outer:continue
            m=metas[outer];full=outer_train[outer];use=(m.origin.dt.year==year).to_numpy()
            label={k:v[use] for k,v in full.items() if k!='X'}
            np.testing.assert_array_equal(label['sample_ids'],ev['sample_ids']);np.testing.assert_array_equal(full['X'][use],ev['X'])
            assert not set(tr['groups'])&set(label['groups'])
            # Independently replay the prior-cohort fitting cutoff from the
            # already censored outer snapshot; never expose a later event.
            fit=(m.origin.dt.year<year).to_numpy();cap=pd.Timestamp(f'{year}-01-01')-pd.Timedelta(days=1)
            stop=m.stop.where(m.stop<=cap,cap);event=full['event'].copy();event[(m.stop>cap).to_numpy()]=0
            np.testing.assert_array_equal(full['X'][fit],tr['X']);np.testing.assert_array_equal(event[fit],tr['event'])
            np.testing.assert_array_equal((stop[fit]-m.origin[fit]).dt.days,tr['time'])
            for field in ['financial_available','market_available']:
                assert (m.loc[use,field].isna() | (m.loc[use,field]<=m.loc[use,'origin'])).all()
            np.savez_compressed(folder/f'labels_{outer}_PRIVATE.npz',**label)
            for h in [12,24]:
                mature=all(o+pd.DateOffset(months=h)<pd.Timestamp(f'{outer}-01-01') for o in m.origin[use])
                y,w,known,supported,diag=evaluation_context(label,h,1)
                counts.append(dict(outer_cutoff=outer,cohort=year,horizon=h,train_companies=len(tr['X']),train_termination=int((tr['event']==1).sum()),train_recovery=int((tr['event']==2).sum()),
                    validation_companies=len(y),termination=int((y&known).sum()),recovery=int(((label['event']==2)&(label['analysis_time']<=h)&known).sum()),unknown=int((~known).sum()),
                    at_horizon=int((label['analysis_time']>=h).sum()),max_followup_months=float(label['analysis_time'].max()),mature=mature,supported=supported,
                    fit_cutoff_exclusive=f'{year}-01-01',outcome_cutoff_exclusive=f'{outer}-01-01'))
        checks.append(dict(cohort=year,forests=3,full_reconstruction_error=err,component_sum_error=component_error,min_cif=minval,max_sum_cif=maxsum,min_time_increment=mindiff,items=len(items)))
    for name,rows in [('cohort_support',counts),('model_validation',checks),('artifact_reuse',reused),('feature_quality',feature_quality),('correlation_edges',edges_all)]:
        pd.DataFrame(rows).to_csv(run/f'reports/{name}.csv',index=False)
    write(run/'audit/source_hashes.json',sources)
    write(run/'audit/preparation.json',dict(status='COMPLETE',new_fits=0,reused_forests=len(reused),reference_sha256=sha(repo/'CURRENT_POST_ST_MODEL.json'),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    assert sha(repo/'CURRENT_POST_ST_MODEL.json')==REFERENCE_SHA
    print(pd.DataFrame(counts).to_string(index=False))


def freeze(run,repo):
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    data=subprocess.check_output(['git','archive',commit],cwd=repo)
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:tf.extractall(run/'input/code',filter='data')
    tasks=[];p=json.loads((run/'input/protocol.json').read_text())
    for year in p['rolling_entry_years']:
        for shard in range(p['shards_per_cohort']):
            tasks.append(dict(kind='cpu',argv=['python3',str(run/'input/code/experiments'/NAME/'launch.py'),'--cohort',str(year),'--shard',str(shard)],
                cwd=str(run/'input/code'),timeout=3600,storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results/formal'))))
    with (run/'formal.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
    (run/'formal.py').write_text('# Disjoint saved-model permutation batches. No fitting.\n')
    write(run/'formal.py.hetero.json',dict(independent=True,manifest='formal.jsonl'))
    write(run/'audit/freeze.json',dict(commit=commit,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_sha256=sha(run/'formal.jsonl'),tasks=len(tasks),protocol_sha256=sha(run/'input/protocol.json')))
    write(run/'audit/immutable_input_hashes.json',{str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()})
    print(json.dumps(dict(tasks=len(tasks),commit=commit)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['prepare','freeze'],required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2]
    prepare(a.run,repo) if a.stage=='prepare' else freeze(a.run,repo)
