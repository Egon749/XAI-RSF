"""Saved-forest diagnostic only. Any attempted forest fit is prohibited."""
from pathlib import Path
import gzip,json,time,shutil,itertools
import numpy as np
import pandas as pd
from crforest import CIFExpansion,CompetingRiskForest
from crforest.partition import tree_feature_subset
from experiments.post_st_dropcolumn_selection_v1.common import load,arrays,write,sha,TIMES,subset_hash,metrics,evaluation_context,horizon_predictions
from experiments.post_st_feature_permutation_v1.common import prohibit_fit,replacement_lookup,score_batch
from experiments.post_st_group_selection_v1.selection import interaction_guard
from experiments.post_st_group_selection_v2.run import PARENT,RUN as SOURCE

ROOT=Path(__file__).resolve().parent
RUN=Path('/home/oliver/MainDiskData/xai_rsf_guard_diagnosis_20261101_LSKWYHGG')
FULL=tuple(range(83));DROPS=[71,60,58];GROUPS={71:[68,71],60:[60,62],58:[55,58]}
CompetingRiskForest.fit=prohibit_fit


def rms(x):return float(np.sqrt(np.mean(np.asarray(x)**2)))
def corr(a,b):
    a=np.asarray(a).ravel();b=np.asarray(b).ravel()
    return float(np.corrcoef(a,b)[0,1]) if a.std()>1e-14 and b.std()>1e-14 else np.nan

def compare(a,b):
    ra=rms(a);rb=rms(b);d=rms(b-a)
    return dict(base_rms_pp=ra*100,new_rms_pp=rb*100,difference_rms_pp=d*100,
        base_mae_pp=float(abs(a).mean()*100),new_mae_pp=float(abs(b).mean()*100),
        amplitude_ratio=rb/ra if ra>1e-14 else np.nan,correlation=corr(a,b),relative_l2=d/ra if ra>1e-14 else np.nan,
        base_mean_pp=float(a.mean()*100),new_mean_pp=float(b.mean()*100))

def horizon(z,hh):return z[np.arange(len(hh)),np.searchsorted(TIMES,hh)]

def decompose(ex,X,keep):
    xx=X[:,keep];p=ex.predict(xx,TIMES);mu=np.broadcast_to(ex._coeff(TIMES)[1],p.shape)
    terms={tuple(int(keep[j]) for j in key):ex.component(xx,key,TIMES) for key in [(j,) for j in ex.mains]+ex.pairs}
    reconstructed=mu+sum(terms.values(),np.zeros_like(p))
    error=float(abs(reconstructed-p).max());assert error<1e-10 and np.isfinite(p).all()
    return dict(pred=p,mu=mu,terms=terms,ex=ex,keep=keep,error=error)

def aggregate(model,ids,which='all'):
    ids=set(ids)
    keys=[k for k in model['terms'] if set(k)&ids and (which=='all' or (which=='main' and len(k)==1) or (which=='pair' and len(k)==2))]
    return sum((model['terms'][k] for k in keys),np.zeros_like(model['pred']))

def block(model,group,partner):
    keys=[k for k in model['terms'] if len(k)==2 and partner in k and set(k)&set(group)]
    return sum((model['terms'][k] for k in keys),np.zeros_like(model['pred']))

def main():
    t0=time.perf_counter();RUN.mkdir(exist_ok=True);(RUN/'reports').mkdir(exist_ok=True)
    source=Path(load(SOURCE/'audit/selection_decisions.json')['2024']['output'])/'discovery'
    features=load(PARENT/'input/features.json');hashes={};models_used=[]
    protocol=dict(mode='NO_TRAINING_SAVED_MODELS_ONLY',cutoff='2024-01-01',source_run=str(SOURCE),source_point_year=2024,
        drops=DROPS,groups=GROUPS,folds=[2020,2021],members=['ensemble','42','101','202'],
        scopes='Original fold training reference and corresponding held-out historical cohort; no2023-2025outerlabels.',
        diagnostics='Reproduce old group/external guards; absolute CIF pp, contribution shares, same-model pair importance and seed noise; group main+pair compensation, market-proxy blocks; tree eligibility/bootstrap audit.',
        absolute_sensitivity_pp=[.01,.05,.1,.5],proxy_correlations=[.8,.9,.95],
        permutation='100 common within-cohort joint-group permutations, seed[2026110201,fold]; retain correlated group members together. PFI is model reliance, not unique or causal information; permutations can leave joint support.',
        decision='Descriptive audit only. Do not change original guards, selected set or current model. No claim of true interaction recovery from observational data.')
    write(RUN/'protocol.json',protocol);write(ROOT/'diagnostic_protocol.json',protocol)
    pairrows=[];aggrows=[];corrows=[];coverrows=[];pfrows=[];response=[];verifications=[];repro=[];component_change=[]
    for fold in [2020,2021]:
        tr=arrays(PARENT/f'input/folds/{fold}/train_PRIVATE.npz');ev=arrays(PARENT/f'input/folds/{fold}/eval_PRIVATE.npz');lab=arrays(PARENT/f'input/folds/{fold}/labels_2024_PRIVATE.npz')
        for name in ['train_PRIVATE.npz','eval_PRIVATE.npz','labels_2024_PRIVATE.npz']:
            p=PARENT/f'input/folds/{fold}/{name}';hashes[str(p)]=sha(p)
        np.testing.assert_array_equal(ev['sample_ids'],lab['sample_ids']);X=np.vstack([tr['X'],ev['X']]);n=len(tr['X']);models={}
        corrmatrix=pd.DataFrame(tr['X']).corr(method='spearman',min_periods=30).to_numpy()
        for drop in [None]+DROPS:
            keep=FULL if drop is None else tuple(j for j in FULL if j!=drop)
            path=source/f'{fold}_{subset_hash(keep)[:16]}';index=load(path/'hashes.json')
            for member in ['ensemble','42','101','202']:
                sub=path/member if member=='ensemble' else path/member/'shrunk'
                p=sub/'functions_PRIVATE.npz';rel=str(p.relative_to(path));assert sha(p)==index[rel];hashes[str(p)]=sha(p)
                ex=CIFExpansion.load(p);assert ex.reference_n==n
                model=decompose(ex,X,keep);models[(drop,member)]=model
                pp=arrays(sub/'predictions_PRIVATE.npz');np.testing.assert_array_equal(pp['sample_ids'],ev['sample_ids'])
                assert abs(pp['cif']-model['pred'][n:]).max()<1e-12
                verifications.append(dict(fold=fold,drop=drop,member=member,reconstruction=model['error'],rows=len(X)))
                models_used.append(dict(fold=fold,drop=drop,member=member,path=str(p),sha256=sha(p)))
            base_mean=np.mean([models[(drop,str(s))]['pred'] for s in [42,101,202]],axis=0)
            assert np.max(abs(base_mean-models[(drop,'ensemble')]['pred']))<1e-12
        base=models[(None,'ensemble')]
        for drop in DROPS:
            keep=models[(drop,'ensemble')]['keep'];group=GROUPS[drop];new=models[(drop,'ensemble')]
            for j in range(83):
                overlap=np.isfinite(tr['X'][:,drop])&np.isfinite(tr['X'][:,j])
                corrows.append(dict(fold=fold,removed=drop,removed_name=features[drop]['label'],other=j,other_name=features[j]['label'],spearman=corrmatrix[drop,j],complete_companies=int(overlap.sum()),same_missing_fraction=float(np.mean(np.isnan(tr['X'][:,drop])==np.isnan(tr['X'][:,j])))))
            expanded={a:sorted(set(group)|{j for j in range(48,72) if any(np.isfinite(corrmatrix[g,j]) and abs(corrmatrix[g,j])>=a for g in group)}) for a in [.8,.9,.95]}
            for seed in [42,101,202]:
                oldpath=source/f'{fold}_{subset_hash(FULL)[:16]}/{seed}';newpath=source/f'{fold}_{subset_hash(keep)[:16]}/{seed}'
                pa,pb=load(oldpath/'pairing.json'),load(newpath/'pairing.json');assert all(a['multiplicity_sha256']==b['multiplicity_sha256'] for a,b in zip(pa,pb))
                oldtree=json.loads(gzip.decompress((oldpath/'shrunk/forest_PRIVATE.json.gz').read_bytes()))['trees']
                newtree=json.loads(gzip.decompress((newpath/'shrunk/forest_PRIVATE.json.gz').read_bytes()))['trees']
                for tid,(aa,bb) in enumerate(zip(oldtree,newtree)):
                    a=set(map(int,tree_feature_subset(83,25,seed,tid)));b={keep[j] for j in tree_feature_subset(82,25,seed,tid)}
                    ra=aa[0].get('feature');rb=bb[0].get('feature');rb=None if rb is None else keep[rb]
                    root_same=(ra==rb and aa[0].get('threshold')==bb[0].get('threshold') and aa[0].get('missing_left')==bb[0].get('missing_left'))
                    coverrows.append(dict(fold=fold,removed=drop,seed=seed,tree=tid,bootstrap_identical=True,eligible_intersection=len(a&b),eligible_same=a==b,removed_was_eligible=drop in a,root_split_same=root_same))
            partners=sorted({next(j for j in k if j not in group) for k in base['terms'] if len(k)==2 and len(set(k)&set(group))==1})
            for scope,sl,hh12,hh24 in [('train',slice(0,n),tr['h12'],tr['h24']),('validation',slice(n,None),lab['h12'],lab['h24'])]:
                for h,hh in [(12,hh12),(24,hh24)]:
                    for cause in [1,2]:
                        extract=lambda z:horizon(z[sl],hh)[:,cause-1]
                        all_pair=sum(float(abs(extract(v)).mean()) for k,v in base['terms'].items() if len(k)==2)
                        all_contrib=sum(float(abs(extract(v)).mean()) for v in base['terms'].values())
                        pair_rms=[rms(extract(v)) for k,v in base['terms'].items() if len(k)==2]
                        group_old={kind:extract(aggregate(base,group,kind)) for kind in ['main','pair','all']}
                        group_new={kind:extract(aggregate(new,group,kind)) for kind in ['main','pair','all']}
                        dm=group_new['main']-group_old['main'];dp=group_new['pair']-group_old['pair']
                        common=dict(fold=fold,removed=drop,scope=scope,horizon=h,cause=cause)
                        for kind in ['main','pair','all']:
                            aggrows.append(dict(**common,component='original_group_'+kind,group=json.dumps(group),**compare(group_old[kind],group_new[kind])))
                        for threshold,ids in expanded.items():
                            aggrows.append(dict(**common,component=f'expanded_group_{threshold}',group=json.dumps(ids),**compare(extract(aggregate(base,ids)),extract(aggregate(new,ids)))))
                        aggrows.append(dict(**common,component='full_prediction',group='all',**compare(extract(base['pred']),extract(new['pred']))))
                        for key,zold,znew in [
                            ('group_main',aggregate(base,group,'main'),aggregate(new,group,'main')),
                            ('group_pair',aggregate(base,group,'pair'),aggregate(new,group,'pair')),
                            ('rest_main',aggregate(base,[j for j in FULL if j not in group],'main'),aggregate(new,[j for j in FULL if j not in group],'main')),
                            ('rest_pair',aggregate(base,FULL,'pair')-aggregate(base,group,'pair'),aggregate(new,FULL,'pair')-aggregate(new,group,'pair')),
                            ('intercept',base['mu'],new['mu'])]:
                            component_change.append(dict(**common,component=key,**compare(extract(zold),extract(znew))))
                        aggrows.append(dict(**common,component='main_pair_compensation',group=json.dumps(group),
                            delta_main_pair_correlation=corr(dm,dp),cancellation_projection=float(-dm@dp/(dp@dp)) if dp@dp>1e-24 else np.nan,
                            remaining_delta_fraction=rms(dm+dp)/rms(dp) if rms(dp)>1e-14 else np.nan))
                        for partner in partners:
                            a=extract(block(base,group,partner));b=extract(block(new,group,partner));guard=interaction_guard(a,b)
                            seeds=np.array([extract(block(models[(None,str(s))],group,partner)) for s in [42,101,202]])
                            sd=rms(seeds-a);presence=int(sum(rms(v)>1e-8 for v in seeds));cs=[corr(seeds[i],seeds[j]) for i,j in itertools.combinations(range(3),2)]
                            finite=[v for v in cs if np.isfinite(v)]
                            pairrows.append(dict(**common,removed_name=features[drop]['label'],group=json.dumps(group),partner=partner,partner_name=features[partner]['label'],
                                **compare(a,b),guard_pass=guard['passed'],guard_reason=guard['reason'],
                                all_pair_mae_share=float(abs(a).mean()/all_pair) if all_pair else np.nan,
                                all_explanation_mae_share=float(abs(a).mean()/all_contrib) if all_contrib else np.nan,
                                pair_rms_percentile=float(np.mean(np.array(pair_rms)<=rms(a))) if pair_rms else np.nan,
                                base_seed_presence=presence,base_seed_mean_correlation=float(np.mean(finite)) if finite else np.nan,
                                base_seed_rms_sd_pp=sd*100,change_over_base_seed_sd=rms(b-a)/sd if sd>1e-14 else np.nan))
            # Exact joint-group permutation; same 100 donor maps for all models.
            rng=np.random.default_rng(np.random.SeedSequence([2026110201,fold]));perms=np.array([rng.permutation(len(lab['event'])) for _ in range(100)])
            deltas={}
            for role,m in [('full83',base),('candidate',new)]:
                ids=[m['keep'].index(j) for j in group if j in m['keep']]
                lut=replacement_lookup(m['ex'],ev['X'][:,m['keep']],ids,TIMES)
                delta=lut[np.arange(len(lab['event']))[None,:],perms];deltas[role]=delta
                p=m['pred'][n:];permuted=p[None,:,:,:]+delta
                assert np.isfinite(permuted).all() and permuted.min()>-1e-12 and permuted.sum(-1).max()<=1+1e-10
                # Independent explicit prediction check for one permutation.
                xx=ev['X'][:,m['keep']].copy();xx[:,ids]=xx[perms[0]][:,ids]
                assert np.max(abs(m['ex'].predict(xx,TIMES)-permuted[0]))<1e-12
                for h in [12,24]:
                    hh=lab[f'h{h}'];idx=np.searchsorted(TIMES,hh);ii=np.arange(len(hh))
                    pr=permuted[:,ii,idx];risk=p[ii,idx]
                    for cause in [1,2]:
                        y,w,known,supported,_=evaluation_context(lab,h,cause);m0=metrics(lab,risk[:,cause-1],h,cause);pm=score_batch(pr[:,:,cause-1],y,w)
                        pfrows.append(dict(fold=fold,removed=drop,model=role,horizon=h,cause=cause,events=m0['target_events'],
                            original_auc=m0['auc'],original_ap=m0['ap'],original_brier=m0['brier'],
                            auc_loss=float(m0['auc']-np.nanmean(pm['auc'])),ap_loss=float(m0['ap']-np.nanmean(pm['ap'])),brier_increase=float(np.mean(pm['brier'])-m0['brier']),
                            permutation_ap_sd=float(np.nanstd(pm['ap'])),undefined_permutations=int(sum(~np.isfinite(pm['ap'])))))
            for h in [12,24]:
                idx=np.searchsorted(TIMES,lab[f'h{h}']);ii=np.arange(len(idx))
                for cause in [1,2]:response.append(dict(fold=fold,removed=drop,horizon=h,cause=cause,**compare(deltas['full83'][:,ii,idx,cause-1],deltas['candidate'][:,ii,idx,cause-1])))
        print('Diagnostic fold complete',fold,flush=True)
    datasets=dict(PAIR_DIAGNOSTICS=pairrows,GROUP_AGGREGATE=aggrows,COMPONENT_CHANGES=component_change,CORRELATIONS=corrows,TREE_ELIGIBILITY=coverrows,GROUP_PERMUTATION=pfrows,GROUP_RESPONSE=response,VERIFICATION=verifications,MODEL_SOURCES=models_used)
    for name,rows in datasets.items():
        pd.DataFrame(rows).to_csv(RUN/'reports'/f'{name}.csv',index=False);shutil.copy2(RUN/'reports'/f'{name}.csv',ROOT/f'{name}.csv')
    for p,h in hashes.items():assert sha(Path(p))==h
    write(RUN/'source_hashes.json',hashes);write(ROOT/'source_hashes.json',hashes)
    receipt=dict(new_fits=0,models_loaded=len(models_used),folds=[2020,2021],outer_years_used=[],source_run=str(SOURCE),seconds=time.perf_counter()-t0,
        maximum_component_reconstruction_error=max(r['reconstruction'] for r in verifications),sources_unchanged=True,fit_api_disabled=True,
        script_sha256=sha(Path(__file__)),output_hashes={p.name:sha(p) for p in (RUN/'reports').glob('*.csv')})
    write(RUN/'execution_receipt.json',receipt);write(ROOT/'execution_receipt.json',receipt);print(receipt,flush=True)

if __name__=='__main__':main()
