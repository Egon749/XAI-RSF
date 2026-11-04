from dataclasses import asdict
from itertools import combinations
from .common import *
def diagnostics():
    r=run();items=[];cost=[];profiles=[];coverage=[];members={};leafrows=[];spec=features()
    for stage in ['fit','ensemble']:
        for z in read(r/stage/'published.json'):
            p=Path(z['path']);s=read(p/'summary.json');y=z['year'];seed=z['seed'];arm=z['arm'];v=z['variant']
            key=dict(year=y,seed=seed,arm=arm,variant=v)
            for ver,profile in s['checks'].items():items.append(key|dict(version=ver,**profile,forest_sha256=sha(p/ver/'forest_PRIVATE.json'),function_sha256=sha(p/ver/'functions_PRIVATE.npz'),prediction_sha256=sha(p/ver/'predictions_PRIVATE.npz')))
            cost.append(key|{k:s.get(k) for k in ['forest_fits','host','seconds','fit_seconds','attempted','scored','guard_rejected','leaf_company_median','leaf_target_median','zero_target_leaves']})
            if stage=='fit':
                q=pd.read_csv(p/'usage.csv');coverage.extend(q.assign(**key).to_dict('records'))
                boot=[b['multiplicity_sha256'] for b in read(p/'bootstrap.json')]
                if (y,seed) in members:assert boot==members[y,seed]
                members[y,seed]=boot
                l=pd.read_csv(p/'leaves_PRIVATE.csv');leafrows.append(key|dict(leaves=len(l),company_p10=float(l.unique_companies.quantile(.1)),company_median=l.unique_companies.median(),zero_target=int((l.target_companies==0).sum()),one_target=int((l.target_companies==1).sum()),zero_recovery=int((l.recovery_companies==0).sum()),median_recovery=l.recovery_companies.median()))
    original=read(source()/'input/old_registry.json')
    for k,p in original.items():
        y,s=map(int,k.split('|'));assert members[y,s]==[v['multiplicity_sha256'] for v in read(Path(p)/'pairing.json')]
    ids=pd.DataFrame(items);ids.to_csv(D/'INTERPRETABILITY_COMPLEXITY.csv',index=False)
    for name,rows in [('COST_AND_SEARCH',cost),('FEATURE_COVERAGE',coverage),('LEAF_SUPPORT',leafrows)]:
        pd.DataFrame(rows).to_csv(D/(name+'.csv'),index=False)
    cfgrows=[]
    for a in SETS:
        n=int(spec[a].sum())
        for v in variants(a):
            for seed in SEEDS:
                model=asdict(cfg(seed,n,v));cfgrows.append(dict(arm=a,variant=v,variables=n,candidate_fraction=model['tree_feature_count']/n,
                   status='TIME_GATE_NOT_MET' if a in ['M1','M3','M3_numeric'] else 'FITTED',leaf_parent_strength=10,followup_months=24,**model))
    pd.DataFrame(cfgrows).to_csv(D/'MODEL_CONFIG_AUDIT.csv',index=False)
    before=read(r/'sources_PRIVATE.json')
    assert all(sha(Path(name))==h for name,h in before.items())
    errors=['full_reconstruction','explicit_component_sum','forest_reload','function_reload']
    err=ids[errors].max().to_dict();assert max(err.values())<=1e-10
    dump(D/'EXACT_RECONSTRUCTION_AUDIT.json',dict(formal_new_fits=len(read(r/'fit/published.json')),new_ensemble_exports=len(read(r/'ensemble/published.json')),new_representations=len(ids),max_errors=err,max_path=int(ids.max_path_variables.max()),min_cif=float(ids.probability_min.min()),minimum_increment=float(ids.minimum_increment.min()),max_cause_sum=float(ids.probability_sum_max.max()),bootstrap_paired_to_original=True,sources_before_and_after_identical=True,raw_and_shrunk_all_rows_native_times_both_causes=True))
    bygroup=[];fields=[];entry=[]
    for y in YEARS:
        tr=arrays(r/f'input/folds/{y}/train_PRIVATE.npz');X=pd.DataFrame(tr['X'],columns=spec.english_name)
        corr=X.corr(method='spearman',min_periods=10);flags=spec.set_index('english_name')['info']
        for j,k in combinations(X.columns,2):
            if flags[j]==flags[k]:continue
            rho=corr.at[j,k];fields.append(dict(year=y,field1=j,field2=k,group1=flags[j],group2=flags[k],rho=rho,common=int((X[j].notna()&X[k].notna()).sum())))
        for j in flags[flags=='A'].index:
            for state,g in X.groupby('entry_is_star_st'):entry.append(dict(year=y,field=j,entry_star_st=int(state),companies=len(g),prevalence=g[j].mean(),scope='retrospective vendor association, not proof of public timing'))
    pair=pd.DataFrame(fields);pair['pair']=['+'.join(sorted((a,b))) for a,b in zip(pair.group1,pair.group2)]
    for (y,p),g in pair.groupby(['year','pair']):
        good=g.rho.dropna().abs();bygroup.append(dict(year=y,pair=p,pairs=len(g),defined=len(good),median_abs=good.median(),p95_abs=good.quantile(.95),max_abs=good.max()))
    for name,data in [('TRAIN_CROSS_GROUP_CORRELATIONS',fields),('INFORMATION_OVERLAP',bygroup),('AUDIT_ENTRY_TYPE',entry)]:pd.DataFrame(data).to_csv(D/(name+'.csv'),index=False)
if __name__=='__main__':diagnostics()
