"""Reconcile training budgets, feature opportunities, precision and authentic bootstrap pairs."""
from dataclasses import asdict
from .common import *
def diagnostics():
    r=run();cx=[];cost=[];usage=[];leaf=[];paired={};spec=features()
    def add_usage(df,meta):
        usage.extend(df.assign(**meta).to_dict('records'))
    def add_leaf(df,meta):
        leaf.append(meta|dict(leaves=len(df),median_companies=df.unique_companies.median(),p10_companies=df.unique_companies.quantile(.1),
                        median_target=df.target_companies.median(),zero_target_leaves=int((df.target_companies==0).sum()),
                        zero_recovery_leaves=int((df.recovery_companies==0).sum())))
    def compare_boot(y,s,data):
        val=[(a['bootstrap_sha256'],a['multiplicity_sha256']) for a in data]
        assert len(val)==32 and val==paired.setdefault((y,s),val)
    for stage in ['fit','ensemble']:
        for rec in read(r/stage/'published.json'):
            p=Path(rec['path']);z=read(p/'summary.json')
            key={k:z[k] for k in ['year','arm','variant','seed']};key['provenance']='new_'+stage
            for v,d in z['checks'].items():
                cx.append(key|dict(version=v,**d,forest_sha256=sha(p/v/'forest_PRIVATE.json'),function_sha256=sha(p/v/'functions_PRIVATE.npz'),prediction_sha256=sha(p/v/'predictions_PRIVATE.npz')))
            cost.append(key|{k:z.get(k) for k in ['forest_fits','fit_seconds','seconds','scored','attempted','guard_rejected','host']})
            if stage=='fit':
                compare_boot(z['year'],z['seed'],read(p/'bootstrap.json'));add_usage(pd.read_csv(p/'usage.csv'),key);add_leaf(pd.read_csv(p/'leaves_PRIVATE.csv'),key)
    nnew=len(cx);assert nnew==126
    old=read(r/'anchor_paths_PRIVATE.json');combocx=pd.read_csv(ROOT/'experiments/post_st_seed_explanation_stability_v1/RECONSTRUCTION_DETAIL.csv').query("version=='shrunk' and model=='C039'").set_index('year')
    for y in YEARS:
        for s in SEEDS:
            p=Path(old[f'{y}|{s}']).parent;key=dict(year=y,seed=s,arm='M3',variant='shared_both',provenance='saved_member')
            compare_boot(y,s,read(p/'pairing_PRIVATE.json'))
            df=pd.read_csv(p/'feature_coverage_PRIVATE.csv');df['name']=spec.english_name.to_numpy();df['info']=spec['info'].to_numpy()
            add_usage(df,key);add_leaf(pd.read_csv(p/'leaf_support_PRIVATE.csv'),key)
            for v in ['raw','shrunk']:
                cx.append(key|dict(version=v,**read(p/v/'verification.json'),forest_sha256=sha(p/v/'forest_PRIVATE.json'),function_sha256=sha(p/v/'functions_PRIVATE.npz'),prediction_sha256=sha(p/v/'predictions_PRIVATE.npz')))
            nodes=pd.read_csv(p/'node_search_PRIVATE.csv');cost.append(key|dict(forest_fits=0,scored=nodes.scored_candidates.sum(),attempted=read(p/'shrunk/verification.json')['candidate_evaluations'],guard_rejected=nodes.guard_rejected.sum()))
        z=combocx.loc[y].to_dict()
        cx.append(dict(year=y,seed=0,arm='M3',variant='shared_both',provenance='saved_C039_export',**z,
                       full_reconstruction=z['reconstruction'],function_reload=z['reload'],explicit_component_sum=z['explicit_sum'],max_path_variables=z['max_path'],minimum_increment=z['min_increment'],probability_sum_max=z['max_sum'],prediction_sha256=sha(Path(old[f'{y}|0'])/'predictions_PRIVATE.npz')))
    table=pd.DataFrame(cx);new=table[table.provenance.str.startswith('new')]
    cols=['full_reconstruction','explicit_component_sum','function_reload','forest_reload'];err=new[cols].max().to_dict();assert max(err.values())<=1e-10
    dump(D/'RECONSTRUCTION_AUDIT.json',dict(new_forest_fits=54,new_ensemble_exports=18,new_representations=nnew,anchor_members=9,anchor_annual_ensembles=3,errors=err,all_rows_all_native_times_both_causes=True,
              max_path=int(table.max_path_variables.max()),min_cif=float(table.probability_min.min()),max_sum=float(table.probability_sum_max.max()),min_increment=float(table.minimum_increment.min()),bootstrap_identity_hashes_match=True))
    table.to_csv(D/'INTERPRETABILITY_COMPLEXITY.csv',index=False)
    pd.DataFrame(usage).to_csv(D/'FEATURE_COVERAGE.csv',index=False);u=pd.DataFrame(usage)
    u.groupby(['year','variant','arm','info'])[['eligible_trees','split_nodes','split_trees']].sum().reset_index().rename(columns={'split_trees':'sum_feature_tree_incidences'}).to_csv(D/'GROUP_COVERAGE.csv',index=False)
    pd.DataFrame(leaf).to_csv(D/'LEAF_SUPPORT.csv',index=False);pd.DataFrame(cost).to_csv(D/'COST_AND_SEARCH.csv',index=False)
    cfgrows=[]
    for a in SETS:
        n=int(spec[a].sum())
        for v in ['fixed25','proportional']:
            for s in SEEDS:
                conf=cfg(s,n,v)
                cfgrows.append(dict(arm=a,variant=v,features=n,eligibility_fraction=conf.tree_feature_count/n,source='reused' if a=='M3' else 'new_fit',parent_AJ_strength=10,**asdict(conf)))
    pd.DataFrame(cfgrows).to_csv(D/'MODEL_CONFIG_AUDIT.csv',index=False)
if __name__=='__main__':diagnostics()
