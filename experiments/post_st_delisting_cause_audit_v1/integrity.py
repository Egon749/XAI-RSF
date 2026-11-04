"""Audit original saved model files and grouped arrays independently of the legal register."""
from .common import *
from experiments.post_st_grouped_explanation_v1.common import layout,TRIPLES,SEEDS,allocate,mapping
from experiments.post_st_grouped_explanation_v1.grouped import grouped_components
from experiments.post_st_ablation_v1.verification import merge
from crforest import CompetingRiskForest,CIFExpansion
P=ROOT/'experiments/post_st_seed_explanation_stability_v1'
G=ROOT/'experiments/post_st_grouped_explanation_v1'
def sources(y):
    g=get_run(G.name);inp=g/'analysis_input'
    pub=next(x for x in load(g/'analysis/published.json') if x['args'][:3]==['analysis',str(y),'0'])
    return inp,Path(pub['path']),g/'input'/str(y)/'C039.npz'
def main():
    saved=load(P/'MODEL_HASH_AUDIT.json')['before'];audit=[];hlist={}
    assert [SEEDS[j] for j in TRIPLES[38]]==[17,503,22]
    for y in YEARS:
        root=get_run(P.name)/'input';inp,groupdir,cfile=sources(y);models=[];pred=[];tr,ev,lab=layout(inp,y)
        for rel,h in saved['inputs'].items():
            if str(y) in rel:
                assert sha(root/rel)==h
                if (inp/rel).is_file():assert sha(inp/rel)==h
                hlist[rel]=h
        for seed in (17,503,22):
            key=f'{y}/{seed}/shrunk';folder=root/key
            for name,digest in saved['models'][key].items():assert sha(folder/name)==digest;hlist[key+'/'+name]=digest
            j=load(folder/'forest_PRIVATE.json'); cfg=j['config']
            assert all(cfg[k]==v for k,v in dict(seed=seed,n_trees=32,path_budget=2,max_depth=4,split_min_target_groups=2,split_target_cause=1).items()) and j['n_features']==74
            models.append(CompetingRiskForest.load(folder/'forest_PRIVATE.json'))
            z=arrays(folder/'predictions_PRIVATE.npz');np.testing.assert_array_equal(z['sample_ids'],lab['sample_ids']);pred.append(z['cif'])
        rows=pd.read_csv(P/'RECONSTRUCTION_DETAIL.csv');expected=rows.query("model=='C039' and version=='shrunk' and year==@y").iloc[0]
        assert sha(cfile)==expected.function_sha256
        hlist[f'{y}/C039/functions']=sha(cfile)
        f=merge(models);ex=CIFExpansion.load(cfile);X=np.vstack([tr['X'],ev['X']]);t=np.unique(np.r_[0.,ex.grid,z['times']]);direct=f.predict(X,t)
        error=float(np.max(abs(direct-ex.predict(X,t))));end=f.predict(X,z['times']);pred_error=float(np.max(abs(end[len(tr['X']):]-np.mean(pred,axis=0))))
        mu,a,i,c=grouped_components(ex,X,z['times'],mapping());ge=float(np.max(abs(end-mu-c.sum(-1))));detail=0.
        for h in (12,24):
            ids=np.searchsorted(z['times'],np.r_[tr[f'h{h}'],lab[f'h{h}']]);ix=np.arange(len(X))
            for k in (1,2):
                p=groupdir/f'{h}_{k}_grouped_PRIVATE.npz';hlist[f'{y}/{h}/{k}/grouped']=sha(p);q=arrays(p)
                for name,ar in [('A',a),('I',i),('C',c)]:detail=max(detail,float(np.max(abs(q[name][50]-ar[ix,ids,k-1]))))
        assert max(error,pred_error,ge,detail)<1e-10 and direct.min()>=-1e-12 and direct.sum(-1).max()<=1+1e-10 and np.diff(direct,axis=1).min()>=-1e-12
        audit.append(dict(year=y,train=len(tr['X']),evaluation=len(ev['X']),trees=96,inputs=74,native_and_endpoint_times=len(t),max_path=f.audit_['max_path_variables'],reconstruction_error=error,predictions_error=pred_error,group_error=ge,individual_group_error=detail))
    dump(D/'MODEL_INTEGRITY_AUDIT.json',dict(forest_fits=0,calibrator_fits=0,years=audit,hashes=hlist))
if __name__=='__main__':main()
