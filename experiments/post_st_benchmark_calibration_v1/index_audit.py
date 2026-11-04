from copy import deepcopy
from itertools import product
import time
import pandas as pd
from crforest import CompetingRiskForest,CIFExpansion
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_feature_permutation_v1.common import evaluation_context
from experiments.post_st_risk_index_v1.index import risk_index,tiers
from .models import stored
from .oob_v2 import oob_predictions,threshold_table,verify_membership
from .common import *

def main():
    r=run();hist=old('post_st_seed_explanation_stability_v1')/'input';g=old('post_st_grouped_explanation_v1')/'input'
    rows=[];checks=[];summaries=[];registry=[];prior=read(ROOT/'experiments/post_st_delisting_cause_audit_v1/MODEL_INTEGRITY_AUDIT.json')['hashes']
    for y in YEARS:
        start=time.perf_counter();out=r/f'risk_index/{y}';out.mkdir(exist_ok=True,parents=True)
        tr=arrays(hist/f'folds/{y}/train_PRIVATE.npz');ev=arrays(hist/f'folds/{y}/eval_PRIVATE.npz');lab=arrays(hist/f'evaluation_input/{y}_test_PRIVATE.npz')
        X=np.vstack([tr['X'],ev['X']]);members=[]
        for s in PICK:
            p=hist/str(y)/str(s)/'shrunk';entry={'test_year':y,'seed':s}
            for t,n in [('forest','forest_PRIVATE.json'),('functions','functions_PRIVATE.npz'),('predictions','predictions_PRIVATE.npz')]:
                hh=sha(p/n);assert hh==prior[f'{y}/{s}/shrunk/{n}'];entry[t+'_sha256']=hh
            registry.append(entry);members.append(CompetingRiskForest.load(p/'forest_PRIVATE.json'))
        merged=merge(members);cp=g/str(y)/'C039.npz';assert sha(cp)==prior[f'{y}/C039/functions']
        ex=CIFExpansion.load(cp);tt=np.unique(np.r_[0.,merged.grid_,DAYS]);direct=np.mean([f.predict(X,tt) for f in members],axis=0)
        predictions=stored(y)[0]['XAI/C039'];np.testing.assert_array_equal(predictions,np.mean([f.predict(ev['X'],DAYS) for f in members],axis=0))
        mu=ex._coeff([365,730])[1];assert (np.isfinite(mu)&(mu>0)).all()
        memb=[verify_membership(f,tr['X'],tr['time'],tr['event'],tr['groups'])[0] for f in members]
        pred,count=oob_predictions(members,tr['X'],tr['groups']);threshold,ok,train_scores=threshold_table(pred,count,mu)
        assert ok.sum()>=30
        np.savez_compressed(out/'oob.npz',pred=pred,count=count,threshold=threshold,mu=mu,train_scores=train_scores)
        ids=ev['sample_ids'];np.testing.assert_array_equal(ids,lab['sample_ids'])
        for u,h in enumerate((12,24)):
            factor=100/mu[u];scaled=deepcopy(ex);scaled.coefficients*=factor;scaled.mu*=factor
            file=out/f'index_{h}.npz';scaled.save(file);ex2=CIFExpansion.load(file);worst=0.;inv=0.
            for off in range(0,len(X),32):
                x=X[off:off+32];orig=direct[off:off+32];total=np.broadcast_to(ex2._coeff(tt)[1],orig.shape).copy()
                for key in [(j,) for j in ex2.mains]+ex2.pairs:total+=ex2.component(x,key,tt)
                zz=ex2.predict(x,tt);worst=max(worst,float(np.abs(total-orig*factor).max()),float(np.abs(zz-orig*factor).max()));inv=max(inv,float(abs(zz/factor-orig).max()))
            assert max(worst,inv)<=1e-10
            pp=endpoints(predictions,lab[f'h{h}']);ii=risk_index(pp,mu[u]);np.savez_compressed(out/f'scores_{h}.npz',score=ii,ids=ids)
            for k in (1,2):
                p=pp[:,k-1];v=ii[:,k-1];cat=tiers(v,threshold[u,k-1]);orders=lambda a:np.sign(a[:,None]-a[None,:])
                assert np.array_equal(orders(p),orders(v))
                summaries.append(dict(year=y,months=h,index_cause=k,n=len(v),q25=np.quantile(v,.25),median=np.median(v),q75=np.quantile(v,.75),maximum=v.max(),over100=np.mean(v>100)))
                checks.append(dict(year=y,months=h,cause=k,forest_hash_match=True,merged_function_hash=sha(cp),ensemble_prediction_sha256=hashlib.sha256(np.ascontiguousarray(predictions).tobytes()).hexdigest(),oob_sha256=sha(out/'oob.npz'),train_firms=len(tr['X']),eligible_firms=int(ok.sum()),min_oob=int(count.min()),max_oob=int(count.max()),nodes_checked=sum(m['nodes_checked'] for m in memb),max_leaf_parent_error=max(m['max_leaf_parent_error'] for m in memb),point_sum_reload_error=worst,inverse_error=inv,rank_tie_identity=True,verified_native_times=len(tt)))
                sizes=[sum(cat==scope) for scope in ['low','medium','high']];maskn=any(0<n<5 for n in sizes)
                for cause in (1,2):
                    yy,w,known,supported,_=evaluation_context(lab,h,cause);events=[int(yy[cat==scope].sum()) for scope in ['low','medium','high']]
                    if supported:assert known.all() and (w==1).all()
                    protect=maskn or not supported or any(0<n<5 for n in events)
                    for j,tier in enumerate(['low','medium','high']):
                        z=sizes[j];observed=events[j]/z if z and not protect else np.nan
                        rows.append(dict(year=y,months=h,index_cause=k,outcome_cause=cause,tier=tier,companies=np.nan if maskn else z,share=np.nan if maskn else z/len(yy),events=np.nan if protect else events[j],observed=observed,lift=observed/yy.mean() if supported and yy.any() else np.nan,prediction_supported=supported,withheld='no follow-up' if not supported else 'small cells and complements' if protect else ''))
        write(out/'index_cost.json',dict(seconds=time.perf_counter()-start))
    pd.DataFrame(rows).to_csv(D/'RISK_INDEX.csv',index=False);pd.DataFrame(checks).to_csv(D/'INDEX_CHECKS.csv',index=False);pd.DataFrame(summaries).to_csv(D/'INDEX_DISTRIBUTION.csv',index=False)
    pointer=dict(status='CURRENT_RESEARCH_REFERENCE',decision_date='2026-11-04',experiment='post_st_seed_explanation_stability_v1',arm='C039',version='shrunk',model='XAI-RSF',configuration=dict(n_inputs=74,seeds=list(PICK),n_trees_per_seed=32,total_trees=96,path_budget=2,max_depth=4,split_min_target_groups=2,split_target_cause=1,direct_parent_AJ_shrinkage_strength=10,main_horizon_calendar_months=12,secondary_horizon_calendar_months=24,training_followup_cap_calendar_months=24),source_baseline_commit='16ba26c66faea63852cd068c1e6f05513c41525e',feature_dictionary='experiments/post_st_grouped_explanation_v1/FEATURE_GROUP_MAPPING.csv',artifact_locator='experiments/post_st_seed_explanation_stability_v1/run_location.json',member_path_pattern='input/{year}/{seed}/shrunk/',prediction_aggregation='equal arithmetic mean of the three saved company CIF arrays before scoring',saved_models=registry,merged_exact_function_hashes={str(y):prior[f'{y}/C039/functions'] for y in YEARS},previous_pointer={'file':'experiments/'+D.name+'/PREVIOUS_POST_ST_MODEL.json','sha256':read(D/'PREVIOUS_POST_ST_MODEL.json')['previous_blob_sha256']},selection='Selected on exposed 12-month within-year termination AUC from 120 triples; exploratory; individual probabilities may underestimate incidence.')
    write(ROOT/'CURRENT_POST_ST_MODEL.json',pointer)
if __name__=='__main__':main()
