"""Independent annual A or B. B requires the committed run's aggregate A gate."""
from pathlib import Path
from copy import deepcopy
import argparse,json,os,resource,socket,time,traceback
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest,CIFExpansion
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_reporting_drift_v1.analyze import Reader,write_json,digest,block_fit
from experiments.post_st_reporting_drift_v1.metrics import stratified_metrics
from experiments.post_st_calendar_level_v1.analyze import metrics
from experiments.post_st_prognosis_v1.common import horizon_predictions
from experiments.post_st_feature_permutation_v1.common import evaluation_context
from experiments.post_st_risk_index_v1.index import risk_index,inverse_index,positive_denominator,tiers
from .oob import oob_predictions,threshold_table,verify_membership

GROUPS={'financial':'financial','market':'market','linked_audit_opinion':'audit','quality_staleness':'data_quality','entry_warning':'entry_warning'}


def read(root,year,reader,evaluation=True):
    p=root/str(year);tr=reader.arrays(p/'train_PRIVATE.npz');ev=reader.arrays(p/'eval_PRIVATE.npz') if evaluation else None
    fs=[CompetingRiskForest.load(reader.path(p/str(seed)/'forest_PRIVATE.json')) for seed in [42,101,202]]
    return tr,ev,fs


def part_a(root,out,year,reader):
    tr,ev,fs=read(root,year,reader);f=merge(fs);X=np.vstack([tr['X'],ev['X']])
    ex=CIFExpansion.from_forest(f,tr['X']);mu=ex._coeff([365.,730.])[1];positive_denominator(mu)
    np.savez_compressed(out/'denominators_PRIVATE.npz',mu=mu)
    f.save(out/'forest_PRIVATE.json');reloaded=CompetingRiskForest.load(out/'forest_PRIVATE.json')
    times=np.unique(np.r_[0.,f.grid_,365.,366.,730.,731.]);raw=np.mean([member.predict(X,times) for member in fs],axis=0)
    merged_error=float(abs(f.predict(X,times)-raw).max());reload_error=float(abs(reloaded.predict(X,times)-raw).max())
    assert raw.min()>=-1e-12 and raw.sum(-1).max()<=1+1e-10 and np.diff(raw,axis=1).min()>=-1e-12
    ref=reader.arrays(root/str(year)/'reference_PRIVATE.npz')
    np.testing.assert_array_equal(np.mean([member.predict(ev['X'],ref['times']) for member in fs],axis=0),ref['cif'])
    definitions=reader.json(root/'features.json');keys=[(j,) for j in ex.mains]+ex.pairs
    exact=[];ranks=[];summaries=[];groups=[];endpoint={}
    for hi,h in enumerate([12,24]):
        scale=100/mu[hi];scaled=deepcopy(ex);scaled.coefficients*=scale;scaled.mu*=scale
        scaled.save(out/f'index_{h}_PRIVATE.npz');loaded=CIFExpansion.load(out/f'index_{h}_PRIVATE.npz')
        expected=raw*scale;errors=dict(full_reconstruction=0.,explicit_component_sum=0.,function_reload=0.,inverse=0.,forest_reload=reload_error,merged_prediction=merged_error)
        for start in range(0,len(X),32):
            xx=X[start:start+32];p=expected[start:start+32]
            direct=scaled.predict(xx,times);total=np.broadcast_to(scaled._coeff(times)[1],p.shape).copy()
            for key in keys:total+=scaled.component(xx,key,times)
            for name,value in [('full_reconstruction',abs(direct-p).max()),('explicit_component_sum',abs(total-p).max()),('function_reload',abs(loaded.predict(xx,times)-p).max()),('inverse',abs(inverse_index(direct,mu[hi])-raw[start:start+32]).max())]:errors[name]=max(errors[name],float(value))
        assert max(errors.values())<=1e-10,'STOP: real-index full reconstruction failed'
        for kind,d,begin in [('training',tr,0),('evaluation',ev,len(tr['X']))]:
            day=d[f'h{h}'];p=horizon_predictions(raw[begin:begin+len(d['X'])],times,day)
            if kind=='evaluation':p=horizon_predictions(ref['cif'],ref['times'],day)
            score=risk_index(p,mu[hi]);daytimes=np.unique(day);positions=np.searchsorted(daytimes,day)
            base=scaled._coeff(daytimes)[1][positions];component=np.stack([scaled.component(d['X'],key,daytimes)[np.arange(len(day)),positions] for key in keys],axis=1)
            error=float(abs(base+component.sum(1)-score).max());assert error<=1e-10
            np.savez_compressed(out/f'{kind}_{h}_PRIVATE.npz',score=score,cif=p,baseline_points=base,components=component,keys=np.array([','.join(map(str,key)) for key in keys]),sample_ids=d['sample_ids'])
            for k in [1,2]:
                exact.append(dict(year=year,months=h,cause=k,sample=kind,companies=len(day),native_times=len(f.grid_),verified_times=len(times),**errors,calendar_endpoint_error=error,baseline_deviation_min=float((base[:,k-1]-100).min()),baseline_deviation_max=float((base[:,k-1]-100).max()),baseline_deviation_mean=float((base[:,k-1]-100).mean()),mu_positive_finite=True,denominator_file_sha256=digest(out/'denominators_PRIVATE.npz')))
                summaries.append(dict(year=year,months=h,cause=k,sample=kind,companies=len(day),q25=float(np.quantile(score[:,k-1],.25)),median=float(np.median(score[:,k-1])),q75=float(np.quantile(score[:,k-1],.75)),maximum=float(score[:,k-1].max()),proportion_over100=float((score[:,k-1]>100).mean())))
                for group in list(GROUPS.values())+['all_interactions']:
                    ix=[i for i,key in enumerate(keys) if (len(key)==2 if group=='all_interactions' else len(key)==1 and GROUPS[definitions[key[0]]['feature_group']]==group)]
                    values=component[:,ix,k-1]
                    groups.append(dict(year=year,months=h,cause=k,sample=kind,group=group,functions=len(ix),mean_sum_absolute_points=float(abs(values).sum(1).mean()),mean_absolute_group_sum=float(abs(values.sum(1)).mean())))
            if kind=='evaluation':
                endpoint[f'cif_{h}']=p;endpoint[f'index_{h}']=score
                for k in [1,2]:
                    a=p[:,k-1];b=score[:,k-1]
                    order=np.array_equal(np.argsort(a,kind='stable'),np.argsort(b,kind='stable'));ties_equal=np.array_equal(np.diff(np.sort(a))==0,np.diff(np.sort(b))==0)
                    assert order and ties_equal,'STOP: rank/tie identity failed'
                    record=dict(year=year,months=h,cause=k,order_equal=order,ties_equal=ties_equal,supported=not(year==2025 and h==24))
                    if record['supported']:
                        y,w,known,supported,_=evaluation_context(ev,h,k);assert supported and known.all() and (w==1).all()
                        m=metrics(y,a,np.full(len(a),year),ev['sample_ids']);n=metrics(y,b,np.full(len(b),year),ev['sample_ids'])
                        for name in ['pooled_auc','ap','captured','coverage','precision']:
                            np.testing.assert_allclose(m[name],n[name],rtol=0,atol=0,equal_nan=True)
                            record[name+'_reference']=m[name];record[name+'_index']=n[name]
                    ranks.append(record)
    np.savez_compressed(out/'endpoints_PRIVATE.npz',**endpoint)
    for name,rows in [('EXACTNESS_CHECKS',exact),('RANK_IDENTITY_CHECKS',ranks),('INDEX_SUMMARY',summaries),('GROUP_POINTS_SUMMARY',groups)]:pd.DataFrame(rows).to_csv(out/(name+'_PRIVATE.csv'),index=False)
    return dict(part='A',year=year,rows=len(X),full_curve_checks=4,maximum_error=max(max(r[k] for k in ['full_reconstruction','explicit_component_sum','function_reload','inverse','calendar_endpoint_error']) for r in exact))


def part_b(root,out,year,reader):
    gate=reader.json(root/'A_GATE.json');assert gate['status']=='PASS' and gate['years']==[2023,2024,2025]
    tr,ev,fs=read(root,year,reader,evaluation=False)
    # Evaluation data are not read in B; all predictions
    # and quantiles below take only training X/groups and frozen saved forests.
    audit=[];counts=[]
    for f in fs:
        check,count=verify_membership(f,tr['X'],tr['time'],tr['event'],tr['groups']);audit.append(dict(seed=f.config.seed,**check));counts.append(count)
    np.savez_compressed(out/'bootstrap_counts_PRIVATE.npz',counts=np.concatenate(counts))
    mu=reader.arrays(root/'A'/str(year)/'denominators_PRIVATE.npz')['mu']
    pred,count=oob_predictions(fs,tr['X'],tr['groups']);q,eligible,scores=threshold_table(pred,count,mu)
    # Outcome/evaluation poisoning of the surrounding caller state. Neither
    # pure routine receives these fields; all per-company OOB forecasts replay.
    poisoned=dict(tr);rng=np.random.default_rng(2026110204+year)
    poisoned['event']=rng.integers(0,3,len(tr['X']));poisoned['time']=rng.uniform(1,1000,len(tr['X']))
    pp,cc=oob_predictions(fs,poisoned['X'],poisoned['groups']);qq,ee,ss=threshold_table(pp,cc,mu)
    np.testing.assert_array_equal(pred,pp);np.testing.assert_array_equal(q,qq);np.testing.assert_array_equal(eligible,ee)
    np.savez_compressed(out/'oob_PRIVATE.npz',cif=pred,counts=count,eligible=eligible,score=scores,thresholds=q,sample_ids=tr['sample_ids'])
    rows=[]
    for hi,h in enumerate([12,24]):
        for k in [1,2]:rows.append(dict(year=year,months=h,cause=k,training_companies=len(count),eligible_companies=int(eligible.sum()),excluded_fewer_than10=int((~eligible).sum()),oob_trees_min=int(count.min()),oob_trees_q25=float(np.quantile(count,.25)),oob_trees_median=float(np.median(count)),oob_trees_q75=float(np.quantile(count,.75)),oob_trees_max=int(count.max()),supported=bool(eligible.sum()>=30),label_poisoning_identical=True,nodes_checked=sum(r['nodes_checked'] for r in audit),max_leaf_parent_error=max(r['max_leaf_parent_error'] for r in audit),threshold_artifact_sha256=digest(out/'oob_PRIVATE.npz')))
    pd.DataFrame(rows).to_csv(out/'OOB_SUPPORT.csv',index=False);write_json(out/'membership_audit.json',audit)
    return dict(part='B',year=year,nodes_checked=sum(r['nodes_checked'] for r in audit),thresholds_supported=bool(eligible.sum()>=30))


def main():
    p=argparse.ArgumentParser();p.add_argument('--part',choices=['A','B'],required=True);p.add_argument('--year',type=int,required=True);a=p.parse_args()
    CompetingRiskForest.fit=block_fit;root=Path(os.environ['DATA_DIR']);out=Path(os.environ['OUTPUT_DIR']);reader=Reader();start=time.perf_counter()
    summary=(part_a if a.part=='A' else part_b)(root,out,a.year,reader)
    reader.finish();write_json(out/'input_hashes_PRIVATE.json',reader.hashes)
    write_json(out/'summary.json',dict(status='COMPLETE',forest_fits=0,calibrator_fits=0,**summary,seconds=time.perf_counter()-start,host=socket.gethostname(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    write_json(out/'hashes.json',{str(f.relative_to(out)):digest(f) for f in out.rglob('*') if f.is_file()})


if __name__=='__main__':
    try:main()
    except Exception:
        (Path(os.environ['OUTPUT_DIR'])/'FAILURE_PRIVATE.txt').write_text(traceback.format_exc());raise
