"""Historical-only drop-column experiment; no change to the forest estimator."""
from pathlib import Path
from dataclasses import replace
import hashlib,json
import numpy as np
from experiments.post_st_drop9_v1.common import DROP_NAMES,REFERENCE_SHA,reduced
from experiments.post_st_ablation_v1.common import configuration as reference_configuration,pairing,array_hash
from experiments.post_st_prognosis_v1.common import arrays,sha,write,calendar_elapsed,horizon_predictions,endpoint_weights
from experiments.post_st_feature_permutation_v1.common import evaluation_context,score_batch
from experiments.post_st_paper_features_v1.common import history,references

NAME='post_st_dropcolumn_selection_v1'
YEARS=[2023,2024,2025]
FOLDS=list(range(2019,2025))
SEEDS=[42,101,202]
TIMES=np.array([365.,366.,730.,731.])
SOURCE_COMMIT='31cc93d4bcad16d1e680a88cba5a4b186debcfe2'
BASE=Path('/home/oliver/MainDiskData/xai_rsf_2025_shallow_parent_20261029_pSED0uC5')
PERM=Path('/home/oliver/MainDiskData/xai_rsf_feature_permutation_20261031_iq518ze_')
ANALYSIS_SEED=2026110183

def load(p):return json.loads(Path(p).read_text())

def subset_hash(keep):return hashlib.sha256(json.dumps(list(map(int,keep)),separators=(',',':')).encode()).hexdigest()

def configuration(seed,p,mode='count'):
    count=min(25,p) if mode=='count' else min(p,max(1,int(np.floor(p*25/83+.5))))
    return replace(reference_configuration(seed,'shallow_parent'),tree_feature_count=count)

def feature_group(j):
    return 'financial' if j<48 else 'market' if j<72 else 'quality_staleness' if j<75 else 'entry_warning' if j==75 else 'linked_audit_opinion'

def support_gate(rows):
    """Operational safeguards, not a power theorem or equivalence claim."""
    details={}
    for h in [12,24]:
        r=[x for x in rows if x['horizon']==h and x['mature'] and x['supported'] and x['train_termination']>=1]
        informative=[x for x in r if x['termination']>=3 and x['controls']>=5]
        details[str(h)]=dict(mature_target_events=sum(x['termination'] for x in r),
            informative_cohorts=len(informative),informative_target_events=sum(x['termination'] for x in informative),
            passed=len(informative)>=2 and sum(x['termination'] for x in informative)>=10)
    return dict(passed=all(x['passed'] for x in details.values()),horizons=details,
        reason='PASS' if all(x['passed'] for x in details.values()) else 'INSUFFICIENT_HISTORICAL_TARGET_EVENT_SUPPORT')

def acceptable(d):
    """Declared deletion constraints; d aggregates only eligible historical folds."""
    if not d.get('support_pass',False):return False
    return (d['delta_ap12']>=.005 and d['delta_capture12']>=0 and d['ap12_lower90']>=-.01
        and d['delta_auc24']>=-.02 and d['delta_ap24']>=-.02 and d['delta_capture24']>=-1
        and d['delta_brier12']<=.002 and d['delta_brier24']<=.002
        and d['recovery_delta_brier12']<=.005 and d['recovery_delta_brier24']<=.005
        and d['stable_direction_fraction']>=2/3 and d['pair_jaccard_change']>=-.10
        and d['spatial_cell_change']<=0 and d['valid_model'])

def pick_candidate(candidates):
    valid=[x for x in candidates if acceptable(x)]
    return max(valid,key=lambda x:(x['delta_ap12'],x['delta_capture12'],-x['spatial_cell_change'],x['subset_id'])) if valid else None

def metrics(labels,risk,months,cause):
    y,w,known,supported,diag=evaluation_context(labels,months,cause)
    result=dict(companies=len(y),target_events=int((known&y).sum()),unknown=int((~known).sum()),supported=supported,
        observed=float(np.sum(w*y)/len(y)) if supported else np.nan,risk_mean=float(risk.mean()),
        auc=np.nan,ap=np.nan,brier=np.nan,captured=np.nan,coverage=np.nan,checked=int(np.ceil(.2*len(y))),reason='')
    if not supported:result['reason']='NO_FOLLOWUP_SUPPORT_AT_HORIZON';return result
    s=score_batch(risk,y,w)
    result.update({k:float(s[k][0]) for k in ['auc','ap','brier']})
    chosen=np.lexsort((labels['sample_ids'],-risk))[:result['checked']]
    result['captured']=int((y[chosen]&known[chosen]).sum())
    result['coverage']=result['captured']/result['target_events'] if result['target_events'] else np.nan
    if not result['target_events']:result['reason']='NO_OBSERVED_TARGET_EVENTS_AUC_AP_UNDEFINED'
    return result

def bootstrap_differences(labels,a,b,months,cause,draws):
    """Paired company draws; fixed IPCW weights, mature primary folds have w=1."""
    y,w,known,supported,_=evaluation_context(labels,months,cause)
    if not supported:return {}
    counts=np.stack([np.bincount(x,minlength=len(y)) for x in draws]);ww=counts*w
    out={}
    for key,p in [('a',a),('b',b)]:
        order=np.argsort(-p,kind='stable');ps=p[order];yw=ww[:,order]*y[order]
        tp=np.cumsum(yw,1);total=np.cumsum(ww[:,order],1)
        end=np.r_[np.flatnonzero(ps[:-1]!=ps[1:]),len(ps)-1]
        pos=ww[:,y].sum(1);neg=ww[:,~y].sum(1)
        precision=np.divide(tp[:,end],total[:,end],out=np.zeros_like(tp[:,end]),where=total[:,end]>0)
        ap=np.divide(np.sum(np.diff(np.c_[np.zeros(len(counts)),tp[:,end]],axis=1)*precision,1),pos,out=np.full(len(counts),np.nan),where=pos>0)
        positive=np.flatnonzero(y);negative=np.flatnonzero(~y)
        concordance=(p[positive,None]>p[None,negative])+.5*(p[positive,None]==p[None,negative])
        numerator=np.einsum('bi,ij,bj->b',ww[:,positive],concordance,ww[:,negative],optimize=True)
        auc=np.divide(numerator,pos*neg,out=np.full(len(counts),np.nan),where=pos*neg>0)
        brier=ww@((p-y)**2)/len(y)
        out[key]=dict(auc=auc,ap=ap,brier=brier)
    return {k:out['b'][k]-out['a'][k] for k in ['auc','ap','brier']}
