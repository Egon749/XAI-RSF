"""Complete supported endpoints only; keep empty/one-class subsets explicit."""
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score

def group_metrics(y,p,mask,ids,years):
    y=np.asarray(y,bool);p=np.asarray(p,float);mask=np.asarray(mask,bool);n=int(mask.sum());total=int(y.sum());events=int(y[mask].sum());loss=(p-y)**2
    result=dict(companies=len(y),accepted=n,referred=len(y)-n,coverage=n/len(y),all_events=total,accepted_events=events,referred_events=total-events,
        event_retention=events/total if total else np.nan,brier=np.nan,risk_mean=np.nan,observed=np.nan,calibration_gap=np.nan,auc=np.nan,ap=np.nan,
        event_mse=np.nan,nonevent_mse=np.nan,balanced_mse=np.nan,accepted_top20_checked=0,accepted_top20_captured=0,
        original_budget_checked=0,original_budget_captured=0,original_top20_referred=0,reason='')
    if n:
        z=y[mask];pp=p[mask];result.update(brier=float(loss[mask].mean()),risk_mean=float(pp.mean()),observed=float(z.mean()),calibration_gap=float(pp.mean()-z.mean()),
            event_mse=float(loss[mask&y].mean()) if events else np.nan,nonevent_mse=float(loss[mask&~y].mean()) if events<n else np.nan)
        if events and events<n:result['auc']=float(roc_auc_score(z,pp));result['balanced_mse']=(result['event_mse']+result['nonevent_mse'])/2
        if events:result['ap']=float(average_precision_score(z,pp))
        if not events or events==n:result['reason']='ONE_CLASS_AUC_UNDEFINED'
    else:result['reason']='NO_ACCEPTED_COMPANIES'
    for yr in np.unique(years):
        ix=np.flatnonzero(years==yr);order=ix[np.lexsort((ids[ix],-p[ix]))];eligible=order[mask[order]]
        original_k=int(np.ceil(.2*len(ix)));small_k=int(np.ceil(.2*len(eligible)))
        result['accepted_top20_checked']+=small_k;result['accepted_top20_captured']+=int(y[eligible[:small_k]].sum())
        result['original_budget_checked']+=min(original_k,len(eligible));result['original_budget_captured']+=int(y[eligible[:original_k]].sum());result['original_top20_referred']+=int((~mask[order[:original_k]]).sum())
    return result

def decomposition(full,accepted):
    keys=['observed','event_mse','nonevent_mse']
    if not np.isfinite([full[k] for k in keys]+[accepted[k] for k in keys]).all():
        return dict(delta_brier=accepted['brier']-full['brier'],prevalence_mix_component=np.nan,within_status_component=np.nan,residual=np.nan,reason='BOTH_CLASSES_REQUIRED')
    a,b=accepted,full;meanpi=(a['observed']+b['observed'])/2
    mix=(a['observed']-b['observed'])*((a['event_mse']+b['event_mse']-a['nonevent_mse']-b['nonevent_mse'])/2)
    within=meanpi*(a['event_mse']-b['event_mse'])+(1-meanpi)*(a['nonevent_mse']-b['nonevent_mse'])
    delta=a['brier']-b['brier'];residual=delta-mix-within;assert abs(residual)<1e-12
    return dict(delta_brier=delta,prevalence_mix_component=mix,within_status_component=within,residual=residual,reason='')
