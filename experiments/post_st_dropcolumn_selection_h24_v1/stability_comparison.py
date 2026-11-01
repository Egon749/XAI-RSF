"""Compare saved predictors on identical reference rows and company draws."""
import numpy as np
from .fitting import TIMES


def compare_variation(year,reference,baseline_parts,selected_parts,baseline_anchor,selected_anchor,
                      baseline_mu,selected_mu,baseline_anchor_mu,selected_anchor_mu):
    """Anchor-free function spread plus distance to each pipeline's own anchor.

    Component spread is sqrt(mean_x sum_term Var_bootstrap f_term(x)). It
    includes functions absent from a replicate as zero, without dividing by a
    changing term count. All evaluation rows and normalization references agree.
    """
    n=len(reference['X']);zero=np.zeros((n,4,2));rows=np.arange(n)
    parts=[baseline_parts,selected_parts];anchors=[baseline_anchor,selected_anchor]
    mus=[baseline_mu,selected_mu];anchor_mus=[baseline_anchor_mu,selected_anchor_mu]
    assert len(baseline_parts)==len(selected_parts)==50
    keys=set(baseline_anchor)|set(selected_anchor)
    for group in parts:
        keys.update(k for part in group for k in part)
    prediction=[np.stack([mu+sum(part.values(),start=zero.copy()) for part,mu in zip(group,mm)]) for group,mm in zip(parts,mus)]
    anchor_prediction=[mu+sum(anchor.values(),start=zero.copy()) for anchor,mu in zip(anchors,anchor_mus)]
    summary=[];paired=[];pred=[]
    for h in [12,24]:
        ix=np.searchsorted(TIMES,reference[f'h{h}'])
        for cause in [1,2]:
            for i,label in enumerate(['full83','selected_h24']):
                pp=prediction[i][:,rows,ix,cause-1];aa=anchor_prediction[i][rows,ix,cause-1]
                variance=np.var(pp,axis=0,ddof=1)
                pred.append(dict(year=year,horizon=h,cause=cause,model=label,replicates=50,reference_companies=n,
                    root_mean_prediction_variance=float(np.sqrt(variance.mean())),mean_company_prediction_sd=float(np.sqrt(variance).mean()),
                    anchor_mean_risk=float(aa.mean()),bootstrap_mean_risk=float(pp.mean())))
            for order in [1,2]:
                spread=np.zeros(2);distance=np.zeros((2,50));intersection=np.zeros((2,50));union=np.zeros((2,50))
                for key in sorted(k for k in keys if len(k)==order):
                    for i in [0,1]:
                        vv=np.stack([p.get(key,zero)[rows,ix,cause-1] for p in parts[i]])
                        aa=anchors[i].get(key,zero)[rows,ix,cause-1]
                        spread[i]+=np.var(vv,axis=0,ddof=1).mean()
                        distance[i]+=np.mean((vv-aa)**2,axis=1)
                        active=np.sqrt(np.mean(vv*vv,axis=1))>1e-14;anchor_active=np.sqrt(np.mean(aa*aa))>1e-14
                        intersection[i]+=active&anchor_active;union[i]+=active|anchor_active
                jaccard=np.divide(intersection,union,out=np.ones_like(union),where=union>0)
                distance=np.sqrt(distance)
                for i,label in enumerate(['full83','selected_h24']):
                    summary.append(dict(year=year,horizon=h,cause=cause,order=order,model=label,replicates=50,reference_companies=n,
                        component_spread_l2=float(np.sqrt(spread[i])),mean_anchor_distance_l2=float(distance[i].mean()),
                        anchor_distance_p025=float(np.quantile(distance[i],.025)),anchor_distance_p975=float(np.quantile(distance[i],.975)),
                        mean_nonzero_jaccard_to_anchor=float(jaccard[i].mean())))
                for repeat in range(50):
                    paired.append(dict(year=year,horizon=h,cause=cause,order=order,repeat=repeat,
                        full83_anchor_distance=distance[0,repeat],selected_anchor_distance=distance[1,repeat],
                        difference=distance[1,repeat]-distance[0,repeat],full83_jaccard=jaccard[0,repeat],selected_jaccard=jaccard[1,repeat]))
    return summary,paired,pred
