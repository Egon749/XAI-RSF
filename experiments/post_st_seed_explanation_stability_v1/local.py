"""All-company local explanation comparisons on calendar endpoints."""
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from .common import *
from .similarity import *
from .geometry import med

def local(year,version,members,labels,ev,keys,out,root):
    group_rows=[]; firm_rows=[]; case_rows=[]; profile=[]; term_rows=[]; global_cloud=[]
    keynames=['+'.join(map(str,k)) for k in keys];features=read(root/'features.json')
    labels_term=[' × '.join(features[j]['feature_name'] for j in key) for key in keys]
    n=len(ev['X']); ix=np.arange(n)
    for h in (12,24):
        hi=labels[f'h{h}'];days=np.unique(hi);di=np.searchsorted(days,hi)
        terms=np.asarray([np.stack([ex.component(ev['X'],key,days)[ix,di,:] for key in keys],axis=-1) for ex in members])
        mu=np.stack([ex._coeff(days)[1][di] for ex in members])
        risk=np.stack([ex.predict(ev['X'],days)[ix,di] for ex in members])
        np.testing.assert_allclose(mu+terms.sum(-1),risk,atol=1e-10,rtol=0)
        for c in (1,2):
            base=dict(year=year,version=version,months=h,cause=c)
            stats=[]; examples=[]; coords=np.array(TRIPLES)
            for row in range(n):
                x=terms[:,row,c-1,:][coords].mean(1); scores=risk[:,row,c-1][coords].mean(1);baseline=mu[:,row,c-1][coords].mean(1)
                np.testing.assert_allclose(x.sum(1)+baseline,scores,atol=1e-10,rtol=0)
                pos=(x>=MATERIAL).astype(float);neg=(x<=-MATERIAL).astype(float)
                valid=(pos+neg)@(pos+neg).T
                sign=divide(pos@pos.T+neg@neg.T,valid)
                l1=pdist(x,'cityblock');riskdiff=abs(scores[LEFT]-scores[RIGHT]); top={}
                order=np.argsort(-abs(x),axis=1,kind='stable')
                for k in [5,10]:
                    mask=np.zeros(x.shape,bool)
                    np.put_along_axis(mask,order[:,:k],True,axis=1);mask&=abs(x)>=MATERIAL
                    top[k]=jaccards(mask)[LEFT,RIGHT]
                ranks=correlations(rankdata(abs(x),axis=1))[LEFT,RIGHT]
                record=dict(row=row,base_sd=baseline.std(ddof=1),base_range=np.ptp(baseline),score_sd=scores.std(ddof=1),
                            main_contribution_sd=np.mean(x[:,[len(k)==1 for k in keys]].std(0,ddof=1)),
                            pair_contribution_sd=np.mean(x[:,[len(k)==2 for k in keys]].std(0,ddof=1)),
                            median_l1=med(l1),risk_l1_correlation=med(correlations(np.stack([riskdiff,l1]))[0,1]))
                for tag,a in [('top5_jaccard',top[5]),('top10_jaccard',top[10]),('contribution_rank_spearman',ranks),('sign_agreement',sign[LEFT,RIGHT]),
                              ('common_material_terms',valid[LEFT,RIGHT]),('l1_distance',l1),('absolute_risk_difference',riskdiff)]:
                    for overlap in ('all',0,1,2):
                        measure=med(a if overlap=='all' else a[SHARED==overlap])
                        record[f'{tag}_{overlap}']=measure
                stats.append(record)
                examples.append((x,baseline,scores))
            data=pd.DataFrame(stats)
            for key in data.drop(columns='row'):
                group_rows.append(dict(**base,metric=key,**distributions(data[key])))
            firm_rows.extend([base|s for s in stats])
            xall=np.stack([z[0] for z in examples])
            for j,name in enumerate(labels_term):
                sd=xall[:,:,j].std(1,ddof=1)
                term_rows.append(dict(**base,term=name,mean_company_sd=sd.mean(),p95_company_sd=np.quantile(sd,.95),mean_absolute=np.mean(abs(xall[:,:,j]))))
            for case,q in [('stable',.1),('unstable',.9)]:
                distances=data.median_l1.to_numpy()
                row=int(np.argmin(abs(distances-np.quantile(distances,q))))
                xx,baseline,scores=examples[row]
                for t,name in enumerate(keynames):
                    case_rows.append(dict(**base,case=case,row=row,term=name,median=med(xx[:,t]),p05=np.quantile(xx[:,t],.05),p95=np.quantile(xx[:,t],.95),mean_risk=scores.mean(),mean_base=baseline.mean()))
                sort=np.argsort(distances,kind='stable');size=max(5,int(np.ceil(.2*n)));inds=sort[:size] if q<.5 else sort[-size:]
                xgroup=np.stack([abs(examples[i][0]).mean(0) for i in inds])
                for j in np.argsort(-xgroup.mean(0),kind='stable')[:5]:
                    profile.append(dict(**base,band=case,companies=len(inds),term=labels_term[j],mean_absolute=xgroup[:,j].mean()))
            # Private exact case inputs, including per-combination values and term list.
            for q in [.1,.9]:
                row=int(np.argmin(abs(data.median_l1-np.quantile(data.median_l1,q))))
                x,baseline,scores=examples[row]
                np.savez_compressed(out/f'case_{h}_{c}_{q}_PRIVATE.npz',row=row,contributions=x,baseline=baseline,risk=scores,terms=keynames)
    for name,data in [('local',group_rows),('firms_PRIVATE',firm_rows),('cases_PRIVATE',case_rows),('profiles',profile),('local_term',term_rows)]:
        pd.DataFrame(data).to_csv(out/(name+'.csv'),index=False)
