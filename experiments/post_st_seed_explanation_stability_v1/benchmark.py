"""Compare every tree budget against the same disjoint 42/101/202 anchor."""
import pandas as pd
from scipy.stats import rankdata
from crforest import CIFExpansion
from experiments.post_st_ablation_v1.analyze import selected
from .common import *
from .similarity import jaccards,correlations

def run():
    r=run_path();root=r/'input';rows=[]
    for year in YEARS:
        tr=arrays(root/f'folds/{year}/train_PRIVATE.npz');ev=arrays(root/f'folds/{year}/eval_PRIVATE.npz')
        labels=arrays(root/f'evaluation_input/{year}_test_PRIVATE.npz')
        for v in VERSIONS:
            arms=[*map(str,SEEDS),'ensemble320','reference96']
            base=[CIFExpansion.load(root/str(year)/s/v/'functions_PRIVATE.npz') for s in arms]
            keys=sorted({(j,) for e in base for j in e.mains}|{k for e in base for k in e.pairs})
            orders=np.array(list(map(len,keys)));X=tr['X']
            structures={n:set(a) for n,a in read(r/f'analysis_input/mask_{year}_{v}.json').items()}
            paths=__import__(__package__+'.stage',fromlist=['catalog']).catalog()
            for a in arms:
                details=read(paths[year,a,v]/'complexity_PRIVATE.json')['functions']
                structures[a]={','.join(map(str,d['features'])) for d in details if d['nonzero']}
            coordinates=np.stack([np.stack([e.component(X,k,[365,730]) for k in keys]) for e in base])
            forecasts=np.stack([e.predict(ev['X'],TIMES) for e in base])
            for name,ids in [(s,[i]) for i,s in enumerate(arms[:-1])]+list(zip(NAMES,TRIPLES)):
                vals=np.mean(coordinates[list(ids)],axis=0);ref=coordinates[-1]
                p=np.mean(forecasts[list(ids)],axis=0);ar=forecasts[-1]
                for hi,h in enumerate((12,24)):
                    endpoint=np.searchsorted(TIMES,labels[f'h{h}'])
                    for c in (1,2):
                        pp=p[np.arange(len(endpoint)),endpoint,c-1];rr=ar[np.arange(len(endpoint)),endpoint,c-1]
                        common=dict(year=year,version=v,model=name,months=h,cause=c)
                        mask=np.stack([selected(z,np.full(len(rr),year),ev['sample_ids']) for z in (pp,rr)])
                        pred=dict(rank_spearman=correlations(rankdata(np.stack([pp,rr]),axis=1))[0,1],top20_jaccard=jaccards(mask)[0,1],mean_abs_risk_difference=abs(pp-rr).mean())
                        for order in (1,2):
                            left={k for k in structures[name] if len(k.split(','))==order}
                            right={k for k in structures['reference96'] if len(k.split(','))==order}
                            mask=np.array([','.join(map(str,k)) in (left&right) for k in keys])
                            a=vals[mask,:,hi,c-1];b=ref[mask,:,hi,c-1]
                            az=a-a.mean(1,keepdims=True);bz=b-b.mean(1,keepdims=True)
                            den=np.linalg.norm(az,axis=1)*np.linalg.norm(bz,axis=1)
                            valid=(a.std(1)>1e-12)&(b.std(1)>1e-12)
                            corr=np.sum(az[valid]*bz[valid],axis=1)/den[valid]
                            rows.append(common|pred|dict(order=order,structure_jaccard=len(left&right)/len(left|right),
                                median_common_shape=np.median(corr) if corr.size else np.nan,
                                mean_common_rms=np.sqrt(np.mean((a-b)**2,axis=1)).mean() if len(a) else np.nan,
                                all_terms_mean_absolute_difference=np.mean(np.abs(vals[orders==order,:,hi,c-1]-ref[orders==order,:,hi,c-1]))))
    pd.DataFrame(rows).to_csv(HERE/'FIXED_ANCHOR_COMPARISON.csv',index=False)
if __name__=='__main__':run()
