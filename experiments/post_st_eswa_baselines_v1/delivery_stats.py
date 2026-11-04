import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from .evaluate import tasks
from .common import *
def main():
    r=run();jobs=tasks(r);oldlogs=[];param={};fits=['PROBoost_tune','PROBoost_final','IPSO_proxy_fit','Hyperopt_fit','base_refit','GA_meta_fit','final_meta_fit']
    z=pd.read_csv(D/'PUBLICATION_CHECKS.csv');root=old('post_st_seed_explanation_stability_v1')/'input'
    for j in r.glob('published/*/task-*/output/result.json'):
        s=load(j);p=j.parent
        if (s['year'],s['months'],s['arm'])!=(2023,12,'Stacking') or not (r/'repair_published').exists():continue
        lab=arrays(root/'evaluation_input/2023_test_PRIVATE.npz');hd=Horizon(lab,12,'complete');pp=arrays(p/'predictions.npz')['score'];ids=lab['sample_ids'];take=np.lexsort((ids,-pp))[:int(np.ceil(.2*len(pp)))];ll=pd.read_csv(p/'training.csv')
        oldlogs.append(dict(job=s['job'],year=2023,months=12,mode=s['mode'],status='SUPERSEDED_GA_FALLBACK',fit_operations=int(ll.stage.isin(fits).sum()),auc=roc_auc_score(hd.y,pp),ap=average_precision_score(hd.y,pp),captured=int(hd.y[take].sum()),elapsed_seconds=s['elapsed'],reason='One OOF fold used estimated F1 instead of uniform weights when GA lacked support; raw files retained.'))
    for (y,h,a,mode),(p,s) in jobs.items():
        if a=='Stacking':
            with (p/'stack.pkl').open('rb') as stream:b,meta=pickle.load(stream)
            def clean(j):
                return {k:(float(v) if isinstance(v,float) and np.isfinite(v) else None if isinstance(v,float) else str(v) if not isinstance(v,(str,int,bool,type(None))) else v) for k,v in j.items()}
            param[f'{y}/{h}/{a}/{mode}']=dict(base=[clean(m.get_params(deep=False)) for m in b['models']],meta=clean(meta.get_params()),selected=b['mask'].tolist(),positive_weight=(b['weight']>0).tolist())
        else:param[f'{y}/{h}/{a}/{mode}']=dict(beta_start=s['cfg'][0][0],beta_end=s['cfg'][0][1],shift=s['cfg'][1],depth=s['cfg'][2],tree_method='hist',trees=48,eta=.075,lambda_L2=1,gamma=0,min_child_weight=1,base_score=0,seed=811,subsample=1,colsample_bytree=1,objective='paper-equations-3-to-8')
    dump(D/'HYPERPARAMETERS.json',param);pd.DataFrame(oldlogs).to_csv(D/'SUPERSEDED_JOBS.csv',index=False)
    meta=pd.read_csv(D/'SEARCH_BUDGET.csv');active=int(meta.training_runs.sum());other=sum(a['fit_operations'] for a in oldlogs)
    dump(D/'FIT_ACCOUNTING.json',dict(retained_fits=active,superseded_fits=int(other),physically_executed_fits=int(active+other),final_predictors=24,new_xai_rsf_fits=0,definition='One fit is one fitted base model or one fitted meta LR, not one tree, candidate proposal, OOF row, bootstrap or restored model. PROBoost outer/development fits count once per 48 trees.'))
if __name__=='__main__':main()
