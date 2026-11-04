import time
import pandas as pd
from crforest import HigherOrderCIFExpansion,CIFExpansion,CompetingRiskForest
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_seed_explanation_stability_v1.stage import catalog
from .models import *
def main():
    jobs0=jobs();r=run();rows=[];tune=[];perf=[]
    for (arm,y,seed),(p,s) in jobs0.items():
        for entry in s['tuning']:tune.append(dict(model=arm,year=y,chosen=s['parameter'],candidate=entry['parameter'],validation_year=entry['validation_year'],brier_delist=entry['brier'][0],brier_recovery=entry['brier'][1],seconds=entry['seconds']))
        for v in s['models']:
            ss=v['seed'];name=f'{arm}/{ss}' if ss else arm
            rows.append(dict(year=y,model=name,representation={'AJ':'baseline only (two curves)', 'FG':'linear subdistribution log-hazards, nonlinear CIF', 'CSC':'linear cause-specific log-hazards, coupled nonlinear CIF','KNN_AJ':'exact neighbor AJ; no bounded-order CIF export','RF':'exact native forest; no bounded-order CIF export','unrestricted':'complete CIF through order 4'}[arm],observed_main=v.get('main_functions',0 if arm=='AJ' else None),observed_pair=v.get('interaction_functions',0 if arm=='AJ' else None),order3=v.get('order3_functions',0 if arm=='AJ' else None),order4=v.get('order4_functions',0 if arm=='AJ' else None),cells=v.get('spatial_table_cells'),leaves=v.get('leaves'),links_nonzero_both_causes=v.get('link_terms'),training_reference_rows=v.get('reference_rows'),baseline_curves=2 if arm in ('AJ','FG','CSC') else None,new_outer_task_seconds=v['seconds'],forest_fit_seconds=v.get('forest_fit_seconds'),max_cif_sum=v['max_sum'],eval_firms_sum_gt1=v['rows_over_one'],exact_error=v.get('exact_error'),selected_parameter=s['parameter']))
        perf.append(dict(year=y,arm=arm,job=s['job'],seed=seed,host=s['host'],seconds=s['seconds'],python_parent_peak_mib=s['peak_rss_MiB'],outer_model_runs=len(s['models']),internal_model_runs=len(s['tuning'])))
    path=old('post_st_seed_explanation_stability_v1')/'input';source=catalog();anchors=history_members()
    for y in YEARS:
        tr=arrays(path/f'folds/{y}/train_PRIVATE.npz');ev=arrays(path/f'folds/{y}/eval_PRIVATE.npz');xx=np.vstack([tr['X'],ev['X']])
        for ss in (*PICK,*ANCHOR):
            folder=source[y,str(ss),'shrunk'] if ss in PICK else anchors[y,ss]/'shrunk'
            info=read(folder.parent/'summary.json')
            ex=CIFExpansion.load(folder/'functions_PRIVATE.npz');cx=ex.complexity(tr['X'])
            rows.append(dict(year=y,model=f'XAI/{ss}',representation='complete q2 CIF',observed_main=cx['main_functions'],observed_pair=cx['interaction_functions'],order3=0,order4=0,cells=cx['spatial_table_cells'],historical_fit_seconds=info.get('fit_seconds'),new_forest_fits=0))
        for label,ss in [('C039',PICK),('anchor',ANCHOR)]:
            t=time.perf_counter();fs=[CompetingRiskForest.load(jobs0['unrestricted',y,seed][0]/'forest.json') for seed in ss]
            f=merge(fs);ex=HigherOrderCIFExpansion.from_forest(f,tr['X']);cx=ex.complexity(tr['X']);tt=np.unique(np.r_[f.grid_,DAYS])
            err=float(np.max(abs(ex.predict(xx,tt)-np.mean([m.predict(xx,tt) for m in fs],axis=0))));assert err<=1e-10
            name='selected_seeds' if label=='C039' else 'anchor_seeds'
            rows.append(dict(year=y,model='unrestricted/'+name,representation='complete CIF through order 4',observed_main=cx['main_functions'],observed_pair=cx['interaction_functions'],order3=cx['order3_functions'],order4=cx['order4_functions'],cells=cx['spatial_table_cells'],exact_error=err,max_path=f.audit_['max_path_variables'],ensemble_compile_check_seconds=time.perf_counter()-t,member_task_seconds_sum=sum(jobs0['unrestricted',y,seed][1]['seconds'] for seed in ss)))
            p=old('post_st_grouped_explanation_v1')/f'input/{y}/C039.npz' if label=='C039' else path/f'{y}/reference96/shrunk/functions_PRIVATE.npz'
            cx=CIFExpansion.load(p).complexity(tr['X']);rows.append(dict(year=y,model='XAI/'+label,representation='complete q2 CIF',observed_main=cx['main_functions'],observed_pair=cx['interaction_functions'],order3=0,order4=0,cells=cx['spatial_table_cells'],max_path=2,new_forest_fits=0,historical_fit_seconds=sum(v['historical_fit_seconds'] for v in rows if v['year']==y and v['model'] in [f'XAI/{k}' for k in ss]) if all(v.get('historical_fit_seconds') is not None for v in rows if v['year']==y and v['model'] in [f'XAI/{k}' for k in ss]) else None))
            seeds=[v for v in jobs0['RF',y,0][1]['models'] if v['seed'] in ss]
            rows.append(dict(year=y,model='RF/'+name,representation='96-tree native CIF; no low-order bound or compiled function counts',leaves=sum(v['leaves'] for v in seeds),member_task_seconds_sum=sum(v['seconds'] for v in seeds)))
    pd.DataFrame(rows).to_csv(D/'COMPLEXITY.csv',index=False);pd.DataFrame(tune).to_csv(D/'TUNING.csv',index=False);pd.DataFrame(perf).to_csv(D/'COMPUTE_COST.csv',index=False)
if __name__=='__main__':main()
