"""Independent replay of published ensemble metrics and fallback identity."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from .run import RUN,PUBLIC,SEARCH,PARENT
from experiments.post_st_feature_combination_search_v1.analyze import model_specs,seed_folders
from experiments.post_st_dropcolumn_selection_v1.common import arrays,load,write,sha,TIMES,horizon_predictions,evaluation_context

def main():
    frame=pd.read_csv(PUBLIC/'MODEL_COMPARISON.csv');_,specs=model_specs(SEARCH)
    verified=0
    for y in [2023,2024,2025]:
        lab=arrays(PARENT/f'evaluation_input/{y}_test_PRIVATE.npz')
        for label,base in [('full83','full83'),('legacy74','legacy74'),('selected_pipeline','full83')]:
            spec=specs[(y,base)];p=arrays(__import__('pathlib').Path(spec['path'])/'ensemble/predictions_PRIVATE.npz')
            seeds=[arrays(z/'predictions_PRIVATE.npz') for z in seed_folders(spec)]
            for z in seeds:np.testing.assert_array_equal(z['sample_ids'],lab['sample_ids'])
            np.testing.assert_allclose(p['cif'],np.mean([z['cif'] for z in seeds],axis=0),atol=1e-13,rtol=0)
            for h in [12,24]:
                risk=horizon_predictions(p['cif'],TIMES,lab[f'h{h}'])
                for c in [1,2]:
                    yy,w,known,support,_=evaluation_context(lab,h,c)
                    row=frame[(frame.year.astype(str)==str(y))&(frame.model==label)&(frame.seed=='ensemble')&(frame.horizon==h)&(frame.cause==c)].iloc[0]
                    if not support:assert pd.isna(row.auc) and pd.isna(row.ap) and pd.isna(row.brier);continue
                    if yy[known].any() and (~yy[known]).any():
                        assert abs(row.auc-roc_auc_score(yy[known],risk[known,c-1],sample_weight=w[known]))<1e-12
                        assert abs(row.ap-average_precision_score(yy[known],risk[known,c-1],sample_weight=w[known]))<1e-12
                    assert abs(row.brier-np.sum(w*(risk[:,c-1]-yy)**2)/len(yy))<1e-12
                    verified+=1
    for p,h in load(RUN/'audit/source_hashes.json').items():assert sha(__import__('pathlib').Path(p))==h
    receipt=load(PUBLIC/'execution_receipt.json')
    assert receipt['new_raw_forests']==450 and receipt['verified_models']==1050
    for p,h in receipt['output_hashes'].items():assert sha(PUBLIC/p)==h
    s=pd.read_csv(PUBLIC/'FEATURE_SELECTION_STABILITY.csv')
    assert len(s)==249 and s.selection_frequency.isna().all() and (s.supervised_search_replicates==0).all()
    result=dict(status='PASS',independent_sklearn_metric_rows=verified,source_hashes_unchanged=True,
                ensemble_identity=True,undefined_2025_H24_retained=True,frequency_NA_not_faked=True)
    write(RUN/'audit/final_validation.json',result);write(PUBLIC/'audit/final_validation.json',result)
    print(result)

if __name__=='__main__':main()
