"""Replay public metrics independently and verify actual searches were executed."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from .run import RUN,PUBLIC,PARENT
from experiments.post_st_dropcolumn_selection_v1.common import arrays,load,write,sha,TIMES,horizon_predictions,evaluation_context


def main():
    frame=pd.read_csv(PUBLIC/'MODEL_COMPARISON.csv');sources=pd.read_csv(PUBLIC/'MODEL_SOURCES.csv');verified=0
    for row in sources.to_dict('records'):
        y=int(row['year']);folder=Path(row['path']);lab=arrays(PARENT/f'evaluation_input/{y}_test_PRIVATE.npz')
        p=arrays(folder/'ensemble/predictions_PRIVATE.npz');np.testing.assert_array_equal(p['sample_ids'],lab['sample_ids'])
        assert sha(folder/'ensemble/predictions_PRIVATE.npz')==row['prediction_sha256']
        for h in [12,24]:
            risk=horizon_predictions(p['cif'],TIMES,lab[f'h{h}'])
            for c in [1,2]:
                yy,w,known,support,_=evaluation_context(lab,h,c)
                r=frame[(frame.year.astype(str)==str(y))&(frame.model==row['model'])&(frame.seed=='ensemble')&(frame.horizon==h)&(frame.cause==c)].iloc[0]
                if not support:
                    assert pd.isna(r.auc) and pd.isna(r.ap) and pd.isna(r.brier);continue
                if yy[known].any() and (~yy[known]).any():
                    assert abs(r.auc-roc_auc_score(yy[known],risk[known,c-1],sample_weight=w[known]))<1e-12
                    assert abs(r.ap-average_precision_score(yy[known],risk[known,c-1],sample_weight=w[known]))<1e-12
                assert abs(r.brier-np.sum(w*(risk[:,c-1]-yy)**2)/len(yy))<1e-12
                order=np.lexsort((lab['sample_ids'],-risk[:,c-1]))[:int(np.ceil(.2*len(yy)))]
                assert r.captured==int((yy[order]&known[order]).sum());verified+=1
    for h in [12,24]:
        labels={y:arrays(PARENT/f'evaluation_input/{y}_test_PRIVATE.npz') for y in [2023,2024,2025]}
        years=[y for y,l in labels.items() if evaluation_context(l,h,1)[3]]
        lab={k:np.concatenate([labels[y][k] for y in years]) for k in labels[years[0]]}
        for model in ['full83','legacy74','selected_pipeline']:
            risks=[]
            for y in years:
                folder=Path(sources[(sources.year==y)&(sources.model==model)].iloc[0]['path'])
                risks.append(horizon_predictions(arrays(folder/'ensemble/predictions_PRIVATE.npz')['cif'],TIMES,labels[y][f'h{h}']))
            risk=np.concatenate(risks)
            for c in [1,2]:
                yy,w,known,support,_=evaluation_context(lab,h,c);assert support
                row=frame[(frame.year.astype(str)=='pooled')&(frame.model==model)&(frame.horizon==h)&(frame.cause==c)].iloc[0]
                assert abs(row.auc-roc_auc_score(yy[known],risk[known,c-1],sample_weight=w[known]))<1e-12
                assert abs(row.ap-average_precision_score(yy[known],risk[known,c-1],sample_weight=w[known]))<1e-12
                assert abs(row.brier-np.sum(w*(risk[:,c-1]-yy)**2)/len(yy))<1e-12
                annual=frame[(frame.year.astype(str).isin(list(map(str,years))))&(frame.model==model)&(frame.seed=='ensemble')&(frame.horizon==h)&(frame.cause==c)]
                assert row.captured==annual.captured.sum() and row.checked==annual.checked.sum()
                verified+=1
    for p,h in load(RUN/'audit/source_hashes.json').items():assert sha(Path(p))==h
    receipt=load(PUBLIC/'execution_receipt.json');tasks=pd.read_csv(PUBLIC/'PIPELINE_RUNS.csv');scans=pd.read_csv(PUBLIC/'GROUP_DISCOVERY_RESULTS.csv')
    assert len(tasks)==153 and receipt['new_raw_forests']==tasks.new_forests.sum()
    assert receipt['resampled_searches']>0 and receipt['point_searches']==2
    for r in tasks.to_dict('records'):
        rows=scans[(scans.year==r['year'])&(scans.repeat==r['repeat'])&(scans['mode']=='anchored')]
        assert len(rows)==r['scored_candidates']
        if r['discovery_pass']:assert len(rows)>0
    for p,h in receipt['output_hashes'].items():assert sha(PUBLIC/p)==h
    result=dict(status='PASS',independent_sklearn_metric_rows=verified,source_hashes_unchanged=True,
        actual_supported_search_branch_verified=True,all_declared_candidates_retained=True,undefined_2025_H24_preserved=True)
    for p in [RUN/'audit/final_validation.json',PUBLIC/'audit/final_validation.json']:
        if p.exists():assert load(p)==result
        else:write(p,result)
    print(result)

if __name__=='__main__':main()
