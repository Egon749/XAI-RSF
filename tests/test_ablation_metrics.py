import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from experiments.post_st_ablation_v1.uncertainty import fast_metrics
from experiments.post_st_ablation_v1.analyze import selected,contrasts

def test_tied_scores_and_repeated_company_draws_match_weighted_rank_metrics():
    y=np.array([1,0,1,0,0,1,0],bool);p=np.array([.3,.3,.2,.1,.1,.1,0]);year=np.array([2023]*3+[2024]*4);ids=np.array(list('abcdefg'))
    for ix in [np.arange(7),np.array([0,0,2,3,3,6,5])]:
        z=fast_metrics(y[ix],p[ix],year[ix],ids[ix])
        np.testing.assert_allclose(z[:2],[roc_auc_score(y[ix],p[ix]),average_precision_score(y[ix],p[ix])],atol=1e-14,rtol=0)
        assert z[2]==np.mean((p[ix]-y[ix])**2)
        assert selected(p[ix],year[ix],ids[ix]).sum()==2
    z=fast_metrics(np.zeros(7,bool),p,year,ids)
    assert np.isnan(z[0]) and np.isnan(z[1]) and np.isnan(z[-1])

def test_contrasts_encode_raw_shrinkage_and_depth_interaction():
    c=contrasts()
    assert c['depth_shrink_interaction']==dict(shallow_parent=1,shallow=-1,parent_shrink=-1,baseline=1)
    assert c['iid_features_shrink_interaction']==dict(iid_features=1,iid_features_raw=-1,shallow_parent=-1,shallow=1)
    assert all(sum(w.values())==0 for w in c.values())

def test_paired_bootstrap_pipeline_identical_predictions_have_zero_intervals(tmp_path):
    import json,pandas as pd
    from experiments.post_st_ablation_v1.common import ARMS
    from experiments.post_st_ablation_v1.uncertainty import bootstrap
    for name in ['audit','evaluation','evaluation_input','private']:(tmp_path/name).mkdir()
    (tmp_path/'audit/protocol.json').write_text(json.dumps({'bootstrap':{'replicates':20,'analysis_seed':17}}))
    forecasts={}
    for year in [2023,2024]:
        np.savez_compressed(tmp_path/'evaluation_input'/f'{year}_test_PRIVATE.npz',X=np.zeros((5,2)),event=[1,1,2,0,0],analysis_time=[5.,10.,15.,24.,24.],label_eligible=np.ones(5,bool),year=np.full(5,year),sample_ids=np.array([f'{year}-{i}' for i in range(5)]))
        for a in ARMS:forecasts[f'{year}|{a}|ensemble|24']=np.column_stack([[.2,.4,.1,.05,.1],np.zeros(5)])
    for a in ARMS:forecasts[f'pooled|{a}|ensemble|24']=np.concatenate([forecasts[f'{y}|{a}|ensemble|24'] for y in [2023,2024]])
    np.savez_compressed(tmp_path/'private/evaluation_forecasts_PRIVATE.npz',**forecasts)
    bootstrap(tmp_path)
    d=pd.read_csv(tmp_path/'evaluation/paired_bootstrap_intervals.csv')
    np.testing.assert_allclose(d[['estimate','lower95','upper95']].to_numpy(),0,atol=1e-14)
    assert d.undefined_replicates.max()>0  # Zero-event resamples remain undefined.
