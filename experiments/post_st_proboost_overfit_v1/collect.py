"""Verify all publications, compare with original training records, then publish aggregates."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

from experiments.post_st_eswa_baselines_v1 import proboost
from experiments.post_st_eswa_baselines_v1.common import Horizon, arrays, dump, load, sha
from .schedule import D, YEARS
from .worker import CONFIG, response

TOL = 1e-12
KEYS = ['train_ap','train_auc','cv_ap','cv_auc','oof_ap','oof_auc','max_cv_ap']


def original_metrics(year, hd, observed, folder, source):
    # Original launch order is specified in the source benchmark schedule.
    original = source / f'published/2682/task-{1 + 8 * (year-2023)}/output'
    expected = {f: (n, digest) for f,n,digest in load(original.parent/'complete.json')['manifest']}
    hashes = {}
    for name in ('result.json','proboost.ubj','preprocessing.npz','training.csv'):
        p = original / name
        hashes[name] = sha(p)
        assert (p.stat().st_size, hashes[name]) == expected[name]
    info = load(original / 'result.json')
    assert [info[k] for k in ('arm','year','months','mode','trees')] == ['PROBoost',year,12,'IPCW',48]
    cfg = (tuple(info['cfg'][0]), *info['cfg'][1:])
    if year == 2023:
        assert cfg == CONFIG
    prep = arrays(original / 'preprocessing.npz')
    st = tuple(prep[f'arr_{i}'] for i in range(4))
    booster = xgb.Booster()
    booster.load_model(original / 'proboost.ubj')
    booster.set_param({'nthread': 1})
    ix = np.arange(len(hd.y))
    known, y, w = hd.weighted(ix, ix)
    pred = proboost.score(st, booster, hd.x[known], cfg)
    train_ap = average_precision_score(y, pred, sample_weight=w)
    train_auc = roc_auc_score(y, pred, sample_weight=w)
    replay_error = None
    if cfg == CONFIG:
        replay = arrays(folder / 'apparent_private.npz')
        np.testing.assert_array_equal(replay['known'], known)
        np.testing.assert_array_equal(replay['pred'], pred)
        replay_error = 0.
    log = pd.read_csv(original / 'training.csv')
    original_ap = log[(log.stage=='PROBoost_tune') & (log.candidate==proboost.GRID.index(CONFIG))].sort_values('fold')
    actual_ap = observed.iloc[-1][[f'fold{j}_ap' for j in range(1,4)]].to_numpy(float)
    np.testing.assert_allclose(original_ap.score, actual_ap, atol=TOL, rtol=0)
    # Count positive-control score overlaps directly, independent of floating-point metric implementations.
    dif = pred[y==1,None] - pred[None,y==0]
    perfect = (dif>0).all()
    assert bool(perfect) == bool(abs(train_ap-1) <= TOL)
    result = dict(year=year,companies=len(ix),known=len(y),events=int(y.sum()),months=12,
                  beta_start=cfg[0][0],beta_end=cfg[0][1],c=cfg[1],depth=cfg[2],rounds=48,eta=.075,
                  config_matches_fixed=cfg==CONFIG,original_train_ap=train_ap,original_train_auc=train_auc,
                  train_ap_display_3dp=f'{train_ap:.3f}',all_events_strictly_outrank_controls=bool(perfect),
                  incorrectly_ranked_pairs=int((dif<0).sum()),tied_event_control_pairs=int((dif==0).sum()),
                  event_control_pairs=dif.size,prediction_replay_max_abs_error=replay_error)
    return result, dict(sha256=hashes, max_abs_error_archived_three_fold_fixed_config_ap=float(np.max(abs(original_ap.score.to_numpy()-actual_ap))))


def peaks(t):
    result = []
    for year, group in t.groupby('year'):
        for metric in ('cv_ap','cv_auc'):
            values = group[metric].to_numpy()
            best = values.max()
            ties = np.flatnonzero(abs(values - best) <= TOL)
            j = ties[0]
            later = values[j+1:]
            result.append(dict(year=year,metric=metric,best_round=int(group.iloc[j]['round']),
                               best_value=best, ties=len(ties),last_value=values[-1],last_minus_peak=values[-1]-best,
                               ap_at_first_peak=group.iloc[j].cv_ap,auc_at_first_peak=group.iloc[j].cv_auc,
                               later_rounds=len(later),later_rounds_lower=int(sum(later < best-TOL)),
                               min_later=later.min() if len(later) else None))
    return pd.DataFrame(result)


def main():
    r = Path(os.environ['DIAG_RUN'])
    source = Path(os.environ['ESWA_RUN'])
    snapshot = load(r/'snapshot.json')
    folds = []
    results = []
    verified = {}
    originals = []
    per_host = {}
    files = sorted(r.glob('published/*/task-*/output/batch.json'))
    assert len(files) == snapshot['tasks']
    for path in files:
        folder = path.parent
        for name,n,digest in load(folder.parent/'complete.json')['manifest']:
            p = folder/name
            assert p.stat().st_size == n and sha(p)==digest
        b = load(path)
        assert [row['rep'] for row in b['rows']] == list(range(b['start'],b['stop']))
        results.extend(b['rows'])
        per_host[b['host']] = per_host.get(b['host'],0) + len(b['rows'])
        if b['start']==0:
            t = pd.read_csv(folder/'rounds.csv')
            year = b['year']
            assert len(t)==48 and list(t['round'])==list(range(1,49))
            hd = Horizon(arrays(r/f'input/{year}/train.npz'),12,'IPCW')
            orig, check = original_metrics(year,hd,t,folder,source)
            originals.append(orig)
            verified[year] = check
            folds.append(t)
    df = pd.DataFrame(results).sort_values(['year','rep']).reset_index(drop=True)
    assert list(zip(df.year,df.rep)) == [(year,rep) for year in YEARS for rep in range(201)]
    for year in YEARS:
        p = r/f'input/{year}/train.npz'
        assert sha(p) == snapshot['training_inputs'][str(year)]['training_sha256']
        hd = Horizon(arrays(p),12,'IPCW')
        import hashlib
        for row in df[df.year.eq(year) & df.status.eq('ok')].itertuples():
            pseudo = response(hd.y,year,row.rep)
            assert hashlib.sha256(pseudo.astype('i1').tobytes()).hexdigest() == row.pseudo_y_sha256
            assert sum(pseudo==1) == row.positives == sum(hd.y==1)
    df = df.drop(columns='pseudo_y_sha256')
    df[df.rep.gt(0)].to_csv(D/'PERMUTATION_RUNS.csv',index=False)
    observed = df[df.rep.eq(0)].set_index('year')
    t = pd.concat(folds).sort_values(['year','round']).reset_index(drop=True)
    t.to_csv(D/'ROUND_CURVES.csv', index=False)
    peaks(t).to_csv(D/'PEAK_ROUNDS.csv', index=False)
    pd.DataFrame(originals).sort_values('year').to_csv(D/'ORIGINAL_TRAINING.csv',index=False)
    rows = []
    for year in YEARS:
        g = df[(df.year==year)&(df.rep>0)&df.status.eq('ok')]
        for metric in KEYS:
            x = g[metric].dropna().to_numpy()
            v = observed.at[year,metric]
            q = np.quantile(x,[0,.05,.5,.95,1.]) if len(x) else [np.nan]*5
            above = int(sum(x>=v-TOL))
            rows.append(dict(year=year,metric=metric,real_labels=v,scheduled=200,finite=len(x),undefined=200-len(x),
                             minimum=q[0],p05=q[1],median=q[2],p95=q[3],maximum=q[4],
                             count_at_one=int(sum(abs(1-x)<=TOL)),at_least_observed=above,
                             mc_upper_plus_one=(above+1)/201 if len(x)==200 else None))
    pd.DataFrame(rows).to_csv(D/'PERMUTATION_DISTRIBUTION.csv',index=False)
    dump(D/'VALIDATION.json',dict(archive_checks=verified,full_training_snapshot_checks=snapshot['training_inputs'],
                                  dataset_executions_by_node=per_host,requested=len(df),successful=int(df.status.eq('ok').sum()),
                                  failure_count=int(df.status.ne('ok').sum()),fixed_config_id=proboost.GRID.index(CONFIG),
                                  outer_evaluation_files_read_by_diagnostic=0,
                                  final_scores_verified_against_sklearn='Every full-fit, held-out fold and pooled OOF AP/AUC, tolerance 1e-12.'))


if __name__ == '__main__':
    main()
