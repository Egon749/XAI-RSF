"""Diagnose fixed booster prefixes on known historical outcomes; no outer data interface."""
import hashlib
import os
from pathlib import Path
import socket
import sys
import traceback

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb

from experiments.post_st_eswa_baselines_v1 import proboost
from experiments.post_st_eswa_baselines_v1.common import Horizon, arrays, dump, transform

CONFIG = ((2., 1.), .75, 2)
ROUNDS = np.arange(1, 49)
COLS = np.arange(74)


def response(y, year, rep):
    out = y.copy()
    if rep:
        rng = np.random.default_rng(np.random.SeedSequence([20261104, 1212, year, rep]))
        known = y >= 0
        out[known] = rng.permutation(y[known])
    return out


def rankings(y, score, w):
    """Weighted threshold AP and tie-adjusted event-control AUC."""
    order = np.argsort(-score, kind='stable')
    unique = np.r_[0, np.flatnonzero(np.diff(score[order])) + 1]
    pos = np.add.reduceat(w[order] * y[order], unique)
    neg = np.add.reduceat(w[order] * (1 - y[order]), unique)
    if pos.sum() <= 0 or neg.sum() <= 0:
        raise ValueError('both known classes required')
    ap = np.dot(pos / pos.sum(), pos.cumsum() / (pos + neg).cumsum())
    auc = np.dot(neg / neg.sum(), (pos.cumsum() - .5 * pos) / pos.sum())
    return ap, auc


def curves(y, scores, weights):
    values = np.array([rankings(y, s, weights) for s in scores])
    expected = [average_precision_score(y, scores[-1], sample_weight=weights),
                roc_auc_score(y, scores[-1], sample_weight=weights)]
    np.testing.assert_allclose(values[-1], expected, atol=1e-12, rtol=0)
    return values


def fit_prefixes(hd, fit, validate):
    st, booster = proboost.fit(hd, fit, CONFIG)
    keep, y, w = hd.weighted(fit, validate)
    x = hd.x[keep]
    matrix = xgb.DMatrix(transform(x, st, COLS), nthread=1)
    scores = np.array([expit(CONFIG[0][1] * (booster.predict(
        matrix, output_margin=True, iteration_range=(0, int(t))).astype(float) - CONFIG[1]))
        for t in ROUNDS])
    np.testing.assert_array_equal(scores[-1], proboost.score(st, booster, x, CONFIG))
    return keep, y, w, scores, curves(y, scores, w)


def run_one(data, year, rep, out):
    hd = Horizon(data, 12, 'IPCW')
    hd.y = response(hd.y, year, rep)
    ix = np.arange(len(hd.y))
    known, y, w, pred, train = fit_prefixes(hd, ix, ix)
    table = pd.DataFrame({'year': year, 'rep': rep, 'round': ROUNDS,
                          'train_ap': train[:, 0], 'train_auc': train[:, 1]})
    if not rep:
        np.savez(out / 'apparent_private.npz', known=known, pred=pred[-1])
    oof_y, oof_w, oof_p, visits, all_folds = [], [], [], [], []
    folds = hd.split(ix, 3, 741)
    if len(folds) != 3:
        raise ValueError('three distinct-company folds required')
    for fold, (a, b) in enumerate(folds, 1):
        assert not set(hd.d['groups'][a]).intersection(hd.d['groups'][b])
        idx, yv, wv, pv, measures = fit_prefixes(hd, a, b)
        oof_y.append(yv)
        oof_w.append(wv)
        oof_p.append(pv)
        visits.extend(idx)
        all_folds.append(measures)
        table[f'fold{fold}_ap'] = measures[:, 0]
        table[f'fold{fold}_auc'] = measures[:, 1]
        table[f'fold{fold}_events'] = int(yv.sum())
    np.testing.assert_array_equal(np.sort(visits), known)
    pooled = curves(np.concatenate(oof_y), np.concatenate(oof_p, axis=1), np.concatenate(oof_w))
    means = np.mean(all_folds, axis=0)
    table['cv_ap'], table['cv_auc'] = means.T
    table['oof_ap'], table['oof_auc'] = pooled.T
    if not rep:
        table.to_csv(out / 'rounds.csv', index=False)
    result = table.iloc[-1].to_dict()
    result.update(n=len(ix), known=len(known), positives=int(y.sum()),
                  max_cv_ap=means[:, 0].max(), pseudo_y_sha256=hashlib.sha256(hd.y.astype('i1').tobytes()).hexdigest())
    return result


def main(year, start, stop):
    data = arrays(Path(os.environ['DATA_DIR']) / str(year) / 'train.npz')
    out = Path(os.environ['OUTPUT_DIR'])
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for rep in range(start, stop):
        try:
            result = run_one(data, year, rep, out)
            results.append(dict(result, year=year, rep=rep, status='ok'))
        except Exception as exc:
            (out / f'failure-{rep}.txt').write_text(traceback.format_exc())
            results.append(dict(year=year, rep=rep, status='failed', reason=str(exc)))
    dump(out / 'batch.json', dict(year=year, start=start, stop=stop, rows=results,
                                  job=os.environ.get('SLURM_JOB_ID'), host=socket.gethostname()))


if __name__ == '__main__':
    main(*map(int, sys.argv[1:]))
