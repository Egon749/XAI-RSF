"""Independent checks against archived folds, direct score pairs and grouped thresholds."""
import pickle
import numpy as np
import pandas as pd
from experiments.post_st_eswa_baselines_v1.common import Horizon, arrays, dump, load, sha
from experiments.post_st_benchmark_calibration_v1.common import old
from .evaluate import ALGOS, YEARS
from .run import D, ROOT, run


def independent_metrics(parts):
    p, y, w = (np.concatenate([d[key] for d in parts]) for key in ('score', 'y', 'w'))
    pos = (y == 1) & (w > 0)
    neg = (y == 0) & (w > 0)
    pair_weights = np.outer(w[pos], w[neg])
    difference = p[pos, None] - p[None, neg]
    concordant = ((difference > 0) + .5 * (difference == 0)) * pair_weights
    auc = concordant.sum() / pair_weights.sum() if pair_weights.size else np.nan
    ap = np.nan
    if pos.any():
        ap = 0.0
        for level in np.unique(p[pos]):
            above = p >= level
            added = w[pos & (p == level)].sum() / w[pos].sum()
            ap += added * w[pos & above].sum() / w[above].sum()
    caught = 0
    for d in parts:
        quota = int(np.ceil(len(d['score']) * .2))
        order = sorted(range(len(d['score'])), key=lambda i: (-d['score'][i], d['id'][i]))
        caught += int((d['y'][order[:quota]] == 1).sum())
    return [auc, ap, caught]


def check():
    root = run()
    table = pd.read_csv(D / 'ALL_REPLICATES.csv', dtype={'scope': str}).set_index(['scope', 'months', 'rep', 'model'])
    reference = load(D.parent / 'post_st_eswa_baselines_v1' / 'DATA_ALIGNMENT_AUDIT.json')['hashes']
    cache = {}
    for year in YEARS:
        labels = old('post_st_seed_explanation_stability_v1') / f'input/evaluation_input/{year}_test_PRIVATE.npz'
        assert sha(labels) == reference[f'{year}/labels']
        for role in ('train', 'eval'):
            assert sha(root / 'input' / str(year) / f'{role}.npz') == reference[f'{year}/{role}']
        for h in (12, 24):
            train = Horizon(arrays(root / 'input' / str(year) / 'train.npz'), h, 'IPCW')
            test = Horizon(arrays(labels), h, 'complete')
            cache[year, h] = train, test
    available = {}
    checked = 0
    folds = 0
    checks = 0
    largest_error = 0.
    def compare(key, part, values):
        nonlocal checks, largest_error
        fields = [part + '_' + s for s in ('auc','ap','top20_captured')]
        before = table.loc[key, fields].to_numpy(dtype=float)
        after = np.array(independent_metrics(values))
        np.testing.assert_allclose(before, after, atol=1e-12, rtol=0, equal_nan=True)
        largest_error = max(largest_error, np.nanmax(abs(before - after)))
        checks += 1
    for path in root.glob('published/*/task-*/output/done.json'):
        j = load(path)
        if j['status'] != 'ok':
            assert not (path.parent / 'scores.npz').exists()
            continue
        y, h, b, m = j['year'], j['months'], j['rep'], j['model']
        hd, ext = cache[y, h]
        rows = arrays(path.parent / 'draw.npz')['rows']
        om = np.flatnonzero(np.bincount(rows, minlength=len(hd.x)) == 0)
        z = arrays(path.parent / 'scores.npz')
        if m == 'Stacking':
            with (path.parent / 'fold_indices.pkl').open('rb') as f:
                for _, a, v in pickle.load(f):
                    assert not set(rows[list(a)]).intersection(rows[list(v)])
                    folds += 1
        kk, _, known_weights = hd.weighted(rows, om)
        w = np.zeros(len(om))
        w[np.searchsorted(om, kk)] = known_weights
        key = (str(y), h, b, m)
        for part, val in [('oob', dict(score=z['oob'], id=z['oob_ids'], y=hd.y[om], w=w)),
                          ('test', dict(score=z['test'], id=z['test_ids'], y=ext.y, w=(ext.y >= 0).astype(float)))]:
            available[y, h, b, m, part] = val
            if (y, h, part) == (2025, 24, 'test'):
                assert table.loc[key, ['test_auc','test_ap','test_top20_captured','gap_auc']].isna().all()
            else:
                compare(key, part, [val])
        checked += 1
    for h, years in [(12, YEARS), (24, YEARS[:2])]:
        for b in range(1, 51):
            for m in ALGOS:
                key = ('pooled', h, b, m)
                for part in ('test', 'oob'):
                    keys = [(y, h, b, m, part) for y in years]
                    if all(k in available for k in keys):
                        compare(key, part, [available[k] for k in keys])
                    else:
                        assert table.loc[key, 'status'] == 'incomplete' and pd.isna(table.loc[key, part + '_auc'])
    dump(D / 'VERIFICATION.json', dict(successful_fits_verified=checked, archived_chen_folds_checked=folds,
         cross_fold_company_overlaps=0, independently_checked_split_metric_triples=checks,
         max_abs_difference_auc_ap_or_captures=largest_error, input_and_label_hashes_match_baseline=True,
         check='Direct event-control pairs; AP by positive thresholds; Top20 via sorted tuples; all estimable annual and pooled replicates.'))


if __name__ == '__main__':
    check()
