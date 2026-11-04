"""Matched comparison and algebraic audit of removing seed 213."""
from itertools import product
from math import ceil

import numpy as np
import pandas as pd

from experiments.post_st_dropcolumn_selection_v1.common import evaluation_context
from experiments.post_st_requested_seeds_v1.analyze import METRICS, fast_metrics
from .common import ENSEMBLE, YEARS, arrays, previous_path, run_path, write


def main():
    root = run_path()
    old = arrays(previous_path() / 'private/risks_PRIVATE.npz')
    current = arrays(root / 'private/risks_PRIVATE.npz')
    labels = {y: arrays(root / f'evaluation_input/{y}_test_PRIVATE.npz') for y in YEARS}
    draws = arrays(root / 'private/bootstrap_indices_PRIVATE.npz')
    max_error = 0.0
    for y, version, h in product(YEARS, ('raw', 'shrunk'), (12, 24)):
        previous = old[f'{y}|ensemble352|{version}|{h}']
        removed = old[f'{y}|213|{version}|{h}']
        actual = current[f'{y}|{ENSEMBLE}|{version}|{h}']
        error = np.max(np.abs(actual - (11 * previous - removed) / 10))
        assert error <= 1e-12
        max_error = max(max_error, float(error))
    comparisons, intervals = [], []
    for h, cause in product((12, 24), (1, 2)):
        years = [y for y in YEARS if evaluation_context(labels[y], h, cause)[3]]
        assert years == (YEARS if h == 12 else [2023, 2024])
        for scope in [*years, 'pooled']:
            selected = years if scope == 'pooled' else [scope]
            truth, year_blocks, ids, before, after, index_blocks = [], [], [], [], [], []
            shared = checked = offset = 0
            for y in selected:
                target, weights, known, supported, _ = evaluation_context(labels[y], h, cause)
                assert supported and known.all() and np.all(weights == 1)
                sample_ids = labels[y]['sample_ids']
                a = old[f'{y}|ensemble352|shrunk|{h}'][:, cause - 1]
                b = current[f'{y}|{ENSEMBLE}|shrunk|{h}'][:, cause - 1]
                quota = ceil(.2 * len(target))
                top_a = np.lexsort((sample_ids, -a))[:quota]
                top_b = np.lexsort((sample_ids, -b))[:quota]
                shared += len(set(top_a).intersection(top_b))
                checked += quota
                truth.append(target)
                year_blocks.append(np.full(len(target), y))
                ids.append(sample_ids)
                before.append(a)
                after.append(b)
                index_blocks.append(draws[str(y)] + offset)
                offset += len(target)
            target, yrs, sample_ids, a, b = map(np.concatenate, (truth, year_blocks, ids, before, after))
            previous = fast_metrics(target, a, yrs, sample_ids)
            actual = fast_metrics(target, b, yrs, sample_ids)
            delta = actual - previous
            comparison = dict(scope=scope, months=h, cause=cause, companies=offset,
                              events=int(target.sum()), checked=checked, shared_top=shared)
            for j, metric in enumerate(METRICS):
                comparison.update({f'previous_{metric}': previous[j], f'current_{metric}': actual[j], f'delta_{metric}': delta[j]})
            comparisons.append(comparison)
            samples = []
            for ix in np.concatenate(index_blocks, axis=1):
                samples.append(fast_metrics(target[ix], b[ix], yrs[ix], sample_ids[ix]) -
                               fast_metrics(target[ix], a[ix], yrs[ix], sample_ids[ix]))
            distribution = np.asarray(samples)
            for j, metric in enumerate(METRICS):
                finite = np.isfinite(distribution[:, j])
                low, high = np.quantile(distribution[finite, j], [.025, .975]) if finite.any() else (np.nan, np.nan)
                intervals.append(dict(scope=scope, months=h, cause=cause, metric=metric, estimate=delta[j],
                                      lower95=low, upper95=high, repeats=len(finite), undefined=int((~finite).sum())))
    write(root/'reports/DELETION_IDENTITY.json', dict(status='PASS', dropped_seed=213, max_error=max_error))
    pd.DataFrame(comparisons).to_csv(root/'reports/PREVIOUS_352_COMPARISON.csv', index=False, mode='x')
    pd.DataFrame(intervals).to_csv(root/'reports/PREVIOUS_352_PAIRED_95.csv', index=False, mode='x')


if __name__ == '__main__':
    main()
