"""Compare matched company rankings to the preceding 544-tree ensemble."""
from math import ceil

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.post_st_dropcolumn_selection_v1.common import evaluation_context
from .common import ENSEMBLE, YEARS, arrays, previous_path, run_path


def main():
    private = run_path()
    current = arrays(private / 'private/risks_PRIVATE.npz')
    old = arrays(previous_path() / 'private/risks_PRIVATE.npz')
    comparisons = []
    for year in YEARS:
        labels = arrays(private / f'evaluation_input/{year}_test_PRIVATE.npz')
        previous_labels = arrays(previous_path() / f'evaluation_input/{year}_test_PRIVATE.npz')
        np.testing.assert_array_equal(labels['sample_ids'], previous_labels['sample_ids'])
        for months in (12, 24):
            outcomes, weights, known, supported, _ = evaluation_context(labels, months, 1)
            if not supported:
                assert (year, months) == (2025, 24)
                continue
            assert known.all() and np.all(weights == 1)
            risk = current[f'{year}|{ENSEMBLE}|shrunk|{months}'][:, 0]
            previous_risk = old[f'{year}|ensemble544|shrunk|{months}'][:, 0]
            quota = ceil(0.2 * len(outcomes))
            top = np.lexsort((labels['sample_ids'], -risk))[:quota]
            previous_top = np.lexsort((labels['sample_ids'], -previous_risk))[:quota]
            comparisons.append(dict(
                year=year, months=months, n=len(outcomes), checked=quota,
                shared_top=len(set(top).intersection(previous_top)),
                current_captured=int(outcomes[top].sum()), last_captured=int(outcomes[previous_top].sum()),
                previous_auc=roc_auc_score(outcomes, previous_risk),
                previous_ap=average_precision_score(outcomes, previous_risk),
                previous_brier=np.mean((previous_risk - outcomes) ** 2),
            ))
    result = pd.DataFrame(comparisons)
    path = private / 'reports/PREVIOUS_544_TOP20_COMPARISON.csv'
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_csv(path), result, atol=1e-13, rtol=0)
    else:
        result.to_csv(path, index=False, mode='x')


if __name__ == '__main__':
    main()
