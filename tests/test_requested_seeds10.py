from itertools import product
import pytest
from experiments.post_st_requested_seeds10_v1.common import (
    SEEDS, REUSED_SEEDS, YEARS, N_FITS, N_REUSED, ENSEMBLE_TREES, task_rows,
)


def test_saved_subset_only_exposes_six_distinct_export_tasks():
    assert SEEDS == [98, 17, 503, 5001, 2570, 22, 155, 150, 6720, 8679]
    assert SEEDS == REUSED_SEEDS and N_FITS == 0 and N_REUSED == 30 and ENSEMBLE_TREES == 320
    tasks = [(x['year'], x['version']) for x in task_rows('ensemble')]
    assert len(tasks) == 6 and set(tasks) == set(product(YEARS, ('raw', 'shrunk')))
    with pytest.raises(ValueError):
        task_rows('formal')
