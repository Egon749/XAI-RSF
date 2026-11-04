from dataclasses import asdict
from itertools import product
from experiments.post_st_requested_seeds14_v2.common import (
    SEEDS, NEW_SEEDS, REUSED_SEEDS, YEARS, TREES, configuration, task_rows,
)


def test_exact_grid_with_pilot_counted_once_and_no_refits_of_saved_seeds():
    requested = [98, 17, 503, 5001, 2570, 22, 155, 150, 203, 153, 119, 188, 80, 120]
    assert SEEDS == requested and len(set(SEEDS)) * TREES == 448
    assert REUSED_SEEDS == requested[:7] and NEW_SEEDS == requested[7:]
    fits = task_rows('pilot') + task_rows('formal')
    assert len(fits) == 21
    assert {(row['year'], row['seed']) for row in fits} == set(product(YEARS, NEW_SEEDS))
    assert {(row['year'], row['version']) for row in task_rows('ensemble')} == set(product(YEARS, ('raw', 'shrunk')))


def test_all_seed_configurations_match_reference_except_rng():
    reference = asdict(configuration(42, 32))
    for seed in SEEDS:
        assert asdict(configuration(seed, TREES)) == {**reference, 'seed': seed}
