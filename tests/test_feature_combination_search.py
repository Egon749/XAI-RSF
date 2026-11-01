"""Substantive search coverage, chronology and guard regression checks."""
from experiments.post_st_feature_combination_search_v1.search import (
    FULL, LEGACY_REMOVED, AUDIT, anchors, lattice, legal, neighbourhood, centers, decide)


def row(keep, gain, approved=False):
    return dict(keep=list(keep), delta_ap24=gain, delta_capture24=0,
                spatial_cell_change=0, subset_id=str(keep), approved=approved)


def test_complete_lattice_and_historical_admissibility():
    legacy = tuple(j for j in FULL if j not in LEGACY_REMOVED)
    assert len(lattice(2023)) == 256
    assert len(lattice(2024)) == len(lattice(2025)) == 512
    assert legacy not in lattice(2023)
    assert all(58 in keep for keep in lattice(2023))
    for year in [2024, 2025]:
        assert FULL in lattice(year) and legacy in lattice(year)
        assert anchors(year) == [FULL, legacy]


def test_neighbourhood_spans_all_inputs_and_restores_without_partial_audit():
    n = neighbourhood(FULL, [])
    assert len(n) == 77
    assert {tuple(j for j in FULL if j != k) for k in range(76)} <= set(n)
    assert tuple(range(76)) in n
    assert all(legal(k) for k in n)
    assert FULL in neighbourhood(tuple(j for j in FULL if j != 5), [])
    groups = [dict(members=[2, 4, 5], representative_one=[2], representative_two=[2, 5]),
              dict(members=list(AUDIT), representative_one=[76], representative_two=[76,77])]
    n = neighbourhood(FULL, groups)
    assert tuple(j for j in FULL if j not in [2, 4, 5]) in n
    assert all(legal(k) for k in n)


def test_search_can_cross_infeasible_steps_but_final_cannot():
    high = tuple(j for j in FULL if j != 8)
    good = tuple(j for j in FULL if j not in [4, 8])
    rows = [row(high, .3, False), row(good, .1, True)]
    chosen = centers(2024, rows)
    assert set(anchors(2024)) <= set(chosen)
    assert high in chosen and good in chosen
    assert decide(rows, dict(passed=True))['keep'] == list(good)
    assert decide(rows, dict(passed=False))['keep'] == list(FULL)
    assert decide([rows[0]], dict(passed=True))['keep'] == list(FULL)


def test_general_moves_not_future_diagnosis_can_generate_74_in_2023():
    legacy = tuple(j for j in FULL if j not in LEGACY_REMOVED)
    assert legacy not in anchors(2023)
    assert legacy in neighbourhood(anchors(2023)[1], [])


def test_complete_74_candidate_can_win_and_is_not_exempt_from_guards():
    legacy = anchors(2024)[1]
    other = tuple(j for j in FULL if j != 73)
    rows = [row(legacy, .2, True), row(other, .1, True)]
    assert decide(rows, dict(passed=True))['keep'] == list(legacy)
    rows[0]['approved'] = False
    assert decide(rows, dict(passed=True))['keep'] == list(other)
