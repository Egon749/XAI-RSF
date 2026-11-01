"""Finite combination lattice and non-monotone neighbourhood exploration.

Only historical scores reach this module. Search traversal does not require
each intermediate deletion to improve a score. Final acceptance is separate.
"""
import itertools
import math

from experiments.post_st_dropcolumn_selection_v1.common import subset_hash

FULL = tuple(range(83))
AUDIT = frozenset(range(76, 83))
LEGACY_REMOVED = (2, 4, 5, 20, 26, 39, 58, 60, 71)


def historical_removed(year):
    return tuple(j for j in LEGACY_REMOVED if year >= 2024 or j != 58)


def canonical(keep):
    return tuple(sorted(set(keep)))


def legal(keep):
    s = set(keep)
    return bool(s) and s <= set(FULL) and (not (s & AUDIT) or AUDIT <= s)


def lattice(year):
    """All combinations in the historically supported redundancy subspace."""
    removed = historical_removed(year)
    return sorted({tuple(j for j in FULL if j not in chosen)
                   for n in range(len(removed) + 1)
                   for chosen in itertools.combinations(removed, n)})


def anchors(year):
    return [FULL, tuple(j for j in FULL if j not in historical_removed(year))]


def neighbourhood(keep, groups):
    """Every single toggle, atomic audit toggle, and group replacement.

Partial audit-state sets are forbidden. No current performance is consulted.
All original input IDs are used; column-count changes cannot relabel a field.
"""
    s = set(keep)
    units = [{j} for j in range(76)] + [set(AUDIT)]
    candidates = {canonical(s ^ u) for u in units}
    for g in groups:
        members = set(g['members'])
        if members == AUDIT:
            options = [[], sorted(AUDIT)]
        else:
            options = [[], sorted(members), g['representative_one'],
                       g['representative_two']] + [[j] for j in sorted(members)]
        for retained in options:
            candidates.add(canonical((s - members) | set(retained)))
    return sorted(k for k in candidates if legal(k) and k != tuple(keep))


def rank_key(row):
    def finite(key, default):
        value = row.get(key, float('nan'))
        return value if math.isfinite(value) else default
    return (finite('delta_ap24', -math.inf),
            finite('delta_capture24', -math.inf),
            -finite('spatial_cell_change', math.inf), row['subset_id'])


def centers(year, rows, width=4):
    """Both reference anchors plus best raw/feasible historical candidates.

Always retaining both anchors avoids losing the 74/75 branch after an early
score comparison. Infeasible intermediate candidates remain explorable.
"""
    chosen = anchors(year)
    ranked = sorted(rows, key=rank_key, reverse=True)
    for pool in [ranked, [r for r in ranked if r['approved']], ranked]:
        for row in pool:
            key = tuple(row['keep'])
            if key not in chosen:
                chosen.append(key)
                break
        if len(chosen) >= width:
            break
    return chosen[:width]


def decide(rows, support):
    if not support['passed']:
        return dict(keep=list(FULL), status='FALLBACK_INSUFFICIENT_SUPPORT',
                    reason='Historical H24 event/cohort support gate failed')
    feasible = [r for r in rows if r['approved']]
    if not feasible:
        return dict(keep=list(FULL), status='FALLBACK_NO_FEASIBLE_GAIN',
                    reason='No candidate passes every frozen guard against full83')
    best = max(feasible, key=rank_key)
    return dict(keep=best['keep'], status='SELECTED_HISTORICAL_FEASIBLE',
                reason='Highest historical AP24 among evaluated feasible candidates',
                evidence=best)
