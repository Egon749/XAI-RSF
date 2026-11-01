"""Finite exploratory search; lack of confirmation is a limitation, not a gate."""
from pathlib import Path
import numpy as np
from crforest import CIFExpansion
from experiments.post_st_group_selection_v1.selection import (
    FULL, grouped_candidates, support_at, discovery_gate, time_gate, performance_guards, interaction_guard)
from experiments.post_st_dropcolumn_selection_v1.common import arrays, TIMES, subset_hash


def select(gate, candidates, evaluator, check_interactions):
    if not gate['passed']:
        return FULL, 'INSUFFICIENT_DISCOVERY_SUPPORT', [], []
    # All declared candidates are actually refitted and scored, including when
    # H12 protection cannot be estimated. An unavailable guard fails closed.
    evaluator.prefetch(candidates)
    scans, interactions = [], []
    for keep in sorted(set(map(tuple, candidates)) - {FULL}):
        contrast = evaluator.contrast(FULL, keep)
        checks = check_interactions(keep)
        failures = performance_guards(contrast)
        if not checks:
            failures.append('INTERACTION_EVIDENCE_UNAVAILABLE')
        elif not all(r['passed'] for r in checks):
            failures.append('INTERACTION_SHAPE_OR_AMPLITUDE')
        row = dict(contrast, failed_guards=failures, approved=not failures,
                   cells=evaluator.summary(keep)['cells'], interaction_checks=len(checks),
                   failed_interaction_checks=sum(not r['passed'] for r in checks))
        scans.append(row)
        interactions.extend(dict(keep=list(keep), **r) for r in checks)
    valid = [r for r in scans if r['approved']]
    if not valid:
        return FULL, 'NO_CANDIDATE_PASSES_FROZEN_GUARDS', scans, interactions
    best = min(valid, key=lambda r: (r['cells'], len(r['keep']), tuple(r['keep'])))
    return tuple(best['keep']), 'EXPLORATORY_SELECTED', scans, interactions


class InteractionCheck:
    """Affected group/external and within-group pairs on a fixed fold reference."""
    def __init__(self, parent, store, cohorts, groups):
        self.parent, self.store, self.cohorts, self.groups = parent, store, cohorts, groups
        self.baselines = {}

    def __call__(self, keep):
        removed = set(FULL) - set(keep)
        group = next(g['members'] for g in self.groups if removed <= set(g['members']))
        result = []
        for cohort in self.cohorts:
            if cohort not in self.baselines:
                data = arrays(self.parent / f'input/folds/{cohort}/train_PRIVATE.npz')
                old = CIFExpansion.load(self.path(FULL, cohort) / 'ensemble/functions_PRIVATE.npz')
                self.baselines[cohort] = data, old
            data, old = self.baselines[cohort]
            new = CIFExpansion.load(self.path(keep, cohort) / 'ensemble/functions_PRIVATE.npz')
            old_blocks, new_blocks = {}, {}
            for expansion, mapping, blocks in [(old, FULL, old_blocks), (new, keep, new_blocks)]:
                for key in expansion.pairs:
                    original = [mapping[j] for j in key]
                    if not set(original) & set(group):
                        continue
                    outside = set(original) - set(group)
                    block = int(next(iter(outside))) if outside else -1
                    blocks.setdefault(block, []).append(key)
            if not old_blocks:
                result.append(dict(cohort=cohort, group=list(group), partner=-2,
                    horizon=0, cause=0, passed=True, reason='NO_BASE_AFFECTED_PAIRS',
                    ratio=None, correlation=None, relative_l2=None))
            for partner, keys in old_blocks.items():
                a = sum(old.component(data['X'], k, TIMES) for k in keys)
                b = sum((new.component(data['X'][:, keep], k, TIMES)
                         for k in new_blocks.get(partner, [])), np.zeros_like(a))
                for h in [12, 24]:
                    ix = np.searchsorted(TIMES, data[f'h{h}']); rr = np.arange(len(ix))
                    for cause in [1, 2]:
                        result.append(dict(cohort=cohort, group=list(group), partner=partner,
                            horizon=h, cause=cause,
                            **interaction_guard(a[rr, ix, cause-1], b[rr, ix, cause-1])))
        return result

    def path(self, keep, cohort):
        return self.store.out / f'{cohort}_{subset_hash(keep)[:16]}'
