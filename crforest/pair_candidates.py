"""Opt-in joint two-coordinate candidate trees, not a hidden residual learner.

A bounded joint search can enter a pair even when both marginal split scores
vanish. Leaves remain Aalen-Johansen; all paths use exactly the declared pair.
The score is a heuristic four-group survival-separation objective, not a p value.
Existing forest fitting and equal-weight defaults are deliberately untouched.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from itertools import combinations, product
import copy
import numpy as np
from .core import CompetingRiskForest, cluster_bootstrap, aalen_johansen, _split_parent, logrank_cr_score
from .partition import shared_quantile_cuts, tree_feature_subset


@dataclass(frozen=True)
class PairConfig:
    max_pairs: int = 64
    max_trees: int = 16
    max_cuts: int = 3
    min_score: float = 0.
    seed: int = 20261028

    def __post_init__(self):
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1
               for v in (self.max_pairs, self.max_trees, self.max_cuts)):
            raise ValueError('Positive finite integer pair budgets required')
        if self.max_trees > self.max_pairs or not np.isfinite(self.min_score) or self.min_score < 0:
            raise ValueError('Invalid pair selection budget / score')


def _pair_language(forest, p, maximum, seed):
    """Half bounded slots reuse fitted path pairs; remainder cover unseen pairs.

    Both sources use structure-training data ONLY. No validation score enters.
    Deterministic shuffle prevents feature order deciding the truncation.
    """
    existing = set()
    for tree in forest.trees_:
        for node in tree:
            if 'cif' in node and len(node['ancestors']) == 2:
                existing.add(tuple(node['ancestors']))
    all_pairs = list(combinations(range(p), 2))
    rng = np.random.default_rng(seed)
    if len(all_pairs) <= maximum:
        return all_pairs
    old = sorted(existing); rng.shuffle(old)
    chosen = old[:maximum // 2]
    rest = [pair for pair in all_pairs if pair not in set(chosen)]
    rng.shuffle(rest)
    return chosen + rest[:maximum - len(chosen)]


def pair_coverage(forest, pairs):
    """Eligibility, actual path occurrence and shared forest dilution opportunity."""
    B, p = len(forest.trees_), forest.n_features_
    rows = []
    for pair in pairs:
        s = set(pair); eligible = paths = 0
        for b, tree in enumerate(forest.trees_):
            # For augmented trees the stored root records the actual explicit pair.
            allowed = tree[0].get('joint_pair')
            if allowed is None:
                allowed = tree_feature_subset(p, forest.config.tree_feature_count, forest.config.seed, b)
            eligible += s.issubset(set(allowed))
            paths += any('cif' in n and s.issubset(n['ancestors']) for n in tree)
        rows.append({'pair': list(pair), 'trees': B, 'eligible_trees': eligible,
                     'trees_with_pair_path': paths, 'pair_path_fraction': paths / B})
    return rows


def augment_pair_bank(forest, X, time, event, groups, config=PairConfig()):
    """Return a NEW bank. Original forest, leaves and predictions stay unchanged.

    Each candidate uses its own reproducible company bootstrap; every one of its
    four cells must meet original record/group/event support. All cuts come from
    the SAME structure-training dictionary. No scaling by inverse coverage.
    """
    X, t, e, g = np.asarray(X, float), np.asarray(time, float), np.asarray(event), np.asarray(groups)
    cfg = forest.config
    if (cfg.path_budget != 2 or forest.audit()['max_path_variables'] > 2 or
            cfg.max_depth < 2 or cfg.global_cell_budget is not None):
        raise ValueError('Pair bank requires q2, depth>=2, no active global cell ledger')
    if (X.ndim != 2 or X.shape[1] != forest.n_features_ or np.isinf(X).any() or
            t.shape != (len(X),) or e.shape != t.shape or g.shape != t.shape or
            not np.isfinite(t).all() or np.any(t <= 0) or not np.isfinite(e).all() or
            np.any(e != np.floor(e)) or np.any((e < 0) | (e > cfg.n_causes))):
        raise ValueError('Invalid structure-training inputs')
    e = e.astype(int)
    expected_grid = np.unique(t[e > 0]); expected_grid = expected_grid if len(expected_grid) else np.array([0.])
    if not np.array_equal(expected_grid, forest.grid_):
        raise ValueError('Pair bank data must match the original structure-training event grid')
    cuts = getattr(forest, 'shared_cuts_', None)
    if cuts is None:
        cuts = shared_quantile_cuts(X, cfg.n_thresholds)
    limited = []
    for c in cuts:
        c = np.asarray(c)
        ids = np.unique(np.linspace(0, len(c) - 1, min(len(c), config.max_cuts)).round().astype(int)) if len(c) else []
        limited.append(c[ids])
    pairs = _pair_language(forest, X.shape[1], config.max_pairs, config.seed)
    candidates, logs = [], []
    for pair_id, (j, k) in enumerate(pairs):
        rows = cluster_bootstrap(g, config.seed, pair_id)
        xx, tt, ee, gg = X[rows], t[rows], e[rows], g[rows]
        parent = _split_parent(tt, ee, cfg.n_causes)
        choices = []
        for feature in (j, k):
            has_missing = np.isnan(xx[:, feature]).any()
            options = [(float(c), bool(r)) for c in limited[feature]
                       for r in ([False, True] if has_missing else [False])]
            if has_missing and np.isfinite(xx[:, feature]).any():
                options.append((-np.inf, True))
            choices.append(options)
        best = None; evaluated = feasible = 0
        for (a, ma), (b, mb) in product(*choices):
            evaluated += 1
            left_j = (xx[:, j] <= a) | (np.isnan(xx[:, j]) & ma)
            left_k = (xx[:, k] <= b) | (np.isnan(xx[:, k]) & mb)
            cells = [left_j & left_k, left_j & ~left_k, ~left_j & left_k, ~left_j & ~left_k]
            if any(mask.sum() < cfg.min_leaf or np.sum(ee[mask] > 0) < cfg.min_events or
                   len(np.unique(gg[mask])) < cfg.min_groups for mask in cells):
                continue
            feasible += 1
            score = sum(float(mask.mean()) * logrank_cr_score(mask, ee, parent) ** 2 for mask in cells)
            if best is None or score > best[0] + 1e-14:
                best = (score, a, ma, b, mb, cells)
        record = {'pair': [int(j), int(k)], 'candidate_index': pair_id,
                  'evaluated_partitions': evaluated, 'supported_partitions': feasible,
                  'score': None if best is None else float(best[0]),
                  'status': 'NO_SUPPORTED_PARTITION' if best is None else 'SCORED'}
        logs.append(record)
        if best is None or best[0] <= config.min_score:
            continue
        score, a, ma, b, mb, cells = best
        def node(mask, ancestors, depth):
            return {'ancestors': sorted(ancestors), 'depth': depth, 'n': int(mask.sum()),
                    'groups': int(len(np.unique(gg[mask]))), 'events': int(np.sum(ee[mask] > 0)),
                    'cause_event_rows': [int(np.sum(ee[mask] == cause)) for cause in range(1, cfg.n_causes + 1)],
                    'cause_company_support': [int(len(np.unique(gg[mask & (ee == cause)]))) for cause in range(1, cfg.n_causes + 1)]}
        root = node(np.ones(len(rows), bool), [], 0)
        root.update(feature=int(j), threshold=a, missing_left=ma, score=float(score),
                    left=1, right=4, joint_pair=[int(j), int(k)], joint_score_scope='four_group_survival_heuristic')
        left = node(cells[0] | cells[1], [j], 1)
        right = node(cells[2] | cells[3], [j], 1)
        left.update(feature=int(k), threshold=b, missing_left=mb, score=float(score), left=2, right=3)
        right.update(feature=int(k), threshold=b, missing_left=mb, score=float(score), left=5, right=6)
        leaves = []
        for mask in cells:
            leaf = node(mask, [j, k], 2)
            leaf['cif'] = aalen_johansen(tt[mask], ee[mask], forest.grid_, cfg.n_causes).tolist()
            leaves.append(leaf)
        tree = [root, left, leaves[0], leaves[1], right, leaves[2], leaves[3]]
        candidates.append((score, pair_id, tree, np.bincount(rows, minlength=len(X))))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    selected = candidates[:config.max_trees]; chosen = {v[1] for v in selected}
    for record in logs:
        record['selected'] = record['candidate_index'] in chosen
    augmented = copy.copy(forest)
    augmented.trees_ = list(forest.trees_) + [v[2] for v in selected]
    augmented.inbag_ = list(getattr(forest, 'inbag_', [])) + [v[3] for v in selected]
    augmented.config = replace(cfg, n_trees=len(augmented.trees_))
    augmented.candidate_evaluations_ = forest.candidate_evaluations_ + sum(x['evaluated_partitions'] for x in logs)
    augmented.audit_ = augmented.audit()
    audit = {'config': asdict(config), 'original_trees': len(forest.trees_),
             'added_trees': len(selected), 'total_trees': len(augmented.trees_),
             'candidate_log_PRIVATE': logs, 'pair_coverage_before': pair_coverage(forest, pairs),
             'pair_coverage_after': pair_coverage(augmented, pairs),
             'new_leaf_estimator': 'unchanged Aalen-Johansen',
             'selection': 'bounded joint pair search; not global optimum, significance or causal discovery',
             'global_cell_budget': 'unsupported combination rejected, not silently bypassed'}
    return augmented, audit


def extend_ordinary_bank(forest, X, time, event, groups, count):
    """Tree-count matched control: append ordinary trees, not pair candidates."""
    from .core import _grow_tree
    if isinstance(count, bool) or not isinstance(count, int) or count < 0 or forest.config.global_cell_budget is not None:
        raise ValueError('Invalid extension count or unsupported global ledger')
    result = copy.copy(forest)
    result.trees_ = list(forest.trees_)
    result.inbag_ = list(getattr(forest, 'inbag_', []))
    result.candidate_evaluations_ = forest.candidate_evaluations_
    for tree_id in range(len(forest.trees_), len(forest.trees_) + count):
        nodes, inbag, evaluations = _grow_tree(forest.config, forest.grid_, np.asarray(X),
             np.asarray(time), np.asarray(event), np.asarray(groups), tree_id,
             getattr(forest, 'shared_cuts_', None))
        result.trees_.append(nodes); result.inbag_.append(inbag)
        result.candidate_evaluations_ += evaluations
    result.config = replace(forest.config, n_trees=len(result.trees_))
    result.audit_ = result.audit()
    return result
