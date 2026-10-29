"""Bootstrap / random-split / AJ reference engine. No hazard regression layer.

The logrankCR statistic uses the inclusive competing-risk risk set described in
Ishwaran et al. (2014), with the variance convention of randomForestSRC 3.9.0.
It is a splitting score, not an inferential test on clustered records.
"""
from dataclasses import dataclass, asdict
import json
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .partition import (PartitionLedger, split_region, training_cut_dictionary,
                        node_shared_candidates, tree_feature_subset)


def aalen_johansen(time, event, grid, n_causes=2):
    time, event, grid = np.asarray(time, float), np.asarray(event), np.asarray(grid, float)
    if len(time) == 0 or len(time) != len(event):
        raise ValueError("nonempty aligned outcomes required")
    if (time.ndim != 1 or event.ndim != 1 or grid.ndim != 1 or
        not np.isfinite(time).all() or not np.isfinite(grid).all() or
        not np.isfinite(event).all() or np.any(time <= 0) or
        np.any(event != np.floor(event)) or np.any((event < 0) | (event > n_causes))):
        raise ValueError("invalid time/event/grid")
    event = event.astype(int)
    times, inv, counts = np.unique(time, return_inverse=True, return_counts=True)
    risk = np.cumsum(counts[::-1])[::-1].astype(float)
    deaths = np.column_stack([np.bincount(inv, weights=event == k, minlength=len(times))
                              for k in range(1, n_causes + 1)])
    frac = deaths.sum(1) / risk
    before = np.r_[1., np.cumprod(1. - frac)[:-1]]
    values = np.cumsum(before[:, None] * deaths / risk[:, None], axis=0)
    ix = np.searchsorted(times, grid, side="right") - 1
    ans = np.zeros((len(grid), n_causes))
    ok = ix >= 0
    ans[ok] = values[ix[ok]]
    return ans


def cluster_bootstrap(groups, seed, tree):
    """Draw G company groups with replacement, retaining every member record."""
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(np.random.SeedSequence([seed, tree, 0]))
    group_counts = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
    return np.repeat(np.arange(len(groups)), group_counts[inverse])


def _split_parent(time, event, n_causes):
    times, bins, count = np.unique(time, return_inverse=True, return_counts=True)
    risk = np.cumsum(count[::-1])[::-1].astype(float)
    deaths = np.array([np.bincount(bins, weights=event == k, minlength=len(times))
                       for k in range(1, n_causes + 1)])
    other = deaths.sum(0)[None, :] - deaths
    inclusive = risk[None, :] + np.cumsum(other, axis=1) - other
    return bins, risk, deaths, inclusive


def _cause_weights(weights, n_causes):
    if weights is None:
        return np.ones(n_causes)
    w = np.asarray(weights, dtype=float)
    if (w.shape != (n_causes,) or not np.isfinite(w).all() or
            np.any(w < 0) or not np.any(w > 0)):
        raise ValueError('cause_weights requires one finite nonnegative weight per cause, with at least one positive')
    return w


def logrank_cr_score(left, event, parent, cause_weights=None):
    """Absolute weighted score / variance, inclusive risk sets, tied-time correction."""
    bins, risk, deaths, inclusive = parent
    size = len(risk)
    lc = np.bincount(bins, weights=left, minlength=size)
    lr = np.cumsum(lc[::-1])[::-1]
    ld = np.array([np.bincount(bins, weights=left & (event == k + 1), minlength=size)
                   for k in range(len(deaths))])
    lo = ld.sum(0)[None, :] - ld
    li = lr[None, :] + np.cumsum(lo, axis=1) - lo
    prop = np.divide(li, inclusive, out=np.zeros_like(li), where=inclusive > 0)
    num = (ld - deaths * prop).sum(1)
    correction = np.divide(inclusive - deaths, inclusive - 1,
                           out=np.zeros_like(inclusive), where=inclusive > 1)
    var = (deaths * prop * (1 - prop) * correction * (risk[None, :] >= 2)).sum(1)
    w = _cause_weights(cause_weights, len(deaths))
    numerator, variance = abs(w @ num), (w ** 2) @ var
    return float(numerator / np.sqrt(variance)) if variance > 1e-18 else 0.


@dataclass(frozen=True)
class ForestConfig:
    n_trees: int = 64
    max_depth: int = 8
    path_budget: int | None = None
    min_leaf: int = 30
    min_events: int = 3
    min_groups: int = 10
    mtry: int | None = None
    n_thresholds: int = 10
    n_causes: int = 2
    seed: int = 42
    cut_strategy: str = 'local_random'
    global_cell_budget: int | None = None
    tree_feature_count: int | None = None
    cause_weights: tuple[float, ...] | None = None
    tree_feature_strategy: str = 'balanced'

    def __post_init__(self):
        if self.tree_feature_strategy not in ('balanced', 'iid'):
            raise ValueError('unknown tree feature strategy')
        if self.cause_weights is not None:
            object.__setattr__(self, 'cause_weights', tuple(_cause_weights(self.cause_weights, self.n_causes)))


def _grow_tree(cfg, grid, X, time, event, groups, tree_id, shared_cuts=None, ledger=None, budget_log=None):
    n, p = X.shape
    evaluations = 0
    rows = cluster_bootstrap(groups, cfg.seed, tree_id)
    tree_features = tree_feature_subset(p, cfg.tree_feature_count, cfg.seed, tree_id,
                                       cfg.tree_feature_strategy)
    nodes = []

    def grow(rows, used, depth, node_key, region):
        nonlocal evaluations
        index = len(nodes)
        node = dict(ancestors=sorted(used), depth=depth, n=len(rows),
                    groups=int(len(np.unique(groups[rows]))), events=int((event[rows] > 0).sum()))
        node['cause_event_rows'] = [int((event[rows] == k).sum()) for k in range(1, cfg.n_causes+1)]
        node['cause_company_support'] = [int(len(np.unique(groups[rows[event[rows] == k]]))) for k in range(1, cfg.n_causes+1)]
        nodes.append(node)

        def leaf():
            node['cif'] = aalen_johansen(time[rows], event[rows], grid, cfg.n_causes).tolist()
            return index

        if (depth >= cfg.max_depth or len(rows) < 2 * cfg.min_leaf or
            node['events'] < 2 * cfg.min_events or node['groups'] < 2 * cfg.min_groups):
            return leaf()
        eligible = tree_features if cfg.path_budget is None or len(used) < cfg.path_budget else np.array(sorted(used))
        rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, tree_id, node_key, 1]))
        mtry = cfg.mtry or int(np.ceil(np.sqrt(p)))
        features = rng.choice(eligible, min(mtry, len(eligible)), replace=False)
        parent = _split_parent(time[rows], event[rows], cfg.n_causes)
        best, best_left, best_split = 0., None, None
        rejected = 0
        best_rejected = None
        before_cells = ledger.cells if ledger is not None else None
        for j in features:
            x = X[rows, j]
            missing = np.isnan(x)
            vals = np.unique(x[~missing])
            if shared_cuts is None:
                cuts = vals[:-1]  # <= observed threshold, no rounding/merging
                if len(cuts) > cfg.n_thresholds:
                    cuts = rng.choice(cuts, cfg.n_thresholds, replace=False)
            elif cfg.cut_strategy == 'shared_refined':
                cuts = node_shared_candidates(shared_cuts[j], x, cfg.n_thresholds)
            else:
                cuts = shared_cuts[j][(shared_cuts[j] >= vals[0]) & (shared_cuts[j] < vals[-1])] if len(vals) else np.array([])
            candidates = [(float(c), route) for c in sorted(cuts)
                          for route in ([False, True] if missing.any() else [False])]
            if missing.any() and len(vals):
                candidates.append((-np.inf, True))  # missing-only left child
            for cut, missing_left in candidates:
                left = (x <= cut) | (missing & missing_left)
                evaluations += 1
                if min(left.sum(), (~left).sum()) < cfg.min_leaf:
                    continue
                if any((event[rows[side]] > 0).sum() < cfg.min_events for side in (left, ~left)):
                    continue
                if any(len(np.unique(groups[rows[side]])) < cfg.min_groups for side in (left, ~left)):
                    continue
                score = logrank_cr_score(left, event[rows], parent, cfg.cause_weights)
                if ledger is not None:
                    proposal = ledger.project(split_region(region, int(j), cut, bool(missing_left)))
                    if not ledger.accepts(proposal):
                        rejected += 1
                        ledger.rejected_candidates += 1
                        if best_rejected is None or score > best_rejected['score']:
                            best_rejected = dict(score=score, proposed_cells=proposal[0],
                                                 feature=int(j), threshold=cut, missing_left=bool(missing_left))
                        continue
                if score > best + 1e-14:
                    best, best_left, best_split = score, left, (int(j), cut, bool(missing_left))
        if ledger is not None and budget_log is not None:
            budget_log.append(dict(tree=tree_id, node=node_key, depth=depth, cells_before=before_cells,
                                   rejected_candidates=rejected, best_rejected=best_rejected,
                                   selected_score=best if best_split is not None else None))
        if best_split is None:
            return leaf()
        j, cut, route = best_split
        children = split_region(region, j, cut, route)
        if ledger is not None:
            ledger.commit(ledger.project(children))
            if budget_log is not None:
                budget_log[-1]['cells_after'] = ledger.cells
        node.update(feature=j, threshold=cut, missing_left=route, score=best)
        node['left'] = grow(rows[best_left], used | {j}, depth + 1, node_key * 2, children[0])
        node['right'] = grow(rows[~best_left], used | {j}, depth + 1, node_key * 2 + 1, children[1])
        return index

    grow(rows, set(), 0, 1, {})
    return nodes, np.bincount(rows, minlength=n), evaluations


def physical_worker_cpus():
    """Discover permitted physical cores; prefer P cores, never expand cgroups."""
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count() or 1))
    cores = {}
    for c in allowed:
        topology = Path(f'/sys/devices/system/cpu/cpu{c}/topology')
        if topology.exists():
            key = ((topology/'physical_package_id').read_text().strip(), (topology/'core_id').read_text().strip())
            siblings = (topology/'thread_siblings_list').read_text().strip()
            is_p = ',' in siblings or '-' in siblings
            freq = Path(f'/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq')
            frequency = int(freq.read_text()) if freq.exists() else 0
        else:
            key, is_p, frequency = (0, c), False, 0
        cores.setdefault(key, (not is_p, -frequency, c))
    return [c for _, _, c in sorted(cores.values())]


_TREE_CONTEXT = None


def _initialize_tree_worker(cpu_queue, cfg, grid, X, time, event, groups, shared_cuts):
    global _TREE_CONTEXT
    cpu = cpu_queue.get()
    if hasattr(os, 'sched_setaffinity'):
        if cpu not in os.sched_getaffinity(0):
            raise RuntimeError('worker CPU outside inherited scheduler affinity')
        os.sched_setaffinity(0, {cpu})
    # Limit native libraries too; process-level parallelism is the only parallelism.
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    _TREE_CONTEXT = (cfg, grid, X, time, event, groups, shared_cuts)


def _parallel_tree(tree_id):
    return _grow_tree(*_TREE_CONTEXT[:-1], tree_id, shared_cuts=_TREE_CONTEXT[-1])


class CompetingRiskForest:
    def __init__(self, config=ForestConfig()):
        self.config = config

    def fit(self, X, time, event, groups=None, feature_sources=None, n_jobs=1):
        X, time, event = np.asarray(X, float), np.asarray(time, float), np.asarray(event)
        if X.ndim != 2 or X.shape[1] == 0 or time.ndim != 1 or event.ndim != 1:
            raise ValueError("aligned matrix and outcome vectors required")
        n, p = X.shape
        if np.isinf(X).any() or len(time) != n or len(event) != n or n == 0:
            raise ValueError("invalid predictors/outcomes")
        if not np.isfinite(time).all() or np.any(time <= 0):
            raise ValueError("fit requires positive finite follow-up")
        if not np.isfinite(event).all() or np.any(event != np.floor(event)) or np.any((event < 0) | (event > self.config.n_causes)):
            raise ValueError("invalid event code")
        # This first version accepts one declared original coordinate per column.
        # Native missingness belongs to that coordinate. Extra indicators/derived
        # multivariate columns cannot be presented as independent original inputs.
        sources = [(j,) for j in range(p)] if feature_sources is None else list(feature_sources)
        if len(sources) != p or any(len(s) != 1 for s in sources) or len({s[0] for s in sources}) != p:
            raise ValueError("one unique original source per column; no composite/duplicate indicators")
        cfg = self.config
        if cfg.path_budget not in (None, 1, 2, 3, 4) or cfg.n_trees < 1 or cfg.min_leaf < 1:
            raise ValueError("invalid prototype configuration")
        if (cfg.max_depth < 0 or cfg.min_events < 0 or cfg.min_groups < 1 or
            cfg.n_thresholds < 1 or cfg.n_causes < 1 or
            (cfg.mtry is not None and not 1 <= cfg.mtry <= p)):
            raise ValueError("invalid forest budget")
        if cfg.cut_strategy not in ('local_random', 'shared_quantile', 'shared_refined'):
            raise ValueError('unknown cut strategy')
        if cfg.tree_feature_count is not None and (
                isinstance(cfg.tree_feature_count, bool) or not isinstance(cfg.tree_feature_count, int) or
                not 1 <= cfg.tree_feature_count <= p):
            raise ValueError('tree feature count must be an integer in [1, p]')
        if cfg.global_cell_budget is not None and (
                isinstance(cfg.global_cell_budget, bool) or not isinstance(cfg.global_cell_budget, int) or
                cfg.global_cell_budget < 0 or cfg.path_budget not in (1, 2)):
            raise ValueError('global cell budget requires nonnegative integer and path order 1 or 2')
        self.feature_sources_ = sources
        self.n_features_ = p
        groups = np.arange(n) if groups is None else np.asarray(groups)
        if len(groups) != n:
            raise ValueError("invalid group length")
        event = event.astype(int)
        self.grid_ = np.unique(time[event > 0])
        if not len(self.grid_):
            self.grid_ = np.array([0.])
        self.trees_, self.inbag_, self.candidate_evaluations_ = [], [], 0
        if not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
            raise ValueError('n_jobs must be a positive integer or -1')
        cpus = physical_worker_cpus()
        workers = min(len(cpus), cfg.n_trees) if n_jobs == -1 else n_jobs
        if workers > len(cpus):
            raise ValueError('requested workers exceed permitted physical cores; request a multi-CPU Slurm allocation')
        if cfg.global_cell_budget is not None and workers != 1:
            raise ValueError('global partition budget requires deterministic sequential tree order; parallelize independent fits')
        self.shared_cuts_ = training_cut_dictionary(X, cfg)
        ledger = PartitionLedger(cfg.global_cell_budget) if cfg.global_cell_budget is not None else None
        self.budget_log_ = []
        if workers == 1:
            results = (_grow_tree(cfg, self.grid_, X, time, event, groups, i, self.shared_cuts_, ledger, self.budget_log_) for i in range(cfg.n_trees))
            for nodes, inbag, evaluations in results:
                self.trees_.append(nodes); self.inbag_.append(inbag)
                self.candidate_evaluations_ += evaluations
        else:
            context = mp.get_context('spawn')
            queue = context.Queue()
            for cpu in cpus[:workers]:
                queue.put(cpu)
            with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                    initializer=_initialize_tree_worker,
                    initargs=(queue, cfg, self.grid_, X, time, event, groups, self.shared_cuts_)) as executor:
                # map preserves tree index order regardless of P/E completion order.
                for nodes, inbag, evaluations in executor.map(_parallel_tree, range(cfg.n_trees)):
                    self.trees_.append(nodes); self.inbag_.append(inbag)
                    self.candidate_evaluations_ += evaluations
            queue.close(); queue.join_thread()
        self.partition_audit_ = ledger.summary() if ledger is not None else None
        self.parallelism_ = dict(workers=workers, permitted_physical_cores=len(cpus),
                                 cpu_policy='dynamic physical cores, P before E; no SMT sibling duplication')
        self.audit_ = self.audit()
        return self

    def predict(self, X, times=None):
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[1] != self.n_features_ or np.isinf(X).any():
            raise ValueError("invalid prediction matrix")
        times = self.grid_ if times is None else np.asarray(times, float)
        if times.ndim != 1 or not np.isfinite(times).all():
            raise ValueError("finite time vector required")
        ix = np.searchsorted(self.grid_, times, side='right') - 1
        out = np.zeros((len(X), len(times), self.config.n_causes))
        for nodes in self.trees_:
            def visit(node_id, ids):
                node = nodes[node_id]
                if 'cif' in node:
                    values = np.asarray(node['cif'])[np.maximum(ix, 0)].copy()
                    values[ix < 0] = 0.
                    out[ids] += values / len(self.trees_)
                    return
                x = X[ids, node['feature']]
                left = (x <= node['threshold']) | (np.isnan(x) & node['missing_left'])
                if left.any():
                    visit(node['left'], ids[left])
                if (~left).any():
                    visit(node['right'], ids[~left])
            visit(0, np.arange(len(X)))
        return out

    def audit(self):
        paths, depths, cuts, leaves, violations, ancestor_errors = [], [], set(), 0, 0, 0
        for nodes in self.trees_:
            def visit(index, used, depth):
                nonlocal leaves, violations, ancestor_errors
                node = nodes[index]
                ancestor_errors += sorted(used) != node['ancestors']
                if 'cif' in node:
                    paths.append(len(used)); depths.append(depth); leaves += 1
                    violations += self.config.path_budget is not None and len(used) > self.config.path_budget
                    return
                cuts.add((node['feature'], node['threshold']))
                for child in ('left', 'right'):
                    visit(node[child], used | {node['feature']}, depth + 1)
            visit(0, set(), 0)
        if violations or ancestor_errors:
            raise AssertionError("path constraint/ancestry violation")
        return dict(trees=len(self.trees_), leaves=leaves, nodes=sum(map(len, self.trees_)),
                    max_path_variables=max(paths), max_depth=max(depths), distinct_thresholds=len(cuts),
                    path_violations=int(violations), ancestor_errors=int(ancestor_errors),
                    candidate_evaluations=self.candidate_evaluations_)

    def save(self, path):
        path = Path(path)
        with path.open('x') as f:
            json.dump(dict(config=asdict(self.config), grid=self.grid_.tolist(), n_features=self.n_features_,
                           trees=self.trees_, audit=self.audit_,
                           partition_audit=getattr(self, 'partition_audit_', None),
                           shared_cuts=[c.tolist() for c in self.shared_cuts_] if getattr(self, 'shared_cuts_', None) is not None else None), f)

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        obj = cls(ForestConfig(**d['config']))
        obj.grid_, obj.n_features_, obj.trees_, obj.audit_ = np.array(d['grid']), d['n_features'], d['trees'], d['audit']
        obj.candidate_evaluations_ = obj.audit_['candidate_evaluations']
        obj.partition_audit_ = d.get('partition_audit')
        obj.shared_cuts_ = tuple(np.asarray(c) for c in d['shared_cuts']) if d.get('shared_cuts') is not None else None
        obj.audit()
        return obj
