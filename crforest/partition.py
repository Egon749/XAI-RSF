"""Training-only cut dictionary and monotone whole-forest spatial-cell ledger.

The ledger counts the same unmerged product-reference function grids as the
compiler, including one missing state per coordinate. It contains no outcome
information and does not remove predictions after fitting.
"""
import numpy as np


def shared_quantile_cuts(X, count):
    quantiles = np.arange(1, count + 1) / (count + 1)
    answer = []
    for x in np.asarray(X, float).T:
        v = x[np.isfinite(x)]
        cuts = np.unique(np.quantile(v, quantiles, method='inverted_cdf')) if len(v) else np.array([])
        if len(v):
            cuts = cuts[cuts < v.max()]
        answer.append(cuts)
    return tuple(answer)


def shared_refined_cuts(X, count):
    """Training-only dictionary: legacy anchors plus 5% grid and 1/2.5% tails."""
    q = np.unique(np.r_[np.arange(1, count + 1) / (count + 1),
                        np.arange(1, 20) / 20, .01, .025, .975, .99])
    answer = []
    for x in np.asarray(X, float).T:
        v = x[np.isfinite(x)]
        cuts = np.unique(np.quantile(v, q, method='inverted_cdf')) if len(v) else np.array([])
        answer.append(cuts[cuts < v.max()] if len(v) else cuts)
    return tuple(answer)


def node_shared_candidates(dictionary, values, count):
    """Snap node empirical quantiles to shared cuts; at most count candidates.

    Quantiles include record multiplicities (including bootstrap repetitions).
    Nearest ties select the lower cut. Duplicate snaps are not refilled.
    """
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.array([])
    cuts = dictionary[(dictionary >= values.min()) & (dictionary < values.max())]
    if len(cuts) <= count:
        return cuts
    targets = np.quantile(values, np.arange(1, count + 1)/(count + 1), method='inverted_cdf')
    return np.unique(cuts[np.argmin(abs(cuts[:, None] - targets[None, :]), axis=0)])


def training_cut_dictionary(X, config):
    if config.cut_strategy == 'shared_quantile':
        return shared_quantile_cuts(X, config.n_thresholds)
    if config.cut_strategy == 'shared_refined':
        return shared_refined_cuts(X, config.n_thresholds)
    return None


def tree_feature_subset(p, count, seed, tree_id):
    """Balanced shuffled blocks; independent of bootstrap and split RNG streams."""
    if count is None:
        return np.arange(p)
    blocks = (p + count - 1) // count
    permutation = np.random.default_rng(np.random.SeedSequence([seed, tree_id // blocks, 731])).permutation(p)
    start = (tree_id % blocks) * count
    return np.sort(permutation[np.arange(start, start + count) % p])


def split_region(region, feature, cut, missing_left):
    low, high, missing = region.get(feature, (-np.inf, np.inf, True))
    left, right = dict(region), dict(region)
    left[feature] = (low, min(high, cut), missing and missing_left)
    right[feature] = (max(low, cut), high, missing and not missing_left)
    return left, right


class PartitionLedger:
    def __init__(self, limit):
        self.limit = limit
        self.main = {}
        self.pairs = {}
        self.cells = 0
        self.accepted_splits = 0
        self.reused_splits = 0
        self.rejected_candidates = 0

    def project(self, children):
        main, pairs = {}, {}
        for region in children:
            key = tuple(sorted(region))
            if len(key) > 2:
                raise ValueError('partition budget requires paths of order at most two')
            bounds = {j: {float(v) for v in region[j][:2] if np.isfinite(v)} for j in key}
            for j in key:
                main.setdefault(j, set(self.main.get(j, set()))).update(bounds[j])
            if len(key) == 2:
                old = self.pairs.get(key, (set(), set()))
                pair = pairs.setdefault(key, (set(old[0]), set(old[1])))
                for axis, j in enumerate(key):
                    pair[axis].update(bounds[j])
        cost = self.cells
        for j, cuts in main.items():
            cost += len(cuts) + 2 - (len(self.main[j]) + 2 if j in self.main else 0)
        for key, cuts in pairs.items():
            cost += (len(cuts[0])+2)*(len(cuts[1])+2)
            if key in self.pairs:
                old = self.pairs[key]
                cost -= (len(old[0])+2)*(len(old[1])+2)
        return cost, main, pairs

    def accepts(self, proposal):
        return self.limit is None or proposal[0] <= self.limit

    def commit(self, proposal):
        if not self.accepts(proposal):
            raise ValueError('global cell budget exceeded')
        if proposal[0] < self.cells:
            raise AssertionError('nonmonotone partition ledger')
        self.reused_splits += proposal[0] == self.cells
        self.accepted_splits += 1
        self.cells, main, pairs = proposal
        self.main.update(main); self.pairs.update(pairs)

    def summary(self):
        return dict(limit=self.limit, spatial_cells=self.cells,
                    structural_mains=len(self.main), structural_pairs=len(self.pairs),
                    accepted_splits=self.accepted_splits, zero_cost_splits=self.reused_splits,
                    rejected_candidates=self.rejected_candidates,
                    definition='unmerged spatial cells, including missing states; not time/cause scalars')
