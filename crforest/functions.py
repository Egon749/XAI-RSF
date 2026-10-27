"""Exact product-reference functional ANOVA of the leaf-region CIF expansion.

No model object or tree traversal is used by the exported evaluator. Tables are
stored in lossless factored rectangle form; their logical spatial cell count is
reported separately from storage size. No intervals are coalesced.
"""
from itertools import product
from pathlib import Path
import numpy as np


def region_indicator(x, low, high, missing):
    return ((x > low) & (x <= high)) | (np.isnan(x) & missing)


class CIFExpansion:
    @classmethod
    def from_forest(cls, forest, reference):
        reference = np.asarray(reference, float)
        if reference.ndim != 2 or reference.shape[1] != forest.n_features_ or not len(reference) or np.isinf(reference).any():
            raise ValueError("finite-or-missing nonempty reference matrix required")
        if forest.audit()['max_path_variables'] > 2:
            raise ValueError("exact order-two export requires every path to have <=2 original variables")
        obj = cls()
        obj.grid = forest.grid_.copy()
        obj.n_features = forest.n_features_
        regions, coefficients = [], []
        for tree in forest.trees_:
            def visit(index, region):
                node = tree[index]
                if 'cif' in node:
                    regions.append(region)
                    coefficients.append(np.asarray(node['cif']) / len(forest.trees_))
                    return
                j, cut, missing_left = node['feature'], node['threshold'], node['missing_left']
                low, high, missing = region.get(j, (-np.inf, np.inf, True))
                left, right = dict(region), dict(region)
                left[j] = (low, min(high, cut), missing and missing_left)
                right[j] = (max(low, cut), high, missing and not missing_left)
                visit(node['left'], left)
                visit(node['right'], right)
            visit(0, {})
        size = len(regions)
        obj.features = np.full((size, 2), -1, int)
        obj.low, obj.high = np.full((size, 2), -np.inf), np.full((size, 2), np.inf)
        obj.missing, obj.mass = np.ones((size, 2), bool), np.ones((size, 2))
        for i, region in enumerate(regions):
            for a, (j, (low, high, missing)) in enumerate(sorted(region.items())):
                obj.features[i, a] = j
                obj.low[i, a], obj.high[i, a], obj.missing[i, a] = low, high, missing
                obj.mass[i, a] = region_indicator(reference[:, j], low, high, missing).mean()
        obj.coefficients = np.array(coefficients)
        obj.mu = np.einsum('i,itk->tk', obj.mass.prod(1), obj.coefficients)
        obj.reference_n = len(reference)
        obj._index()
        return obj

    def _index(self):
        self.mains = sorted(set(self.features.ravel()) - {-1})
        self.pairs = sorted({tuple(f) for f in self.features if np.all(f >= 0)})

    def _coeff(self, times):
        times = self.grid if times is None else np.asarray(times, float)
        ix = np.searchsorted(self.grid, times, side='right') - 1
        c, mu = self.coefficients[:, np.maximum(ix, 0)].copy(), self.mu[np.maximum(ix, 0)].copy()
        c[:, ix < 0], mu[ix < 0] = 0., 0.
        return c, mu

    def _indicators(self, X, ids=None):
        ids = np.arange(len(self.features)) if ids is None else np.asarray(ids)
        ans = np.ones((len(X), len(ids), 2))
        for axis in range(2):
            f = self.features[ids, axis]
            valid = f >= 0
            ans[:, valid, axis] = region_indicator(X[:, f[valid]], self.low[ids[valid], axis],
                                                   self.high[ids[valid], axis], self.missing[ids[valid], axis])
        return ans

    def predict(self, X, times=None):
        """Compute mu + all main and purified pair components, factored exactly."""
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[1] != self.n_features or np.isinf(X).any():
            raise ValueError("invalid prediction matrix")
        c, mu = self._coeff(times)
        out = np.empty((len(X), *mu.shape))
        for start in range(0, len(X), 128):
            ix = slice(start, start + 128)
            delta = self._indicators(X[ix]) - self.mass
            weights = (delta[:, :, 0] * self.mass[:, 1] + delta[:, :, 1] * self.mass[:, 0]
                       + delta[:, :, 0] * delta[:, :, 1])
            out[ix] = mu + (weights @ c.reshape(len(c), -1)).reshape(-1, *mu.shape)
        return out

    def component(self, X, key, times=None):
        X, key = np.asarray(X, float), tuple(key)
        c, mu = self._coeff(times)
        if len(key) == 1:
            ids, axes = np.where(self.features == key[0])
            if len(ids) == 0:
                return np.zeros((len(X), *mu.shape))
            z = self._indicators(X, ids)
            w = ((z[np.arange(len(X))[:, None], np.arange(len(ids))[None, :], axes[None, :]]
                  - self.mass[ids, axes]) * self.mass[ids, 1 - axes])
        elif len(key) == 2:
            ids = np.flatnonzero(np.all(self.features == key, axis=1))
            if len(ids) == 0:
                return np.zeros((len(X), *mu.shape))
            w = np.prod(self._indicators(X, ids) - self.mass[ids], axis=2)
        else:
            raise ValueError("only main and pair components")
        return (w @ c[ids].reshape(len(ids), -1)).reshape(len(X), *mu.shape)

    def cuts(self, key, j):
        ids = np.flatnonzero(np.any(self.features == j, axis=1)) if len(key) == 1 else np.flatnonzero(np.all(self.features == key, axis=1))
        axis = np.argmax(self.features[ids] == j, axis=1)
        values = np.r_[self.low[ids, axis], self.high[ids, axis]]
        return np.unique(values[np.isfinite(values)])

    @staticmethod
    def representatives(cuts):
        return np.r_[cuts, np.nextafter(cuts[-1], np.inf) if len(cuts) else 0., np.nan]

    def complexity(self, reference, support_min=10):
        """Exact unmerged spatial grids, empirical JOINT support, product mass."""
        records = []
        for key in [(j,) for j in self.mains] + self.pairs:
            cuts = [self.cuts(key, j) for j in key]
            shape = tuple(len(c) + 2 for c in cuts)  # finite intervals + missing
            bins = []
            margins = []
            for j, cc in zip(key, cuts):
                b = np.searchsorted(cc, reference[:, j], side='left')
                b[np.isnan(reference[:, j])] = len(cc) + 1
                bins.append(b)
                margins.append(np.bincount(b, minlength=len(cc) + 2) / len(reference))
            flat = np.ravel_multi_index(tuple(bins), shape)
            support = np.bincount(flat, minlength=int(np.prod(shape))).reshape(shape)
            probability = margins[0] if len(key) == 1 else margins[0][:, None] * margins[1][None, :]
            # Detect identically-zero components on every EXACT spatial cell and
            # every native time. This is algebra on the stored factor coefficients.
            nonzero = False
            reps = [self.representatives(c) for c in cuts]
            buffer = []
            for values in product(*reps):
                buffer.append(values)
                if len(buffer) == 128:
                    xx = np.zeros((len(buffer), self.n_features)); xx[:, list(key)] = buffer
                    nonzero |= bool(np.any(abs(self.component(xx, key)) > 1e-14)); buffer = []
                    if nonzero:
                        break
            if buffer and not nonzero:
                xx = np.zeros((len(buffer), self.n_features)); xx[:, list(key)] = buffer
                nonzero = bool(np.any(abs(self.component(xx, key)) > 1e-14))
            records.append(dict(features=list(map(int, key)), order=len(key), cells=int(np.prod(shape)),
                                nonzero=nonzero, low_support_cells=int((support < support_min).sum()),
                                zero_support_cells=int((support == 0).sum()),
                                low_support_product_mass=float(probability[support < support_min].sum())))
        cells = sum(r['cells'] for r in records)
        return dict(main_functions=sum(r['nonzero'] and r['order'] == 1 for r in records),
                    interaction_functions=sum(r['nonzero'] and r['order'] == 2 for r in records),
                    spatial_table_cells=cells, scalar_table_cells=cells * len(self.grid) * self.coefficients.shape[-1],
                    factored_leaf_coefficients=int(self.coefficients.size),
                    low_support_cell_fraction=sum(r['low_support_cells'] for r in records) / max(cells, 1),
                    zero_support_cell_fraction=sum(r['zero_support_cells'] for r in records) / max(cells, 1),
                    support_threshold=support_min, functions=records)

    def save(self, path):
        with Path(path).open('xb') as f:
            np.savez_compressed(f, grid=self.grid, features=self.features, low=self.low, high=self.high,
                                missing=self.missing, mass=self.mass, coefficients=self.coefficients, mu=self.mu,
                                n_features=np.array(self.n_features), reference_n=np.array(self.reference_n))

    @classmethod
    def load(cls, path):
        obj = cls()
        with np.load(path, allow_pickle=False) as a:
            for name in a.files:
                setattr(obj, name, a[name])
        obj.n_features, obj.reference_n = int(obj.n_features), int(obj.reference_n)
        obj._index()
        return obj
