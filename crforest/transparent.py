"""Transparent CIF deployment and honest convex weighting of frozen trees.

Prediction is evaluated by the COMPLETE native main/pair expansion, not by a
surrogate. Weights are a single nonnegative vector shared by EVERY input, cause,
and time. No gates, residual model, component-wise amplification or risk clipping.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.optimize import minimize
from .functions import CIFExpansion


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def validate_weights(weights, count):
    w = np.asarray(weights, float)
    if w.shape != (count,) or not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError('One finite nonnegative weight per tree required')
    if not np.isclose(w.sum(), 1., atol=1e-12, rtol=0):
        raise ValueError('Tree weights must sum to one; no implicit renormalization')
    return w.copy()


def group_roles(groups, fraction=.2, seed=20261028):
    """Outcome-independent, order-invariant split of DISTINCT company groups."""
    g = np.asarray(groups).astype(str)
    if g.ndim != 1 or np.any(g == '') or not 0 < fraction < 1:
        raise ValueError('Nonempty one-dimensional groups and fraction in (0,1) required')
    unique = sorted(set(g))
    if len(unique) < 2:
        raise ValueError('At least two distinct company groups required')
    ordered = sorted(unique, key=lambda x: hashlib.sha256(f'{seed}\0{x}'.encode()).hexdigest())
    count = min(len(unique) - 1, max(1, int(np.ceil(fraction * len(unique)))))
    holdout = set(ordered[:count])
    second = np.array([x in holdout for x in g])
    return np.flatnonzero(~second), np.flatnonzero(second)


def disjoint_groups(structure_groups, weight_groups):
    a, b = np.asarray(structure_groups).astype(str), np.asarray(weight_groups).astype(str)
    if a.ndim != 1 or b.ndim != 1 or not len(a) or not len(b):
        raise ValueError('Both nonempty role manifests required')
    if np.any(a == '') or np.any(b == '') or set(a) & set(b):
        raise ValueError('Structure and weight-learning company groups must be disjoint')
    return {'structure_groups': len(set(a)), 'weight_groups': len(set(b)),
            'group_overlap': 0}


def tree_predictions(forest, X, horizons):
    """Return [record, horizon, cause, tree], using actual day horizons per row.

    horizons is explicitly n x M; no inference of calendar dates from 30-day bins.
    Traversal is independent of the native compiler and preserves missing routes.
    """
    X, h = np.asarray(X, float), np.asarray(horizons, float)
    if X.ndim != 2 or X.shape[1] != forest.n_features_ or np.isinf(X).any():
        raise ValueError('Invalid predictors')
    if h.ndim != 2 or h.shape[0] != len(X) or not h.shape[1] or not np.isfinite(h).all() or np.any(h <= 0):
        raise ValueError('Positive finite n-by-M day horizons required')
    if np.any(np.diff(h, axis=1) <= 0):
        raise ValueError('Each record needs strictly increasing horizons')
    ix = np.searchsorted(forest.grid_, h, side='right') - 1
    out = np.zeros((len(X), h.shape[1], forest.config.n_causes, len(forest.trees_)))
    for b, nodes in enumerate(forest.trees_):
        stack = [(0, np.arange(len(X)))]
        while stack:
            node_id, rows = stack.pop()
            if not len(rows):
                continue
            node = nodes[node_id]
            if 'cif' in node:
                values = np.asarray(node['cif'], float)[np.maximum(ix[rows], 0)].copy()
                values[ix[rows] < 0] = 0.
                out[rows, :, :, b] = values
            else:
                x = X[rows, node['feature']]
                left = (x <= node['threshold']) | (np.isnan(x) & node['missing_left'])
                stack.extend([(node['left'], rows[left]), (node['right'], rows[~left])])
    if not np.isfinite(out).all() or out.min(initial=0) < -1e-12 or out.sum(2).max(initial=0) > 1 + 1e-12:
        raise ValueError('Invalid leaf probabilities')
    if np.any(np.diff(out, axis=1) < -1e-12):
        raise ValueError('Nonmonotone leaf CIF')
    return out


def _survival_inputs(time, event, n_causes):
    t, e = np.asarray(time, float), np.asarray(event)
    if (t.ndim != 1 or e.shape != t.shape or not len(t) or
            not np.isfinite(t).all() or np.any(t <= 0) or
            not np.isfinite(e).all() or np.any(e != np.floor(e)) or
            np.any((e < 0) | (e > n_causes))):
        raise ValueError('Positive followup and integer first-event causes required')
    return t, e.astype(int)


def state_targets(time, event, horizons, eligible, censor_time, censor_event,
                  n_causes=2, min_censor_survival=1e-4):
    """Multicategory endpoint targets and marginal reverse-KM IPCW.

    Censor nuisance uses STRUCTURE-training data only. Unknown endpoints remain
    weight zero. A failure AFTER a horizon is event-free AT that horizon.
    eligible is an explicit source-supported bool n x M contract, not inferred
    from a full duration for source-unknown rows. No clipping of unsupported G.
    Independence/positivity remain assumptions, not certified by this function.
    """
    t, e = _survival_inputs(time, event, n_causes)
    ct, ce = _survival_inputs(censor_time, censor_event, n_causes)
    h, ok = np.asarray(horizons, float), np.asarray(eligible)
    if h.ndim != 2 or h.shape[0] != len(t) or ok.shape != h.shape or ok.dtype != bool:
        raise ValueError('Explicit aligned boolean endpoint eligibility required')
    if not np.isfinite(h).all() or np.any(h <= 0) or np.any(np.diff(h, axis=1) <= 0):
        raise ValueError('Invalid horizons')
    if not 0 < min_censor_survival < 1:
        raise ValueError('Invalid positivity threshold')
    tt, inv, count = np.unique(ct, return_inverse=True, return_counts=True)
    risk = np.cumsum(count[::-1])[::-1]
    censor = np.bincount(inv, weights=ce == 0, minlength=len(tt))
    G = np.cumprod(1 - censor / risk)
    before_event = (e[:, None] > 0) & (t[:, None] <= h)
    known = ok & (before_event | (t[:, None] >= h))
    endpoint = np.where(before_event, t[:, None], h)
    ix = np.searchsorted(tt, endpoint, side='left') - 1
    denominator = np.where(ix >= 0, G[np.maximum(ix, 0)], 1.)
    if np.any(known & (denominator < min_censor_survival)):
        raise ValueError('Censoring positivity unsupported; do not floor weights silently')
    state = np.where(before_event, e[:, None], 0)
    y = np.eye(n_causes + 1)[state]
    ipc = np.divide(known.astype(float), denominator, out=np.zeros_like(h), where=known)
    return y, ipc, {'known_by_horizon': known.sum(0).tolist(),
                    'unknown_by_horizon': (~known).sum(0).tolist(),
                    'cause_events_in_weight_rows': [int(np.sum(e == k)) for k in range(1, n_causes + 1)],
                    'minimum_supported_G': float(denominator[known].min()) if known.any() else None,
                    'nuisance': 'structure-only marginal reverse-KM, left limits',
                    'assumptions': 'independent censoring and source eligibility; not guaranteed'}


@dataclass(frozen=True)
class WeightConfig:
    ridge_to_uniform: float = .001
    max_iterations: int = 3000
    gap_tolerance: float = 1e-8

    def __post_init__(self):
        if (not np.isfinite(self.ridge_to_uniform) or self.ridge_to_uniform < 0 or
            not isinstance(self.max_iterations, int) or self.max_iterations < 1 or
            not np.isfinite(self.gap_tolerance) or self.gap_tolerance <= 0):
            raise ValueError('Invalid frozen weight-learning configuration')


def solve_simplex_brier(predictions, targets, endpoint_weights, config=WeightConfig(),
                        record_weights=None):
    """Convex quadratic program over one shared simplex; analytic gradient.

    Targets have K+1 states, including event-free. Loss denominator is total
    base record weight x horizons, not known-only rows. FW gap checks actual
    numerical optimality independently of the optimizer success flag.
    """
    P, Y, W = map(lambda a: np.asarray(a, float), (predictions, targets, endpoint_weights))
    if P.ndim != 4 or not all(P.shape) or Y.shape != P.shape[:2] + (P.shape[2] + 1,) or W.shape != P.shape[:2]:
        raise ValueError('Invalid tree prediction / target dimensions')
    if not all(np.isfinite(a).all() for a in (P, Y, W)) or np.any(W < 0):
        raise ValueError('Nonfinite input or negative endpoint weights')
    if (P.min() < -1e-12 or P.sum(2).max() > 1 + 1e-12 or
            Y.min() < 0 or not np.allclose(Y.sum(2), 1., atol=1e-12, rtol=0)):
        raise ValueError('Illegal probability or state target')
    n, m, K, B = P.shape
    rw = np.ones(n) if record_weights is None else np.asarray(record_weights, float)
    if rw.shape != (n,) or not np.isfinite(rw).all() or np.any(rw <= 0) or not np.any(W > 0):
        raise ValueError('Positive base weights and some known endpoints required')
    W = W * (rw / rw.sum())[:, None] / m
    uniform = np.full(B, 1 / B)
    H = np.eye(B) * (2 * config.ridge_to_uniform)
    g = -2 * config.ridge_to_uniform * uniform
    constant = config.ridge_to_uniform * float(uniform @ uniform)
    for h in range(m):
        for k in range(K + 1):
            A = 1 - P[:, h].sum(1) if k == 0 else P[:, h, k - 1, :]
            weight, target = W[:, h], Y[:, h, k]
            H += 2 * A.T @ (weight[:, None] * A)
            g -= 2 * A.T @ (weight * target)
            constant += float(weight @ (target ** 2))
    def objective(w):
        return float(.5 * w @ H @ w + g @ w + constant)
    def gradient(w):
        return H @ w + g
    result = minimize(objective, uniform, jac=gradient, method='SLSQP',
                      bounds=[(0., 1.)] * B,
                      constraints=[{'type': 'eq', 'fun': lambda w: float(w.sum() - 1),
                                    'jac': lambda w: np.ones(B)}],
                      options={'maxiter': config.max_iterations, 'ftol': 1e-13})
    w = np.asarray(result.x)
    if not np.isfinite(w).all() or w.min() < -1e-10 or abs(w.sum() - 1) > 1e-9:
        raise RuntimeError('INFEASIBLE_WEIGHT_SOLUTION')
    # Explicit numerical simplex repair only; no pruning or probability clipping.
    repaired = np.maximum(w, 0.)
    repaired /= repaired.sum()
    adjustment = float(abs(repaired - w).max())
    w = validate_weights(repaired, B)
    grad = gradient(w)
    fw_gap = max(0., float(grad @ w - grad.min()))
    # Deterministic Frank-Wolfe cleanup; each step stays on the simplex.
    cleanup = 0
    while fw_gap > config.gap_tolerance and cleanup < config.max_iterations:
        s = np.zeros(B); s[int(np.argmin(grad))] = 1.
        direction = s - w
        curvature = float(direction @ H @ direction)
        step = min(1., fw_gap / curvature) if curvature > 0 else 1.
        w += step * direction
        grad = gradient(w)
        fw_gap = max(0., float(grad @ w - grad.min()))
        cleanup += 1
    w = validate_weights(w, B)
    receipt = {'status': 'CONVERGED' if fw_gap <= config.gap_tolerance else 'UNRESOLVED',
               'optimizer_message': str(result.message), 'iterations': int(result.nit),
               'cleanup_iterations': cleanup, 'simplex_roundoff_adjustment': adjustment,
               'frank_wolfe_gap': fw_gap, 'objective': objective(w),
               'uniform_objective': objective(uniform),
               'effective_tree_count': float(1 / (w @ w)), 'positive_weights': int(np.sum(w > 0)),
               'largest_weight': float(w.max()), 'config': asdict(config),
               'certificate_scope': 'floating point convex QP gap; not interval-arithmetic proof',
               'loss': 'IPCW sum of squared K+1-state errors, base-record and horizon average'}
    if receipt['status'] != 'CONVERGED' or objective(w) > objective(uniform) + 1e-10:
        raise RuntimeError('WEIGHT_OPTIMIZATION_UNRESOLVED: ' + json.dumps(receipt))
    return w, receipt


def learn_tree_weights(forest, X, time, event, horizons, eligible,
                       structure_groups, weight_groups, censor_time, censor_event,
                       config=WeightConfig(), record_weights=None):
    isolation = disjoint_groups(structure_groups, weight_groups)
    if len(weight_groups) != len(X) or len(censor_time) != len(structure_groups):
        raise ValueError('Role manifest / data length mismatch')
    P = tree_predictions(forest, X, horizons)
    Y, W, audit = state_targets(time, event, horizons, eligible, censor_time, censor_event,
                                forest.config.n_causes)
    w, receipt = solve_simplex_brier(P, Y, W, config, record_weights)
    receipt.update(isolation=isolation, endpoint_audit=audit,
                   all_inputs_same_weights=True, all_causes_same_weights=True,
                   all_times_same_weights=True)
    return w, receipt


class TransparentCIF:
    """The deployed model IS its full exact CIFExpansion; no forest is needed.

    The original forest can be retained privately for audit. This object cannot
    call hidden tree traversal, residuals, gates or fitted calibration at inference.
    """
    schema = 1

    @classmethod
    def from_forest(cls, forest, reference, weights=None, feature_names=None,
                    cause_names=None, weight_receipt=None):
        count = len(forest.trees_)
        w = validate_weights(np.full(count, 1 / count) if weights is None else weights, count)
        ex = CIFExpansion.from_forest(forest, reference)
        leaf_counts = [sum('cif' in n for n in tree) for tree in forest.trees_]
        if sum(leaf_counts) != len(ex.coefficients):
            raise AssertionError('Compiler / tree leaf identity mismatch')
        ex.coefficients *= np.repeat(w * count, leaf_counts)[:, None, None]
        ex.mu = np.einsum('i,itk->tk', ex.mass.prod(1), ex.coefficients)
        obj = cls(); obj.expansion = ex; obj.weights = w
        obj.feature_names = list(feature_names or [f'x{j}' for j in range(ex.n_features)])
        obj.cause_names = list(cause_names or [f'cause_{k+1}' for k in range(forest.config.n_causes)])
        if len(obj.feature_names) != ex.n_features or len(set(obj.feature_names)) != len(obj.feature_names):
            raise ValueError('Unique aligned feature names required')
        if len(obj.cause_names) != forest.config.n_causes or len(set(obj.cause_names)) != len(obj.cause_names):
            raise ValueError('Unique aligned cause names required')
        obj.audit = {'forest_audit': forest.audit(), 'leaf_counts': leaf_counts,
                     'weight_learning': weight_receipt or {'status': 'FIXED_EQUAL_WEIGHTS'},
                     'native_scale': 'CIF_probability', 'link': 'identity',
                     'reference': 'fixed structure-training product marginals',
                     'hidden_predictor': False, 'maximum_function_order': 2,
                     'interpretation': 'model arithmetic, not causal attribution'}
        return obj

    def _check_X(self, X):
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[1] != self.expansion.n_features or np.isinf(X).any():
            raise ValueError('Invalid declared feature matrix')
        return X

    def predict(self, X, times=None):
        X = self._check_X(X)
        if times is not None and (np.asarray(times).ndim != 1 or not np.isfinite(times).all()):
            raise ValueError('Finite time vector required')
        return self.expansion.predict(X, times)

    def predict_horizons(self, X, horizons):
        X = self._check_X(X); h = np.asarray(horizons, float)
        if h.ndim != 2 or h.shape[0] != len(X) or not np.isfinite(h).all() or np.any(h <= 0):
            raise ValueError('Positive n-by-M day horizons required')
        if np.any(np.diff(h, axis=1) <= 0):
            raise ValueError('Horizons must increase within each record')
        out = np.empty((*h.shape, len(self.cause_names)))
        unique, inverse = np.unique(h, axis=0, return_inverse=True)
        for index, times in enumerate(unique):
            rows = np.flatnonzero(inverse == index)
            out[rows] = self.predict(X[rows], times)
        return out

    def explain(self, X, times):
        X = self._check_X(X)
        keys = [(j,) for j in self.expansion.mains] + list(self.expansion.pairs)
        _, baseline = self.expansion._coeff(times)
        reconstruction = np.broadcast_to(baseline, (len(X), *baseline.shape)).copy()
        components = []
        for key in keys:
            value = self.expansion.component(X, key, times)
            reconstruction += value
            components.append({'indices': list(map(int, key)),
                               'features': [self.feature_names[j] for j in key],
                               'order': len(key), 'contribution': value})
        pred = self.predict(X, times)
        return {'baseline': baseline, 'components': components, 'prediction': pred,
                'survival': 1 - pred.sum(2), 'reconstructed': reconstruction,
                'max_error': float(np.max(abs(pred - reconstruction), initial=0)),
                'scale': 'CIF_probability; signed component values are NOT stand-alone probabilities',
                'complete': True, 'causal': False}

    def complete_tables(self, times, reference):
        """Every main/pair cell, all causes and horizons, explicit missing states.

        The empirical joint support is reported separately from the product
        reference used for centering. Table size is not a human-usability score.
        """
        from itertools import product
        reference = self._check_X(reference)
        output = []
        for key in [(j,) for j in self.expansion.mains] + list(self.expansion.pairs):
            cuts = [self.expansion.cuts(key, j) for j in key]
            reps = [self.expansion.representatives(c) for c in cuts]
            shape = tuple(len(r) for r in reps)
            ids = list(product(*[range(n) for n in shape]))
            X = np.zeros((len(ids), self.expansion.n_features))
            for axis, j in enumerate(key):
                X[:, j] = [reps[axis][idx[axis]] for idx in ids]
            values = self.expansion.component(X, key, times)
            bins = []
            for j, cc in zip(key, cuts):
                bb = np.searchsorted(cc, reference[:, j], side='left')
                bb[np.isnan(reference[:, j])] = len(cc) + 1
                bins.append(bb)
            support = np.bincount(np.ravel_multi_index(tuple(bins), shape),
                                  minlength=len(ids)).reshape(shape)
            axes = []
            for cc in cuts:
                axes.append([{'missing': False,
                              'lower_open': None if z == 0 else float(cc[z - 1]),
                              'upper_closed': None if z == len(cc) else float(cc[z])}
                             for z in range(len(cc) + 1)] + [{'missing': True}])
            output.append({'features': [self.feature_names[j] for j in key],
                           'indices': list(map(int, key)), 'order': len(key),
                           'axes': axes, 'shape': list(shape),
                           'values': values.reshape(*shape, len(times), len(self.cause_names)).tolist(),
                           'reference_joint_record_support': support.tolist(),
                           'spatial_cells': len(ids)})
        return {'times_days': list(map(float, times)), 'causes': self.cause_names,
                'baseline': self.expansion._coeff(times)[1].tolist(),
                'functions': output, 'complete': True, 'scale': 'CIF_probability',
                'reference_records': len(reference),
                'spatial_cells': sum(x['spatial_cells'] for x in output),
                'note': 'All cells, including missing and zero support; no TopK truncation.'}

    def save(self, directory):
        root = Path(directory); root.mkdir(parents=True, exist_ok=False)
        self.expansion.save(root / 'functions_PRIVATE.npz')
        write_json(root / 'manifest_PRIVATE.json', {
            'schema': self.schema, 'feature_names': self.feature_names,
            'cause_names': self.cause_names, 'weights': self.weights.tolist(), 'audit': self.audit,
            'functions_sha256': file_sha256(root / 'functions_PRIVATE.npz')})
        write_json(root / 'integrity.json', {'manifest_sha256': file_sha256(root / 'manifest_PRIVATE.json')})

    @classmethod
    def load(cls, directory):
        root = Path(directory)
        integrity = json.loads((root / 'integrity.json').read_text())
        if file_sha256(root / 'manifest_PRIVATE.json') != integrity['manifest_sha256']:
            raise ValueError('Manifest hash mismatch')
        meta = json.loads((root / 'manifest_PRIVATE.json').read_text())
        if meta['schema'] != cls.schema or file_sha256(root / 'functions_PRIVATE.npz') != meta['functions_sha256']:
            raise ValueError('Function schema/hash mismatch')
        obj = cls(); obj.expansion = CIFExpansion.load(root / 'functions_PRIVATE.npz')
        obj.weights = validate_weights(meta['weights'], len(meta['weights']))
        obj.feature_names, obj.cause_names, obj.audit = meta['feature_names'], meta['cause_names'], meta['audit']
        if obj.expansion.n_features != len(obj.feature_names) or obj.expansion.coefficients.shape[2] != len(obj.cause_names):
            raise ValueError('Invalid model dimensions')
        return obj
