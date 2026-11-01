"""Read-only, historical, fixed-model permutation diagnostics. No fit API."""
from pathlib import Path
import json
import numpy as np
from experiments.post_st_prognosis_v1.common import arrays, sha, write, endpoint_weights

NAME = 'post_st_feature_permutation_v1'
SEEDS = [42, 101, 202]
TIMES = np.array([365., 366., 730., 731.])
ROOT = Path('/home/oliver/MainDiskData/xai_rsf_level_coverage_20261031_ckCB3u1l')
OLD = Path('/home/oliver/MainDiskData/xai_rsf_temporal_leaf_20261028__kvhg19v')
HISTORY = Path('/home/oliver/MainDiskData/xai_rsf_leaf_shallow_20261028_ep_9hazj')
REFERENCE_SHA = 'aa12ff6d41a515625566cf94f20faa4435d8f9c8e71fdeef8d8faba77fc57c65'


def prohibit_fit(*args, **kwargs):
    raise RuntimeError('This protocol prohibits forest fitting or refitting.')


def logical_closure(indices):
    # Audit opinion indicators come from one categorical source; move the entire
    # block with one donor, never create multiple mutually exclusive opinions.
    indices = set(indices)
    if indices & set(range(76, 83)):
        indices |= set(range(76, 83))
    return sorted(indices)


def groups_from_training(X, threshold=.95, minimum_overlap=30):
    import pandas as pd
    corr = pd.DataFrame(X).corr(method='spearman', min_periods=minimum_overlap).to_numpy()
    edges = [(j, k, float(corr[j, k]), int((np.isfinite(X[:, j]) & np.isfinite(X[:, k])).sum()))
             for j in range(X.shape[1]) for k in range(j+1, X.shape[1])
             if np.isfinite(corr[j, k]) and abs(corr[j, k]) >= threshold]
    parent = list(range(X.shape[1]))
    def find(j):
        while parent[j] != j:
            parent[j] = parent[parent[j]]; j = parent[j]
        return j
    for j, k, _, _ in edges:
        parent[find(k)] = find(j)
    components = {}
    for j in range(X.shape[1]): components.setdefault(find(j), []).append(j)
    return corr, edges, [g for g in components.values() if len(g) > 1]


def replacement_lookup(ex, X, indices, times=TIMES):
    """Exact prediction change [recipient, donor, time, cause], no traversal.

    Only affected leaf rectangles are recomputed. All fields in indices use the
    SAME donor. This is an algebraic acceleration, not a change to permutation.
    """
    c, _ = ex._coeff(times)
    moved = np.isin(ex.features, indices)
    ids = np.flatnonzero(moved.any(axis=1))
    n, width = len(X), c.shape[1]*c.shape[2]
    if not len(ids): return np.zeros((n, n, len(times), c.shape[-1]))
    z = ex._indicators(X, ids)
    co = c[ids].reshape(len(ids), width)
    out = np.zeros((n, n, width))
    for a, b in [(True, False), (False, True), (True, True)]:
        use = (moved[ids, 0] == a) & (moved[ids, 1] == b)
        if not use.any(): continue
        cc, zz = co[use], z[:, use]
        if a and b:
            out += (zz.prod(axis=2) @ cc)[None, :, :]
        else:
            changed, fixed = (0, 1) if a else (1, 0)
            out += np.einsum('il,jl,lk->ijk', zz[:, :, fixed], zz[:, :, changed], cc, optimize=True)
    out -= (z.prod(axis=2) @ co)[:, None, :]
    return out.reshape(n, n, len(times), c.shape[-1])


def score_batch(pred, y, w):
    """Weighted AUC/AP (correct ties), IPCW Brier; rows are predictions."""
    p = np.atleast_2d(pred); y = np.asarray(y, bool); w = np.asarray(w, float)
    active = w > 0
    pp, yy, ww = p[:, active], y[active], w[active]
    result = {'risk_mean': p.mean(axis=1), 'auc': np.full(len(p), np.nan),
              'ap': np.full(len(p), np.nan), 'brier': np.sum(w*(p-y)**2, axis=1)/len(y)}
    if not active.any():
        result['brier'][:] = np.nan; return result
    pos, neg = np.flatnonzero(yy), np.flatnonzero(~yy)
    if len(pos) and len(neg):
        d = pp[:, pos, None]-pp[:, None, neg]
        result['auc'] = np.sum(((d > 0)+.5*(d == 0))*ww[pos][None, :, None]*ww[neg][None, None, :],axis=(1,2))/(ww[pos].sum()*ww[neg].sum())
    if len(pos):
        order = np.argsort(-pp, axis=1, kind='stable')
        v = np.take_along_axis(pp, order, axis=1)
        weights = ww[order]; hits = weights*yy[order]
        tp, total = np.cumsum(hits, axis=1), np.cumsum(weights, axis=1)
        ends = np.concatenate([v[:, :-1] != v[:, 1:], np.ones((len(p), 1), bool)], axis=1)
        # AP sums precision at score-group ends times positive mass in each group.
        for i in range(len(p)):
            ix = np.flatnonzero(ends[i]); delta = np.diff(np.r_[0., tp[i, ix]])
            result['ap'][i] = np.sum(delta*tp[i, ix]/total[i, ix])/ww[pos].sum()
    return result


def bootstrap_effect(original, permuted, y, w, counts):
    """Average over ALL fixed permutations; paired company bootstrap.

    IPCW weights stay fixed. Main horizon-complete comparisons have w=1.
    No claim of uncertainty over training or feature selection.
    """
    p = np.atleast_2d(permuted); pos = np.flatnonzero(y & (w>0)); neg = np.flatnonzero(~y & (w>0))
    result = np.full((len(counts), 2), np.nan)
    weighted = counts*w
    delta_b = np.mean((p-y)**2, axis=0)-(original-y)**2
    result[:, 1] = weighted @ delta_b / len(y)
    if len(pos) and len(neg):
        d0 = original[pos, None]-original[None, neg]
        dp = p[:, pos, None]-p[:, None, neg]
        change = (d0>0)+.5*(d0==0)-np.mean((dp>0)+.5*(dp==0), axis=0)
        den = weighted[:, pos].sum(axis=1)*weighted[:, neg].sum(axis=1)
        num = np.einsum('bi,ij,bj->b', weighted[:, pos], change, weighted[:, neg], optimize=True)
        np.divide(num, den, out=result[:, 0], where=den>0)
    return result


def evaluation_context(labels, months, cause):
    t, e = labels['analysis_time'], labels['event']
    w, known, diag = endpoint_weights(t, e, np.full(len(t), months), labels['label_eligible'])
    y = (e == cause) & (t <= months)
    # No one remaining under observation at H => probability at H is unsupported
    # unless all endpoints are already known (everyone had an earlier event).
    supported = bool((t >= months).any() or known.all())
    return y, w, known, supported, diag


def metric_effect(original, permuted, y, w):
    base, sh = score_batch(original, y, w), score_batch(permuted, y, w)
    return {k: (float(np.mean(sh[k])-base[k][0]) if k == 'brier'
                else float(base[k][0]-np.mean(sh[k]))) for k in ['auc', 'ap', 'brier']}
