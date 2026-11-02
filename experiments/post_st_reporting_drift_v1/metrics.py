"""Within-year discrimination and fixed-company resampling.

Undefined years contribute zero pairs, not an AUC of zero. Equal-year averages
use only years with both classes and always report the number omitted.
"""
import numpy as np
from experiments.post_st_ablation_v1.uncertainty import fast_metrics


def paired_auc(y, p):
    """Independent literal positive-negative pair counting, ties count half."""
    y, p = np.asarray(y, bool), np.asarray(p, float)
    positive, negative = p[y], p[~y]
    pairs = len(positive) * len(negative)
    concordant = float(np.sum(positive[:, None] > negative[None, :])
                       + .5 * np.sum(positive[:, None] == negative[None, :]))
    return (concordant / pairs if pairs else np.nan), pairs, concordant


def stratified_metrics(y, p, year, ids):
    y, p, year, ids = (np.asarray(x) for x in (y, p, year, ids))
    annual, counts = [], []
    for value in np.unique(year):
        ix = year == value
        annual.append(fast_metrics(y[ix], p[ix], year[ix], ids[ix])[0])
        counts.append(int(y[ix].sum()) * int((~y[ix].astype(bool)).sum()))
    annual, counts = np.array(annual), np.array(counts)
    usable = counts > 0
    pooled = fast_metrics(y, p, year, ids)[0]
    within = float(np.sum(annual[usable]*counts[usable])/counts.sum()) if usable.any() else np.nan
    equal = float(np.mean(annual[usable])) if usable.any() else np.nan
    return dict(within_year_auc=within, equal_year_auc=equal, pooled_auc=float(pooled),
                within_minus_pooled=within-pooled, within_pairs=int(counts.sum()),
                informative_years=int(usable.sum()), zero_pair_years=int((~usable).sum()))


def paired_draws(sizes, repeats=1000, seed=2026110201):
    # Exact existing cause-support logic/order: one generator, chronological years.
    rng = np.random.default_rng(seed)
    return {year: rng.integers(n, size=(repeats, n)) for year, n in sizes.items()}


def interval(values):
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    lo, hi = np.quantile(values[valid], [.025, .975]) if valid.any() else (np.nan, np.nan)
    return dict(lower95=float(lo), upper95=float(hi), valid_repeats=int(valid.sum()),
                undefined_repeats=int((~valid).sum()), repeats=len(values))


def alert_metrics(y, p, years, ids):
    y, p, years, ids = (np.asarray(x) for x in (y, p, years, ids))
    selected = []
    for year in np.unique(years):
        ix = np.flatnonzero(years == year)
        selected.extend(ix[np.lexsort((ids[ix], -p[ix]))[:int(np.ceil(.2*len(ix)))]])
    nselected = len(selected)
    captured = int(y[selected].sum())
    observed, mean = float(y.mean()), float(p.mean())
    precision = captured / nselected
    return dict(observed=observed, mean_predicted=mean,
                observed_expected=observed/mean if mean > 0 else np.nan,
                checked=nselected, captured=captured, precision=precision,
                lift=precision/observed if observed > 0 else np.nan,
                coverage=captured/int(y.sum()) if y.sum() else np.nan)
