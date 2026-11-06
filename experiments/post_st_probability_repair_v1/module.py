"""Age-adjusted historical levels and exact, globally valid affine CIF maps."""
from copy import deepcopy
import numpy as np
import pandas as pd

from experiments.post_st_calendar_level_v1.rates import spells


def age_exposures(metadata, cutoff):
    """Historical and recent risk exposure in two calendar-anniversary age bins."""
    m = spells(metadata, cutoff)
    end = pd.Timestamp(cutoff)
    start = end - pd.DateOffset(months=24)
    anniversary = m.origin.map(lambda d: d + pd.DateOffset(months=12))
    middle = pd.concat([anniversary, m.stop], axis=1).min(axis=1)
    hist_days = np.column_stack([
        (middle - m.origin).dt.total_seconds().to_numpy()/86400,
        np.maximum((m.stop-anniversary).dt.total_seconds().to_numpy()/86400, 0.)])
    recent_days = np.zeros_like(hist_days)
    for j, (left, right) in enumerate([(m.origin, middle), (anniversary, m.stop)]):
        recent_days[:, j] = np.maximum((right.clip(upper=end) - left.clip(lower=start)).dt.total_seconds().to_numpy()/86400, 0.)
    hist_events = np.zeros((len(m), 2, 2), dtype=int)
    for cause in (1, 2):
        observed = m.event.eq(cause) & m.stop.lt(end)
        hist_events[:, 0, cause-1] = observed & m.stop.le(anniversary)
        hist_events[:, 1, cause-1] = observed & m.stop.gt(anniversary)
    recent_events = hist_events * m.stop.ge(start).to_numpy()[:, None, None]
    return hist_days, hist_events, recent_days, recent_events


def piecewise_cif(rates, times):
    """Two-cause exponential segments, changing at nominal day 365."""
    rates = np.asarray(rates, float)
    times = np.asarray(times, float)
    if rates.shape != (2, 2) or not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError('two nonnegative cause rates in each of two bins required')
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError('finite nonnegative times required')
    cif = np.zeros((len(times), 2))
    survival = np.ones(len(times))
    durations = [np.minimum(times, 365.), np.maximum(np.minimum(times, 730.)-365., 0.)]
    for row, duration in zip(rates, durations):
        total = row.sum()
        if total > 0:
            probability = -np.expm1(-total * duration)
            cif += survival[:, None] * probability[:, None] * (row / total)
            survival *= np.exp(-total * duration)
    return cif


def age_model(metadata, cutoff):
    """Frozen five-total-pseudoevent regularization toward own age-bin history."""
    hd, he, rd, re = age_exposures(metadata, cutoff)
    historical = np.zeros((2, 2))
    recent = np.zeros((2, 2))
    diagnostics = []
    for j in range(2):
        exposure, window = float(hd[:, j].sum()), float(rd[:, j].sum())
        counts, recent_counts = he[:, j].sum(axis=0), re[:, j].sum(axis=0)
        if exposure > 0:
            historical[j] = counts / exposure
        total_rate = historical[j].sum()
        prior = 5. / total_rate if total_rate > 0 else 0.
        if total_rate > 0:
            recent[j] = (recent_counts + historical[j] * prior) / (window + prior)
        else:
            if recent_counts.sum() != 0:
                raise ValueError('recent events cannot exist in event-free history')
        diagnostics.append(dict(age_bin=j, historical_days=exposure, recent_days=window,
                                historical_events=counts.tolist(), recent_events=recent_counts.tolist(),
                                prior_days=prior, historical_rates=historical[j].tolist(),
                                recent_rates=recent[j].tolist(), no_historical_exposure=exposure==0))
    grid = np.r_[0., np.ceil(365. * np.arange(1, 25) / 12.)]
    hist = piecewise_cif(historical, grid)
    new = piecewise_cif(recent, grid)
    denominator = hist[12]
    numerator = new[12]
    support = re[:, 0].sum(axis=0)
    slopes = np.ones(2)
    fallbacks = (support < 5) | (denominator <= 0)
    raw = np.divide(numerator, denominator, out=np.ones(2), where=denominator > 0)
    slopes[~fallbacks] = np.clip(raw[~fallbacks], .5, 4.)
    return dict(grid=grid.tolist(), recent=new.tolist(), historical=hist.tolist(),
                requested_slopes=slopes.tolist(), raw_slopes=raw.tolist(),
                early_recent_event_support=support.tolist(), sparse_fallback=fallbacks.tolist(),
                diagnostics=diagnostics)


def offset_at(times, grid, values):
    grid, values, times = np.asarray(grid, float), np.asarray(values, float), np.asarray(times, float)
    if grid.ndim != 1 or len(grid) == 0 or values.shape != (len(grid), 2):
        raise ValueError('compatible nonempty two-cause offset grid required')
    if not np.isfinite(grid).all() or not np.isfinite(values).all() or (np.diff(grid) <= 0).any():
        raise ValueError('finite strictly increasing offset grid required')
    ix = np.searchsorted(grid, times, side='right') - 1
    result = values[np.maximum(ix, 0)].copy()
    result[ix < 0] = 0.
    return result


def mapping(pred, times, cal):
    return np.asarray(pred) * np.asarray(cal['b']) + offset_at(times, cal['shape_grid'], cal['offset_values'])


def transformed(original, cal):
    obj = deepcopy(original)
    grid = np.unique(np.r_[original.grid, cal['shape_grid']])
    coefficients, mu = original._coeff(grid)
    obj.grid = grid
    obj.coefficients = coefficients * np.asarray(cal['b'])
    obj.mu = mu * np.asarray(cal['b']) + offset_at(grid, cal['shape_grid'], cal['offset_values'])
    return obj


def certify(vertices, cal):
    """Sufficient all-input/all-time certificate; never clips mapped predictions."""
    b = np.asarray(cal['b'], float)
    offset = np.asarray(cal['offset_values'], float)
    grid = np.asarray(cal['shape_grid'], float)
    if b.shape != (2,) or (b <= 0).any() or not np.isfinite(b).all():
        raise ValueError('positive finite cause slopes required')
    offset_at(grid, grid, offset)
    if (offset < -1e-12).any() or (np.diff(offset, axis=0) < -1e-12).any():
        raise ValueError('nonnegative nondecreasing baseline required')
    vertices = np.asarray(vertices, float)
    bound = float(np.max(vertices @ b) + offset[-1].sum())
    if bound > 1 + 1e-10:
        raise ValueError('all-input terminal probability certificate failed')
    return dict(terminal_sum_bound=bound, minimum_offset=float(offset.min()),
                minimum_slope=float(b.min()), certificate='terminal envelope plus monotone baseline')
