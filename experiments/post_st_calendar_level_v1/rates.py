"""Calendar risk-time statistics; no model fitting or evaluation inputs."""
import calendar
from datetime import datetime
import numpy as np
import pandas as pd

DAYS_PER_MONTH = 365.25 / 12


def spells(metadata, cutoff):
    m = metadata.copy()
    m['origin'] = pd.to_datetime(m.origin)
    m['stop'] = pd.to_datetime(m.stop)
    cutoff = pd.Timestamp(cutoff)
    m = m.loc[m.origin < cutoff].copy()
    assert not m.group.duplicated().any(), 'One frozen risk spell per company required'
    cap = m.origin.map(lambda d: d + pd.DateOffset(months=24))
    m['event'] = np.where((m.stop < cutoff) & (m.stop <= cap), m.event, 0)
    m['stop'] = pd.concat([m.stop, cap, pd.Series(cutoff, index=m.index)], axis=1).min(axis=1)
    assert (m.stop >= m.origin).all()
    return m.reset_index(drop=True)


def contributions(m, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    left = m.origin.clip(lower=start)
    right = m.stop.clip(upper=end)
    days = np.maximum((right-left).dt.total_seconds().to_numpy()/86400, 0.)
    events = np.column_stack([(m.event.eq(k) & m.stop.ge(start) & m.stop.lt(end)).to_numpy() for k in [1, 2]])
    return days, events


def independent_totals(metadata, cutoff, start, end):
    """Pure Python per-company arithmetic; does not call the vector implementation."""
    cutoff, start, end = [pd.Timestamp(t).to_pydatetime() for t in [cutoff, start, end]]
    days, counts = 0., [0, 0]
    for row in metadata.itertuples(index=False):
        origin, stop = [pd.Timestamp(t).to_pydatetime() for t in [row.origin, row.stop]]
        if origin >= cutoff:
            continue
        cap = origin.replace(year=origin.year+2, day=min(origin.day, calendar.monthrange(origin.year+2, origin.month)[1]))
        event = int(row.event) if stop < cutoff and stop <= cap else 0
        stop = min(stop, cap, cutoff)
        days += max((min(stop, end)-max(origin, start)).total_seconds()/86400, 0.)
        if start <= stop < end and event in [1, 2]:
            counts[event-1] += 1
    return days, np.array(counts)


def requested_slope(hdays, hevents, wdays, wevents, ix=None):
    ix = np.arange(len(hdays)) if ix is None else np.asarray(ix)
    hist_rate = hevents[ix].sum() / (hdays[ix].sum()/DAYS_PER_MONTH) if hdays[ix].sum() else np.nan
    window_rate = wevents[ix].sum() / (wdays[ix].sum()/DAYS_PER_MONTH) if wdays[ix].sum() else np.nan
    distinct = int(wevents[np.unique(ix)].sum())
    raw = window_rate / hist_rate if hist_rate > 0 else np.nan
    slope = 1. if distinct < 5 else float(np.clip(raw, .5, 4.))
    if not np.isfinite(slope):
        raise ValueError('Nonfinite supported slope')
    return dict(hist_rate=hist_rate, window_rate=window_rate, raw_ratio=raw,
                requested=slope, distinct_events=distinct, sparse_fallback=distinct < 5,
                bound_hit=bool(distinct >= 5 and (raw <= .5 or raw >= 4.)))


def certified_slopes(vertices, requested):
    b = np.array(requested, float)
    if np.max(vertices@b) <= 1+1e-12:
        return b, float(np.max(vertices@b))
    if np.max(vertices[:, 1]*b[1]) > 1+1e-12:
        raise ValueError('STOP: fixed recovery alone violates certificate')
    positive = vertices[:, 0] > 0
    ceiling = np.min((1-vertices[positive, 1]*b[1])/vertices[positive, 0])
    if ceiling < .5:
        raise ValueError('STOP: certificate has no termination slope in frozen bounds')
    b[0] = min(b[0], ceiling)
    assert np.max(vertices@b) <= 1+1e-12
    return b, float(np.max(vertices@b))
