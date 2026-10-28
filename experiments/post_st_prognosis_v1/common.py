from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False, default=str)


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def horizons(origin):
    origin = pd.Timestamp(origin)
    return [(origin + pd.DateOffset(months=m) - origin).days for m in (12, 24)]


def calendar_elapsed(origin, stop):
    """Elapsed calendar months, interpolated only within adjacent month boundaries.

    Every complete 24-month follow-up is exactly 24, including leap years.
    This is the evaluation censoring clock; forest fitting still uses actual days.
    """
    origin=pd.Timestamp(origin);stop=pd.Timestamp(stop)
    if stop<origin:raise ValueError('Follow-up precedes prediction origin')
    m=(stop.year-origin.year)*12+stop.month-origin.month
    while origin+pd.DateOffset(months=m)>stop:m-=1
    lo=origin+pd.DateOffset(months=m);hi=origin+pd.DateOffset(months=m+1)
    return m+(stop-lo).days/(hi-lo).days


def classify_first(origin, endpoints, observed_until, months=24):
    """Calendar horizon and verified coverage only. Event types: X=1, A=2.

    Endpoints contain announcement dates, never execution dates. Simultaneous
    distinct endpoints are unresolved rather than assigned an arbitrary order.
    """
    origin = pd.Timestamp(origin)
    cap = min(pd.Timestamp(observed_until), origin + pd.DateOffset(months=months))
    if any(int(k) not in (1, 2) or pd.isna(pd.Timestamp(d)) for d,k in endpoints):
        raise ValueError('Explicit A/X announcement endpoints required')
    if cap <= origin:
        return dict(time=0, event=0, stop=origin, reason='no_positive_verified_followup', conflict=True)
    if any(pd.Timestamp(d)==origin for d,k in endpoints):
        return dict(time=0,event=0,stop=origin,reason='same_day_origin_endpoint_order_unknown',conflict=True)
    events = sorted((pd.Timestamp(d), int(k)) for d, k in endpoints if origin < pd.Timestamp(d) <= cap)
    if events:
        first = events[0][0]
        if len({k for d, k in events if d == first}) != 1:
            return dict(time=(first-origin).days, event=0, stop=first,
                        reason='simultaneous_recovery_termination_unresolved', conflict=True)
        return dict(time=(first-origin).days, event=events[0][1], stop=first,
                    reason='observed_termination' if events[0][1] == 1 else 'observed_full_recovery', conflict=False)
    return dict(time=(cap-origin).days, event=0, stop=cap,
                reason='complete_horizon' if cap == origin + pd.DateOffset(months=months) else 'administrative_censoring', conflict=False)


def endpoint_weights(time, event, horizon, eligible=None):
    """Full follow-up reverse KM, correct treatment of events AFTER the horizon.

    No positivity floor or replacement of unknown outcomes with negatives.
    """
    time = np.asarray(time, float); event = np.asarray(event, int)
    h = np.broadcast_to(np.asarray(horizon, float), time.shape)
    eligible = np.ones(len(time), bool) if eligible is None else np.asarray(eligible, bool)
    use = eligible & (time > 0)
    if not use.any():
        return np.zeros(len(time)), np.zeros(len(time), bool), dict(min_G=None, known=0)
    tt, inv, count = np.unique(time[use], return_inverse=True, return_counts=True)
    risk = np.cumsum(count[::-1])[::-1]
    censor = np.bincount(inv, weights=event[use] == 0, minlength=len(tt))
    G = np.cumprod(1 - censor / risk)
    def left(t):
        ix = np.searchsorted(tt, t, side='left') - 1
        return np.where(ix >= 0, G[np.maximum(ix, 0)], 1.)
    at_event = (event > 0) & (time <= h)
    known = eligible & (at_event | (time >= h))
    den = np.where(at_event, left(time), left(h))
    if np.any(den[known] <= 0):
        raise ValueError('IPCW positivity violation for observed endpoint')
    w = np.zeros(len(time)); w[known] = 1 / den[known]
    return w, known, dict(min_G=float(den[known].min()) if known.any() else None,
                         known=int(known.sum()), unknown=int((~known).sum()))


def horizon_predictions(cif, times, horizon):
    times=np.asarray(times);h=np.asarray(horizon);ix=np.searchsorted(times,h)
    if np.any(ix>=len(times)) or not np.array_equal(times[ix],h):
        raise ValueError('Requested calendar-day horizon absent from predictions')
    return np.asarray(cif)[np.arange(len(h)),ix]
