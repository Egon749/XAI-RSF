"""Fixed deletion set transported to two saved annual training folds."""
from pathlib import Path
import numpy as np
from experiments.post_st_drop9_v1.common import (
    SEEDS, TIMES, DROP_NAMES, REFERENCE_SHA, kept_columns, reduced,
    configuration, pairing, array_hash, arrays, sha, write, horizon_predictions)

NAME = 'post_st_drop9_transport_v1'
YEARS = [2023, 2025]
HISTORIES = {
    2023: Path('/home/oliver/MainDiskData/xai_rsf_leaf_shallow_20261028_ep_9hazj'),
    2025: Path('/home/oliver/MainDiskData/xai_rsf_2025_shallow_parent_20261029_pSED0uC5'),
}


def horizon_support(data, months):
    t, e, ok = data['analysis_time'], data['event'], data['label_eligible'].astype(bool)
    hit = ok & (e > 0) & (t <= months)
    known = ok & (hit | (t >= months))
    return dict(horizon_months=months, companies=len(t),
        termination=int((hit & (e == 1)).sum()), recovery=int((hit & (e == 2)).sum()),
        known_no_first_event=int((known & ~hit).sum()), unknown=int((~known).sum()),
        followup_reaches_horizon=int((ok & (t >= months)).sum()),
        supported=bool(known.all() or (ok & (t >= months)).any()))


def references(year):
    if year == 2023:
        from experiments.post_st_leaf_shallow_v1.analyze import artifacts
        return {int(s): p for y, a, s, p in artifacts(HISTORIES[year])
                if y == year and a == 'shallow_parent'}
    from experiments.post_st_2025_shallow_parent_v1.analyze import outputs
    return {int(s): p for model, s, p in outputs(HISTORIES[year]) if model == 'xai_q2'}
