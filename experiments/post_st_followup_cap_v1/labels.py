"""Training-only first A/X endpoint reconstruction with a configurable cap.

The legacy 24-month implementation is left unchanged. This module preserves
its event ordering and chain-gap rules while allowing 12, 36, or no month cap.
"""
import pandas as pd
from experiments.post_st_prognosis_v1.prepare import name_state


def rebuild(row, events, reviews, cutoff, months):
    if months not in (12,24,36,None):
        raise ValueError('Only frozen follow-up caps are allowed')
    cutoff=pd.Timestamp(cutoff)
    origin=pd.Timestamp(row['origin'])
    start=pd.Timestamp(row['entry_announcement'])
    available_end=cutoff-pd.Timedelta(days=1)
    limit=available_end if months is None else min(available_end,origin+pd.DateOffset(months=months))
    if limit<=origin:
        raise ValueError('No positive pre-cutoff follow-up')
    # Filter BEFORE examining endpoints or gaps: no future labels may influence
    # an earlier training snapshot, even if present in the authoritative file.
    ee=events[(events.announcement>=start)&(events.announcement<=available_end)]
    terminal=ee[ee.Chgtype.str.endswith(('A','X'))]
    if (terminal.announcement<=origin).any():
        raise ValueError('Endpoint precedes or coincides with shifted origin')
    terminal=terminal[terminal.announcement<=limit].sort_values('announcement')
    stop=limit;event=0
    reason='complete_horizon' if months is not None and limit==origin+pd.DateOffset(months=months) else 'administrative_censoring'
    if len(terminal):
        stop=terminal.announcement.iloc[0]
        kinds={1 if z.endswith('X') else 2 for z in terminal.loc[terminal.announcement.eq(stop),'Chgtype']}
        if len(kinds)!=1:
            raise ValueError('Simultaneous first recovery/termination unresolved')
        event=kinds.pop();reason='observed_termination' if event==1 else 'observed_full_recovery'
    gaps=ee[(ee.announcement<=stop)&(ee.chain_discontinuity=='1')]
    unresolved=False
    for z in gaps.itertuples():
        reused=reviews.get(z.event_id,{}).get('decision','').startswith('resolved')
        final=(z.Chgtype.endswith('X') and
            (name_state(z.ShortNamebc) in ['B','D','T'] or name_state(z.ShortNameac) in ['B','D','T']) and
            ('终止上市' in z.Content or z.all_cause_exit_confirmed=='1'))
        intermediate=z.Chgtype[-1] in ['B','D','C','S','T'] and z.Chgtype[0] in ['B','D','C','S','T']
        unresolved|=not(reused or final or intermediate)
    if unresolved:
        raise ValueError('Unresolved chain gap in extended training follow-up; no silent cohort deletion')
    return dict(time=(stop-origin).days,event=event,stop=stop,reason=reason,conflict=False,
        label_eligible=True,chain_gap_present=bool(len(gaps)),unresolved_chain_gap=False,
        effective_followup_cap=available_end)
