"""Reuse frozen evaluator; exact historical list rule; no fitting/calibration."""
import numpy as np

def additional_utility(d,flags):
 event=d.event.to_numpy();eligible=d.label_eligible.to_numpy(bool)
 unknown=flags&~(eligible|(event==2));hit=flags&(event==1)
 companies=d.loc[flags,'group'].nunique();hits=d.loc[hit,'group'].nunique()
 return dict(confirmed_hit_company_groups=int(hits),company_confirmed_hit_fraction=float(hits/companies) if companies else None,unknown_company_groups=int(d.loc[unknown,'group'].nunique()),followup_supported_non_target_records=int((flags&eligible&(event==0)).sum()))
