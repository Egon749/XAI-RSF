import numpy as np
from experiments.post_st_group_selection_v2.selection import FULL, select, interaction_guard

class Evaluator:
    def __init__(self, fail=False): self.fitted=[];self.fail=fail
    def prefetch(self, candidates): self.fitted.extend(candidates)
    def summary(self, keep): return {'cells':len(keep)*10}
    def contrast(self, full, keep):
        d={f'delta_{m}{h}':0. for h in [12,24] for m in ['auc','ap','capture','brier']}
        d.update({f'recovery_delta_brier{h}':0. for h in [12,24]})
        d.update(keep=list(keep),spatial_cell_change=-10.)
        if self.fail:d['delta_ap12']=np.nan
        return d

def test_supported_branch_actually_scores_and_selects_without_confirmation():
    keep=FULL[:-1]; ev=Evaluator()
    selected,status,scans,checks=select({'passed':True,'confirmation':False},[FULL,keep],ev,lambda k:[{'passed':True}])
    assert selected==keep and status=='EXPLORATORY_SELECTED'
    assert set(ev.fitted)=={FULL,keep} and len(scans)==1 and len(checks)==1

def test_event_support_fallback_does_not_fit_search_candidates():
    ev=Evaluator()
    selected,status,scans,_=select({'passed':False},[FULL,FULL[:-1]],ev,lambda k:[])
    assert selected==FULL and not ev.fitted and not scans and status=='INSUFFICIENT_DISCOVERY_SUPPORT'

def test_interaction_loss_rejects_despite_predictive_guard_pass():
    ev=Evaluator();keep=FULL[:-1]
    selected,_,scans,_=select({'passed':True},[FULL,keep],ev,lambda k:[interaction_guard(np.arange(8),np.zeros(8))])
    assert selected==FULL and 'INTERACTION_SHAPE_OR_AMPLITUDE' in scans[0]['failed_guards']
    assert set(ev.fitted)=={FULL,keep}

def test_undefined_secondary_endpoint_fails_closed_after_actual_search():
    ev=Evaluator(True)
    selected,_,scans,_=select({'passed':True},[FULL,FULL[:-1]],ev,lambda k:[{'passed':True}])
    assert selected==FULL and 'delta_ap12' in scans[0]['failed_guards'] and len(ev.fitted)==2

def test_search_all_candidates_and_deterministic_complexity_tie_break():
    ev=Evaluator();a=FULL[:-1];b=FULL[:-2]
    selected,_,scans,_=select({'passed':True},[a,FULL,b],ev,lambda k:[{'passed':True}])
    assert selected==b and len(scans)==2 and len(ev.fitted)==3
