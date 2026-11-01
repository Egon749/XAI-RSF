"""Constraint-based backward elimination with a mandatory historical support gate.

The evaluate callback must fit/evaluate all requested subsets on internal folds;
reusing the first-round ranking for later rounds is deliberately unsupported.
"""
from .common import pick_candidate,subset_hash

def eliminate(initial,pool,gate,evaluate,max_rounds=12):
    current=tuple(sorted(initial));path=[]
    if not gate['passed']:
        return current,[dict(round=0,features=len(current),action='STOP',reason=gate['reason'],subset_sha256=subset_hash(current),new_candidate_fits=0)]
    for step in range(max_rounds):
        proposals={tuple(j for j in current if j not in group) for group in pool if set(group)<=set(current)}
        proposals.discard(current);proposals.discard(())
        if not proposals:
            path.append(dict(round=step,features=len(current),action='STOP',reason='EMPTY_POOL'));break
        # Every remaining deletion is rescored against the NEW current set.
        scored=evaluate(current,sorted(proposals))
        assert {tuple(x['keep']) for x in scored}==proposals
        winner=pick_candidate(scored)
        if winner is None:
            path.append(dict(round=step,features=len(current),action='STOP',reason='NO_CANDIDATE_PASSES_FROZEN_CONSTRAINTS'));break
        previous=current;current=tuple(winner['keep'])
        path.append(dict(round=step,features=len(current),action='DELETE',removed=sorted(set(previous)-set(current)),subset_sha256=subset_hash(current),evidence=winner))
    return current,path
