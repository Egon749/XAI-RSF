"""Evaluate existing H24 scans; record requests, never invent missing fits."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from .selection import *

class NeedFits(Exception):
    def __init__(self,items):self.items=items

class ExistingStore:
    def __init__(self,parent):
        self.parent=parent;sets=load(parent/'input/subsets.json');self.sid={tuple(s['keep']):sid for sid,s in sets.items() if s['mode']=='count'}
        self.registry={(r['year'],r['subset_id']):r for r in load(parent/'audit/registry_aggregate_inner.json')};self.cache={};self.requests=[]
    def ensure(self,keeps,years):
        missing=[dict(keep=list(k),year=y) for k in sorted(set(map(tuple,keeps))) for y in years if (y,self.sid.get(k)) not in self.registry]
        if missing:raise NeedFits(missing)
    def get(self,keep,year):
        key=(tuple(keep),year)
        if key in self.cache:return self.cache[key]
        record=self.registry[(year,self.sid[tuple(keep)])];p=Path(record['path']);s=load(p/'summary.json');st=pd.read_csv(p/'seed_stability.csv')
        value=dict(ensemble=arrays(p/'ensemble/predictions_PRIVATE.npz')['cif'],seeds=[arrays(Path(x)/'predictions_PRIVATE.npz')['cif'] for x in record['paths']],
            cells=s['check']['spatial_table_cells'],pair_jaccard=float(st[st.order==2].jaccard.mean()))
        self.cache[key]=value;return value

def run_initial(run):
    protocol=load(Path(__file__).with_name('frozen_protocol.json'));parent=Path(protocol['parent_private_run']);support=pd.read_csv(parent/'reports/historical_support.csv')
    store=ExistingStore(parent);allrows=[];outcomes={}
    for outer in [2023,2024,2025]:
        rows=support[support.outer_cutoff==outer].to_dict('records');gate=gate24(rows);units=initial_units(parent,outer)
        evaluator=Evaluator(parent,outer,store,rows)
        # Save every existing initial contrast independently of subsequent fits.
        initial=[]
        if gate['passed']:
            for u in units:
                keep=tuple(j for j in range(83) if j not in u);d=evaluator.contrast(tuple(range(83)),keep)
                d.update(outer=outer,removed=list(u),phase='initial',round=0);initial.append(d)
            allrows.extend(initial)
        try:
            groups=[g['members'] for g in load(parent/'input/group_specs.json') if g['outer']==outer]
            keep,path,scans=select(evaluator,units,gate,groups)
            outcomes[str(outer)]=dict(gate=gate,keep=list(keep),path=path,status='SELECTION_COMPLETE_USING_EXISTING_FITS',new_forests=0)
        except NeedFits as e:outcomes[str(outer)]=dict(gate=gate,status='NEW_MATCHED_FITS_REQUIRED',requests=e.items)
    frame=pd.DataFrame(allrows)
    for key in ['keep','removed','failed_guards']:frame[key]=frame[key].apply(json.dumps)
    frame.to_csv(run/'reports/INITIAL_H24_CONTRASTS.csv',index=False)
    with (run/'audit/initial_selection.json').open('x') as f:json.dump(outcomes,f,ensure_ascii=False,indent=2)
    print(json.dumps({y:{k:v for k,v in r.items() if k in ['gate','status','new_forests','keep']} for y,r in outcomes.items()}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();run_initial(a.run)
