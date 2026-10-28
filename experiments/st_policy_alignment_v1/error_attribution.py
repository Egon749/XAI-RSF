"""Descriptive overlap only; no causal attribution or within-slice reranking."""
import numpy as np
import pandas as pd

def overlap(d,flags):
 targets=set(d.loc[d.event.eq(1),'event_id']);out=[]
 for fraction in [.2,.25]:
  a,b,c=[flags[m,fraction] for m in ['q2','unrestricted','M2_allsector']]
  caught=[set(d.loc[f&d.event.eq(1).to_numpy(),'event_id']) for f in [a,b,c]]
  for name,ids in [('all_models_missed',targets-set.union(*caught)),('q2_and_unrestricted_missed',targets-(caught[0]|caught[1])),('only_q2_missed_vs_unrestricted',caught[1]-caught[0]),('only_unrestricted_missed_vs_q2',caught[0]-caught[1]),('q2_and_M2_missed',targets-(caught[0]|caught[2]))]:out.append(dict(top_fraction=fraction,error=name,level='unique_target_event',count=len(ids)))
  for name,f in [('q2_and_unrestricted',a&b),('q2_and_M2',a&c),('all_three',a&b&c)]:
   for outcome,mask in [('known_target_free',d.event.eq(0).to_numpy()&d.label_eligible.to_numpy(bool)),('competing',d.event.eq(2).to_numpy()),('unknown',d.event.eq(0).to_numpy()&~d.label_eligible.to_numpy(bool))]:
    out.append(dict(top_fraction=fraction,error=name+'_'+outcome,level='quarter_record',count=int((f&mask).sum()),company_groups=int(d.loc[f&mask,'group'].nunique())))
 return pd.DataFrame(out)
