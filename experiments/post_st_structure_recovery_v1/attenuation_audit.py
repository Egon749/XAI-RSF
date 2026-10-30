"""Arithmetic audit of the prespecified per-tree diagnostics, without refitting."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

def audit(run):
    path=run/'reports';data=pd.read_csv(path/'tree_signal_per_repetition.csv.gz');rows=[]
    for keys,g in data.groupby(['scenario','arm','variant','day','order','role']):
        total=int(g.tree_count.sum());eligible=int(g.eligible_trees.sum());nonzero=int(g.nonzero_trees.sum())
        ratio=float(np.average(g.amplitude_ratio,weights=g.tree_count))
        conditional=float((g.nonzero_amplitude_ratio.fillna(0)*g.nonzero_trees).sum()/nonzero) if nonzero else np.nan
        estimate=(eligible/total)*(nonzero/eligible)*conditional if nonzero else 0.
        err=abs(ratio-estimate);assert err<1e-12
        rows.append(dict(zip(['scenario','arm','variant','day','order','role'],keys),total_trees=total,eligible_trees=eligible,
            nonzero_trees=nonzero,eligible_fraction=eligible/total,mean_nonzero_trees=float(g.nonzero_trees.mean()),
            nonzero_in_eligible=nonzero/eligible,conditional_nonzero_ratio=conditional,amplitude_ratio=ratio,identity_error=err))
    result=pd.DataFrame(rows);result.to_csv(path/'interaction_attenuation_identity.csv',index=False)
    print('maximum arithmetic identity error',result.identity_error.max())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();audit(a.run)
