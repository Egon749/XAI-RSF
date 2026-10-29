"""Reuse the already tested forest and shrinkage workers; no new model algorithm."""
from pathlib import Path
import argparse,os
from experiments.post_st_leaf_shallow_v1.worker import run as q2_run
from experiments.post_st_prognosis_v1.worker import run as rsf_run


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['xai_q2','standard_rsf'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--pilot',action='store_true');a=p.parse_args()
    a.year=2025;a.depth=4
    if a.model=='xai_q2':q2_run(a)
    else:
        os.environ['DATA_DIR']=str(Path(os.environ['DATA_DIR'])/'folds/2025')
        rsf_run(a)
