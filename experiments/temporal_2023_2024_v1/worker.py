"""Use the unchanged tested development worker with the authorized final fold."""
import argparse
import os
from pathlib import Path
import traceback
from experiments.search_coverage_v1.worker import run,write

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--fold',choices=['final'],default='final')
    p.add_argument('--variant',choices=['coverage','unrestricted'],required=True)
    p.add_argument('--seed',type=int,choices=[42,101,202],required=True)
    p.add_argument('--mode',choices=['fit','shard','assemble'],required=True)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--tree-start',type=int)
    p.add_argument('--tree-count',type=int,default=4)
    p.add_argument('--index')
    a=p.parse_args()
    try:run(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
        if not (out/'FAILED.json').exists():write(out/'FAILED.json',dict(fold='final',variant=a.variant,seed=a.seed,traceback=traceback.format_exc()))
        raise
