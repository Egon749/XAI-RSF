"""Use the unchanged tested no-fit exporter; only three missing counts."""
import argparse,os,traceback
from experiments.post_st_tree_count_local_v1.worker import main
from .common import *
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--year',type=int,choices=YEARS,required=True)
    p.add_argument('--seed',type=int,choices=[0,*SEEDS],required=True);p.add_argument('--count',type=int,choices=NEW_COUNTS,required=True);a=p.parse_args()
    try:main(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(parents=True,exist_ok=True)
        write(out/'FAILED.json',dict(args=vars(a),traceback=traceback.format_exc()));raise

