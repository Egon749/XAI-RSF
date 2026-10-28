"""One independent q2 fit; reuse established AJ/explanation validation worker."""
import argparse, os, json, traceback
from pathlib import Path
from experiments.post_st_prognosis_v1.worker import run

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--year',type=int,required=True)
    p.add_argument('--variant',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--pilot',action='store_true');a=p.parse_args();a.model='xai_q2'
    os.environ['DATA_DIR']=str(Path(os.environ['DATA_DIR'])/'folds'/str(a.year)/a.variant)
    try:run(a)
    except Exception:
        out=Path(os.environ['OUTPUT_DIR']);out.mkdir(exist_ok=True,parents=True)
        (out/'FAILED.json').write_text(json.dumps({'args':vars(a),'traceback':traceback.format_exc()},indent=2))
        raise
