import json,os,hashlib
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent
ROOT=D.parents[1]
YEARS=(2023,2024,2025)
PICK=(17,503,22)
ANCHOR=(42,101,202)
DAYS=np.array([365.,366.,730.,731.])
def run():return Path(os.environ['BENCH_RUN'])
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n')
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def arrays(p):
    with np.load(p,allow_pickle=False) as f:return dict(f)
def old(name):return Path(read(ROOT/'experiments'/name/'run_location.json')['private_run'])
def endpoints(cif,days):
    ix=np.searchsorted(DAYS,days);np.testing.assert_array_equal(DAYS[ix],days)
    return cif[np.arange(len(days)),ix]
