from pathlib import Path
import os,json,hashlib,re
import numpy as np
import pandas as pd
D=Path(__file__).resolve().parent
ROOT=D.parents[1]
YEARS=[2023,2024,2025]
def run():return Path(os.environ['CAUSE_AUDIT_RUN'])
def load(p):return json.loads(Path(p).read_text())
def dump(p,z):Path(p).write_text(json.dumps(z,indent=2,ensure_ascii=False,allow_nan=False,default=str)+'\n')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def get_run(name):return Path(load(ROOT/'experiments'/name/'run_location.json')['private_run'])
def arrays(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def canon(s):return re.sub(r'\s+','',str(s)).replace('（','(').replace('）',')').replace('︰','：')
