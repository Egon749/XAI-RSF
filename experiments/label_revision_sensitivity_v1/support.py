from pathlib import Path
import hashlib,json
BASE=Path('/home/oliver/MainDiskData/xai_rsf_interpretable_baselines_v1_20261027_hw9gdrol')
ORIGINAL=Path('/home/oliver/MainDiskData/xai_rsf_manufacturing_2023_2024_v1_20261027_XvjGsHee')
PATCH=Path('/home/oliver/MainDiskData/label_audit_clock_v1_20261028T133649Z')
TRAIN_SHA='206e414d05e8cd15e13429637bc6efc7f0abfcd4'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str))
def register(run,paths):
 p=Path(run)/'audit/source_hashes_before.json';d=json.loads(p.read_text()) if p.exists() else {}
 for f in paths:
  f=Path(f).resolve()
  if str(f) not in d:d[str(f)]=sha(f)
 write(p,d);return d
def arrays(p):
 import numpy as np
 with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def semantic(a):
 import numpy as np
 a=np.asarray(a);h=hashlib.sha256();h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes());return h.hexdigest()
