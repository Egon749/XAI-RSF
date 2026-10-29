"""Use the already validated host-specific Python environment."""
import os,socket,sys
from pathlib import Path
node=socket.gethostname().split('.')[0]
py='/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python' if node=='lemon' else '/home/oliver/.venvs/xai-baselines-20261027/bin/python'
repo=Path(__file__).resolve().parents[2]
os.environ['PYTHONPATH']=str(repo)
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.execv(py,[py,'-m','experiments.shap_alignment_v1.worker',*sys.argv[1:]])
