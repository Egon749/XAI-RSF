import os,sys,socket
from pathlib import Path
os.environ['PYTHONPATH']=str(Path(__file__).resolve().parents[2])
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
p='/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python' if socket.gethostname().startswith('lemon') else '/home/oliver/.venvs/xai-baselines-20261027/bin/python'
os.execv(p,[p,'-m','experiments.post_st_risk_index_v2.worker',*sys.argv[1:]])
