"""Use installed node-specific single-thread runtime."""
import os,socket,sys
from pathlib import Path
code=Path(__file__).resolve().parents[2]
exe=('/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python' if socket.gethostname()=='lemon'
     else '/home/oliver/.venvs/xai-baselines-20261027/bin/python')
os.environ.update(PYTHONPATH=str(code),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
os.execv(exe,[exe,'-m','experiments.post_st_feature_pool_grid_v1.worker',*sys.argv[1:]])
