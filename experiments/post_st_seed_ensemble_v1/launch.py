import os,socket,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
exe='/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python'if socket.gethostname()=='lemon'else'/home/oliver/.venvs/xai-baselines-20261027/bin/python'
os.environ.update(PYTHONPATH=str(root),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
os.execv(exe,[exe,'-m','experiments.post_st_seed_ensemble_v1.worker',*sys.argv[1:]])
