import os,sys,socket
from pathlib import Path
repo=Path(__file__).resolve().parents[2]
exe='/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python' if socket.gethostname()=='lemon' else '/home/oliver/.venvs/xai-baselines-20261027/bin/python'
os.environ.update(PYTHONPATH=str(repo),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
os.execv(exe,[exe,'-m','experiments.post_st_probability_repair_v1.worker',*sys.argv[1:]])
