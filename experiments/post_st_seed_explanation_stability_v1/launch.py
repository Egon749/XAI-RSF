import os,socket,sys
from pathlib import Path
exe=('/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python' if socket.gethostname()=='lemon' else '/home/oliver/.venvs/xai-baselines-20261027/bin/python')
os.environ.update(PYTHONPATH=str(Path(__file__).resolve().parents[2]),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
os.execv(exe,[exe,'-m',f'experiments.post_st_seed_explanation_stability_v1.{sys.argv[1]}',*sys.argv[2:]])
