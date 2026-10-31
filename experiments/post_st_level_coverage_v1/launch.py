import os,socket,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
exe='/home/oliver/.venvs/xai-baselines-20261027/bin/python' if socket.gethostname().startswith('oliver') else '/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python'
os.environ.update(PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
module=sys.argv[1]
assert module in ['worker','sim_worker']
os.execv(exe,[exe,'-m',f'experiments.post_st_level_coverage_v1.{module}',*sys.argv[2:]])
