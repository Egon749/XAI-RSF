"""Use each node's verified research Python without changing RNG or workload."""
import os, socket, sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
os.environ['PYTHONPATH']=str(root)
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
python=('/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python' if socket.gethostname().startswith('lemon') else '/home/oliver/.venvs/xai-baselines-20261027/bin/python')
os.execv(python,[python,'-m','experiments.post_st_calendar_level_v1.verify',*sys.argv[1:]])
