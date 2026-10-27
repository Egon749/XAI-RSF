"""Execute immutable snapshot in a verified node-local environment."""
import os,socket,sys
from pathlib import Path
args=sys.argv[1:];i=args.index('--kind');kind=args[i+1];del args[i:i+2]
root=Path(__file__).resolve().parents[2]
exe='/home/oliver/MainDiskData/newrsf2_real_v1_20261026/venv/bin/python' if socket.gethostname().startswith('oliver') else '/home/lemon/.venvs/newrsf2-real-v1-20261026/bin/python'
os.environ.update(PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
script=root/('experiments/development_v1/worker.py' if kind=='real' else 'experiments/crforest_pathbudget_v1/worker.py')
os.execv(exe,[exe,str(script),*args])
