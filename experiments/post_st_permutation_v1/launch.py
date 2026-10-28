"""Use an existing verified interpreter on each node; no environment changes."""
import os
import socket
import sys
from pathlib import Path

exe=('/home/oliver/.venvs/xai-baselines-20261027/bin/python' if socket.gethostname().startswith('oliver')
     else '/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python')
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
os.execv(exe,[exe,str(Path(__file__).with_name('permutation.py')),'worker',*sys.argv[1:]])
