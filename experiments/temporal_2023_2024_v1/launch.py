"""Select the verified node interpreter and inherit scheduler affinity."""
import os
from pathlib import Path
import socket
import sys

root=Path(__file__).resolve().parents[2]
exe='/home/oliver/MainDiskData/newrsf2_real_v1_20261026/venv/bin/python' if socket.gethostname().startswith('oliver') else '/home/lemon/.venvs/newrsf2-real-v1-20261026/bin/python'
os.environ.update(PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
os.execv(exe,[exe,str(Path(__file__).with_name('worker.py')),*sys.argv[1:]])
