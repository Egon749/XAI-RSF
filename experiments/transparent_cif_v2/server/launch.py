from pathlib import Path
import os,socket,sys
root=Path(__file__).resolve().parents[3]
exe='/home/oliver/.venvs/xai-baselines-20261027/bin/python' if socket.gethostname().startswith('oliver') else '/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python'
os.environ.update(PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
os.execv(exe,[exe,str(Path(__file__).with_name('worker.py')),*sys.argv[1:]])
