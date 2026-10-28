"""Node-local original interpreter; immutable original learner first on import path."""
import os,sys,socket
from pathlib import Path
source=Path(os.environ['DATA_DIR']);run=source.parent
python='/home/lemon/.venvs/newrsf2-real-v1-20261026/bin/python' if socket.gethostname().startswith('lemon') else '/home/oliver/MainDiskData/newrsf2_real_v1_20261026/venv/bin/python'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['PYTHONPATH']=str(run/'input/frozen_code')+':'+str(run/'driver')
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.chdir(run/'input/frozen_code')
os.execv(python,[python,str(run/'driver/worker.py'),*sys.argv[1:]])
