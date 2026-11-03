import os,socket,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
exe='/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python' if socket.gethostname()=='lemon' else '/home/oliver/.venvs/xai-baselines-20261027/bin/python'
os.environ.update(PYTHONPATH=str(root),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
ensemble='--ensemble' in sys.argv
args=[a for a in sys.argv[1:] if a!='--ensemble']
module='ensemble' if ensemble else 'worker'
os.execv(exe,[exe,'-m','experiments.post_st_followup_cap_v1.'+module,*args])
