"""Select the already verified node-specific interpreter."""
import os,socket,sys
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
python='/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python' if socket.gethostname()=='lemon' else '/home/oliver/.venvs/xai-baselines-20261027/bin/python'
os.execv(python,[python,'-m','experiments.post_st_dropcolumn_selection_h24_v1.worker',*sys.argv[1:]])
