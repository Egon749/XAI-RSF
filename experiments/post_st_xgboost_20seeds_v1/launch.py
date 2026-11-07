import os,socket,sys,json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
deps=json.loads((root/'experiments/post_st_c061_xgboost_v1/run_location.json').read_text())['deps']
exe='/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python' if socket.gethostname()=='lemon' else '/home/oliver/.venvs/xai-baselines-20261027/bin/python'
os.environ.update(PYTHONPATH=deps+os.pathsep+str(root),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
os.execv(exe,[exe,'-m','experiments.post_st_xgboost_20seeds_v1.worker',*sys.argv[1:]])
