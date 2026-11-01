"""Use the verified, matching per-node Python environment."""
import os
import socket
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
python = ('/home/lemon/.venvs/xai-baselines-py313-20261027/bin/python'
          if socket.gethostname() == 'lemon'
          else '/home/oliver/.venvs/xai-baselines-20261027/bin/python')
os.environ.update(PYTHONPATH=str(root), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
os.execv(python, [python, '-m', 'experiments.post_st_feature_combination_search_v1.worker',
                 *sys.argv[1:]])
