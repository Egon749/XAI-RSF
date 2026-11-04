"""Stage only historical arrays and code, not outer evaluation files."""
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import numpy as np
import pandas as pd
from experiments.post_st_eswa_baselines_v1.common import arrays, dump, load, sha
from experiments.post_st_paper_features_v1.common import history

D = Path(__file__).resolve().parent
ROOT = D.parents[1]
YEARS = (2023, 2024, 2025)


def main():
    r = Path(os.environ['DIAG_RUN'])
    source = Path(os.environ['ESWA_RUN'])
    inp = r / 'input'
    inp.mkdir()
    expected = load(D.parent / 'post_st_eswa_baselines_v1/DATA_ALIGNMENT_AUDIT.json')['hashes']
    audit = {}
    for year in YEARS:
        p = source / f'input/{year}/train.npz'
        folder = inp / str(year)
        folder.mkdir()
        dest = folder / 'train.npz'
        assert sha(p) == expected[f'{year}/train']
        shutil.copyfile(p, dest)
        assert sha(p) == sha(dest)
        data = arrays(dest)
        meta = history(year) / f'evaluation_input/{year}_fit_metadata_PRIVATE.parquet'
        m = pd.read_parquet(meta, columns=['sample_id', 'origin', 'stop'])
        np.testing.assert_array_equal(m.sample_id, data['sample_ids'])
        assert len(m) == m.sample_id.nunique() == len(set(data['groups'])) and data['X'].shape == (len(m), 74)
        assert (m.origin < pd.Timestamp(year, 1, 1)).all() and (m.stop < pd.Timestamp(year, 1, 1)).all()
        audit[year] = dict(companies=len(m), training_sha256=sha(dest), training_metadata_sha256=sha(meta))
    (inp / 'deps_location.txt').write_text(str(source / 'deps'))
    code = inp / 'code'
    code.mkdir()
    rev = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    paths = [str(Path('experiments') / D.name / name) for name in ['__init__.py','worker.py','launch.py']]
    paths += ['experiments/post_st_eswa_baselines_v1/' + f for f in ('__init__.py','common.py','proboost.py')]
    paths += ['experiments/post_st_benchmark_calibration_v1/' + f for f in ('__init__.py','common.py')]
    archive = subprocess.check_output(['git','archive',rev,'--',*paths], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as t:
        t.extractall(code, filter='data')
    tasks = []
    for start, stop in [(0,1)] + [(i,i+5) for i in range(1,201,5)]:
        for year in YEARS:
            args = ['python3', str(code / 'experiments' / D.name / 'launch.py'), str(year), str(start), str(stop)]
            tasks.append(dict(argv=args, cwd=str(code), kind='cpu', timeout=1800,
                              storage=dict(source=str(inp), mode='nfs', results=str(r/'published'))))
    manifest = r / 'tasks.jsonl'
    manifest.write_text(''.join(json.dumps(t) + '\n' for t in tasks))
    dump(r / 'snapshot.json', dict(worker_commit=rev, training_inputs=audit, tasks=len(tasks), manifest_sha256=sha(manifest)))
    subprocess.run([str(Path.home()/'.local/bin/hetero-run'), str(manifest)], check=True)


if __name__ == '__main__':
    main()
