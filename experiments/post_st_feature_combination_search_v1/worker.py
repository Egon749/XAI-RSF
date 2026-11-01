"""Independent, bounded batches of complete three-seed forest bundles."""
import argparse
import json
import os
from pathlib import Path
import resource
import socket
import time
import traceback

from experiments.post_st_dropcolumn_selection_v1.common import load, sha, write
from experiments.post_st_dropcolumn_selection_h24_v1.fitting import fit_bundle, bootstrap_groups


def execute(index):
    resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    started = time.perf_counter()
    root, out = Path(os.environ['DATA_DIR']), Path(os.environ['OUTPUT_DIR'])
    out.mkdir(parents=True, exist_ok=True)
    plan = load(root / 'plan.json')
    parent = Path(plan['parent'])
    records = []
    for i, item in enumerate(plan['batches'][index]):
        groups = bootstrap_groups(parent, item['year'], item['repeat']) if 'repeat' in item else None
        folder = out / f'bundle_{i}'
        fit_bundle(parent, item['keep'], item['year'], folder, groups,
                   outer_final=item.get('outer_final', False))
        records.append(dict(**item, relative_path=folder.name, new_forests=3))
    summary = dict(status='COMPLETE', records=records, new_forests=3 * len(records),
                   host=socket.gethostname(), seconds=time.perf_counter() - started,
                   peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                   code_commit=plan['code_commit'], no_outer_labels_read=True,
                   affinity=sorted(os.sched_getaffinity(0)))
    write(out / 'summary.json', summary)
    write(out / 'hashes.json', {str(p.relative_to(out)): sha(p)
                              for p in out.rglob('*') if p.is_file()})
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args()
    try:
        execute(args.index)
    except Exception:
        out = Path(os.environ['OUTPUT_DIR'])
        out.mkdir(parents=True, exist_ok=True)
        write(out / 'FAILED.json', dict(index=args.index, traceback=traceback.format_exc()))
        raise
