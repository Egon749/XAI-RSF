"""Identity-checked reuse of existing three-seed fits and new bundles."""
from pathlib import Path
import numpy as np
import pandas as pd

from experiments.post_st_dropcolumn_selection_v1.common import load, arrays, sha
from experiments.post_st_dropcolumn_selection_h24_v1.fitting import read_bundle


class Store:
    def __init__(self, parent, previous, run):
        self.parent, self.run = Path(parent), Path(run)
        self.registry = {}
        self.cache, self.used = {}, {}
        for record in load(self.parent / 'audit/registry_aggregate_inner.json'):
            if record['mode'] == 'count':
                self.registry[(record['year'], tuple(record['keep']))] = dict(
                    style='aggregate', path=record['path'], paths=record['paths'], source='parent')
        for base in [Path(previous), self.run]:
            for path in sorted((base / 'audit').glob('registry_*.json')):
                for r in load(path):
                    if r.get('kind') != 'bundle' or r.get('outer_final') or 'repeat' in r:
                        continue
                    location = r['path'] if 'path' in r else str(Path(r['output']) / 'bundle')
                    self.registry[(r['year'], tuple(r['keep']))] = dict(
                        style='bundle', path=location, source=str(base))

    def missing(self, keeps, years):
        return [dict(kind='bundle', keep=list(k), year=y)
                for k in sorted(set(map(tuple, keeps))) for y in sorted(set(years))
                if (y, k) not in self.registry]

    def ensure(self, keeps, years):
        missing = self.missing(keeps, years)
        if missing:
            raise ValueError(f'{len(missing)} historical fit bundles missing')

    def get(self, keep, year):
        key = (year, tuple(keep))
        if key in self.cache:
            return self.cache[key]
        r = self.registry[key]
        p = Path(r['path'])
        hashes = load(p / 'hashes.json')
        checked = {}
        # Check saved predictions/summary before trusting a reused score.
        for rel in ['summary.json', 'ensemble/predictions_PRIVATE.npz']:
            actual = sha(p / rel)
            assert actual == hashes[rel], (p, rel)
            checked[rel] = actual
        expected = arrays(self.parent / f'input/folds/{year}/eval_PRIVATE.npz')['sample_ids']
        saved = arrays(p / 'ensemble/predictions_PRIVATE.npz')
        np.testing.assert_array_equal(saved['sample_ids'], expected)
        if r['style'] == 'bundle':
            result = read_bundle(p)
            summary = load(p / 'summary.json')
            assert tuple(summary['keep']) == tuple(keep) and summary['year'] == year
        else:
            summary = load(p / 'summary.json')
            st = pd.read_csv(p / 'seed_stability.csv')
            seeds = []
            for folder in map(Path, r['paths']):
                z = arrays(folder / 'predictions_PRIVATE.npz')
                np.testing.assert_array_equal(z['sample_ids'], expected)
                seeds.append(z['cif'])
            result = dict(ensemble=saved['cif'], seeds=seeds,
                          cells=summary['check']['spatial_table_cells'],
                          pair_jaccard=float(st[st.order == 2].jaccard.mean()))
        assert np.max(abs(result['ensemble'] - np.mean(result['seeds'], axis=0))) < 1e-12
        self.used[key] = dict(year=year, keep=list(keep), **r, checked_hashes=checked)
        self.cache[key] = result
        return result
