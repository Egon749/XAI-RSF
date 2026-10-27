"""Read-only audit of frozen coverage models; no fitting, pruning or prediction edits.

Exports aggregate diagnostics only. Does not export coordinates, thresholds,
company identifiers, individual contributions, or learned function tables.
"""
import argparse
import csv
import hashlib
import json
from itertools import combinations, product
from pathlib import Path

import numpy as np
from crforest import CIFExpansion
from crforest.partition import tree_feature_subset


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_write(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(run, output):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((run/'input/plan.json').read_text())
    receipt = json.loads((run/'reports/execution_receipt.json').read_text())
    paths, signatures = {}, {}
    for path in sorted((run/'results/assembled').glob('*/task-*/output/summary.json')):
        summary = json.loads(path.read_text())
        if summary['variant'] == 'coverage':
            key = (summary['fold'], summary['seed'])
            assert key not in paths
            paths[key] = path.parent
            for name in ['forest_PRIVATE.json', 'functions_PRIVATE.npz', 'predictions_PRIVATE.npz', 'output_hashes.json']:
                signatures[path.parent/name] = sha(path.parent/name)
            expected = json.loads((path.parent/'output_hashes.json').read_text())
            for name in ['forest_PRIVATE.json', 'functions_PRIVATE.npz', 'predictions_PRIVATE.npz']:
                assert signatures[path.parent/name] == expected[name]
            assert signatures[path.parent/'forest_PRIVATE.json'] == receipt['model_hashes'][f'{key[0]}/{key[1]}/coverage']
    assert set(paths) == {(f, s) for f in plan['folds'] for s in plan['seeds']}
    models = {key: CIFExpansion.load(path/'functions_PRIVATE.npz') for key, path in paths.items()}
    compression = []
    for (fold, seed), ex in sorted(models.items()):
        original, merged, removed, zeros = 0, 0, 0, 0
        for key in [(j,) for j in ex.mains] + ex.pairs:
            reps = [ex.representatives(ex.cuts(key, j)) for j in key]
            values = np.array(list(product(*reps)))
            xx = np.zeros((len(values), ex.n_features))
            xx[:, list(key)] = values
            shape = [len(v) for v in reps]
            yy = ex.component(xx, key).reshape(*shape, len(ex.grid), ex.coefficients.shape[-1])
            reduced = list(shape)
            for axis, n in enumerate(shape):
                # Missing state remains separate. Equality must hold over the
                # entire other axis, ALL native times and ALL causes.
                for i in range(n - 2):
                    if np.array_equal(np.take(yy, i, axis=axis), np.take(yy, i+1, axis=axis)):
                        reduced[axis] -= 1
                        removed += 1
            original += int(np.prod(shape))
            merged += int(np.prod(reduced))
            zeros += int(np.all(yy == 0))
        compression.append(dict(fold=fold, seed=seed, original_cells=original,
            exact_axis_merged_cells=merged, removable_boundaries=removed, exact_zero_functions=zeros))
    exposures = {}
    for seed in plan['seeds']:
        cfg = plan['configs'][f"{plan['folds'][0]}/{seed}/coverage"]
        for fold in plan['folds']:
            assert plan['configs'][f'{fold}/{seed}/coverage'] == cfg
        p = models[plan['folds'][0], seed].n_features
        count = {pair: 0 for pair in combinations(range(p), 2)}
        for tree in range(cfg['n_trees']):
            for pair in combinations(tree_feature_subset(p, cfg['tree_feature_count'], seed, tree), 2):
                count[pair] += 1
        exposures[seed] = count
    schedule = []
    for a, b in combinations(plan['seeds'], 2):
        aa = {k for k, v in exposures[a].items() if v}
        bb = {k for k, v in exposures[b].items() if v}
        schedule.append(dict(seed_a=a, seed_b=b, eligible_pair_jaccard=len(aa & bb)/len(aa | bb),
            exposure_count_correlation=float(np.corrcoef(list(exposures[a].values()), list(exposures[b].values()))[0, 1])))
    strengths = []
    for fold in plan['folds']:
        path = run/'input'/f'{fold}_train_PRIVATE.npz'
        signatures[path] = sha(path)
        assert signatures[path] == plan['source_hashes'][path.name]
        with np.load(path, allow_pickle=False) as data:
            X = data['X']
        xx = X[np.linspace(0, len(X)-1, 512, dtype=int)]
        sets = [set(models[fold, seed].pairs) for seed in plan['seeds']]
        common, union = set.intersection(*sets), set.union(*sets)
        for seed in plan['seeds']:
            ex = models[fold, seed]
            strength = {key: float(np.abs(ex.component(xx, key, [365])[:, 0, 0]).mean()) for key in ex.pairs}
            total = sum(strength.values())
            strengths.append(dict(fold=fold, seed=seed, pairs=len(ex.pairs), common_all_three_pairs=len(common),
                union_all_three_pairs=len(union), common_pair_absolute_contribution_share=sum(v for k, v in strength.items() if k in common)/total if total else None,
                top5_pair_absolute_contribution_share=sum(sorted(strength.values(), reverse=True)[:5])/total if total else None))
    for path, digest in signatures.items():
        assert sha(path) == digest, 'Frozen input changed during read-only audit'
    csv_write(output/'compression_audit.csv', compression)
    csv_write(output/'pair_exposure_audit.csv', schedule)
    csv_write(output/'interaction_strength_audit.csv', strengths)
    result = dict(status='READ_ONLY_NO_TRAINING',models=len(models),training_reference_points=512,
        contribution_horizon_days=365,contribution_cause=1,source_training_commit=plan['code_commit'],
        audit_script_sha256=sha(Path(__file__)),source_files_unchanged=len(signatures),
        model_sha256={f'{f}/{s}': signatures[p/'forest_PRIVATE.json'] for (f,s),p in sorted(paths.items())},
        notes=['No fits, new predictions, pruning or merges applied.',
               'Strict floating-point equality of whole adjacent finite-axis slices at all native times and causes; missing state separate. Not a proof that all equivalent compressions are impossible.',
               'Exposure is input-pair co-eligibility, not actual candidate evaluation or split selection.',
               'Absolute contribution shares are descriptive, not additive risk shares, predictive importance or evidence supporting deletion.',
               'Same fixed 512 training reference rows as prior stability audit; not a new independent sample.'],
        compression=compression,schedule=schedule,strength=strengths)
    with (output/'audit_receipt.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps(dict(status=result['status'],models=len(models),unchanged_files=len(signatures),output=str(output))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True, help='New unique directory; must not already exist')
    args = parser.parse_args()
    main(args.run, args.output)
