"""Authorized named function release; fixed training reference, no fitting.

Only stable-by-selection functions receive percentile-coordinate shape tables.
No original cutpoints, company IDs, individual inputs or predictions exported.
"""
import argparse
import csv
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
from crforest import CIFExpansion


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def csv_write(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, identities, values):
    output = []
    keys = sorted({tuple(r[k] for k in identities) for r in rows})
    for scope in ['V19', 'V20', 'six_models_equal']:
        for key in keys:
            selected = [r for r in rows if tuple(r[k] for k in identities) == key
                        and (scope == 'six_models_equal' or r['fold'] == scope)]
            assert len(selected) == (6 if scope == 'six_models_equal' else 3)
            result = dict(scope=scope, **dict(zip(identities, key)),
                          models=len(selected), selected_models=sum(r['selected'] for r in selected))
            for value in values:
                vals = [r[value] for r in selected]
                result[value+'_mean'] = float(np.mean(vals))
                result[value+'_min'] = float(np.min(vals))
                result[value+'_max'] = float(np.max(vals))
            output.append(result)
    return output


def main(run, output):
    output.mkdir(parents=True, exist_ok=False)
    fields = Path(__file__).with_name('input_fields.json')
    names = json.loads(fields.read_text())['ordered_names']
    plan = json.loads((run/'input/plan.json').read_text())
    receipt = json.loads((run/'reports/execution_receipt.json').read_text())
    signatures = {fields: sha(fields)}
    models, sources = {}, {}
    for path in sorted((run/'results/assembled').glob('*/task-*/output/summary.json')):
        summary = json.loads(path.read_text())
        if summary['variant'] != 'coverage':
            continue
        key = (summary['fold'], summary['seed'])
        assert key not in models
        sources[key] = path.parent
        expected = json.loads((path.parent/'output_hashes.json').read_text())
        for name in ['forest_PRIVATE.json', 'functions_PRIVATE.npz']:
            p = path.parent/name
            signatures[p] = sha(p)
            assert signatures[p] == expected[name]
        assert signatures[path.parent/'forest_PRIVATE.json'] == receipt['model_hashes'][f'{key[0]}/{key[1]}/coverage']
        models[key] = CIFExpansion.load(path.parent/'functions_PRIVATE.npz')
        assert len(models[key].mains) == summary['main_functions']
        assert len(models[key].pairs) == summary['interaction_functions']
    assert set(models) == {(f, s) for f in plan['folds'] for s in plan['seeds']}
    common_main = sorted(set.intersection(*(set(ex.mains) for ex in models.values())))
    common_pair = sorted(set.intersection(*(set(ex.pairs) for ex in models.values())))
    all_pairs = sorted(set.union(*(set(ex.pairs) for ex in models.values())))
    times = [90, 180, 365, 366]
    old_strength = list(csv.DictReader((Path(__file__).parent/'posthoc_explanation_audit/interaction_strength_audit.csv').open()))
    feature_rows, pair_rows, shape_models, checks, selection = [], [], [], [], []
    for fold in plan['folds']:
        path = run/'input'/f'{fold}_train_PRIVATE.npz'
        signatures[path] = sha(path)
        assert signatures[path] == plan['source_hashes'][path.name]
        with np.load(path, allow_pickle=False) as data:
            X = data['X']
        rows = np.linspace(0, len(X)-1, 512, dtype=int)
        reference = X[rows]
        for seed in plan['seeds']:
            ex = models[fold, seed]
            main = {j: ex.component(reference, (j,), times)[:, :, 0] for j in ex.mains}
            pairs = {key: ex.component(reference, key, times)[:, :, 0] for key in ex.pairs}
            _, base = ex._coeff(times)
            reconstructed = np.broadcast_to(base[:, 0], (len(reference), len(times))).copy()
            for val in list(main.values()) + list(pairs.values()):
                reconstructed += val
            error = float(np.max(np.abs(reconstructed - ex.predict(reference, times)[:, :, 0])))
            assert error < 1e-10
            main_abs = {j: np.abs(val).mean(0) for j, val in main.items()}
            pair_abs = {key: np.abs(val).mean(0) for key, val in pairs.items()}
            interaction_total = sum(pair_abs.values())
            function_total = sum(main_abs.values()) + interaction_total
            previous = next(r for r in old_strength if r['fold'] == fold and int(r['seed']) == seed)
            top5 = sum(sorted(v[2] for v in pair_abs.values())[-5:]) / interaction_total[2]
            assert abs(top5 - float(previous['top5_pair_absolute_contribution_share'])) < 1e-12
            for j, name in enumerate(names):
                z = np.zeros((512, len(times)))
                mj = main.get(j, z)
                allocated = mj.copy()
                magnitude = main_abs.get(j, np.zeros(len(times))).copy()
                for pair, val in pairs.items():
                    if j in pair:
                        allocated += .5*val
                        magnitude += .5*pair_abs[pair]
                aj = np.abs(mj).mean(0)
                allocated_abs = np.abs(allocated).mean(0)
                for t, day in enumerate(times):
                    feature_rows.append(dict(fold=fold, seed=seed, feature=name, days=day,
                         selected=j in main, main_mean_abs=float(aj[t]),
                         equal_pair_allocated_mean_abs=float(allocated_abs[t]),
                         absolute_magnitude_score=float(magnitude[t])))
            for key in all_pairs:
                strength = pair_abs.get(key, np.zeros(len(times)))
                for t, day in enumerate(times):
                    pair_rows.append(dict(fold=fold, seed=seed, feature1=names[key[0]],
                         feature2=names[key[1]], days=day, selected=key in pairs,
                         stable_all_six=key in common_pair, mean_abs=float(strength[t]),
                         share_of_interaction_absolute_sum=float(strength[t]/interaction_total[t]) if interaction_total[t] else 0.,
                         share_of_all_function_absolute_sum=float(strength[t]/function_total[t]) if function_total[t] else 0.))
            funs = []
            for key in [(j,) for j in common_main] + common_pair:
                cuts = [ex.cuts(key, j) for j in key]
                reps = list(product(*(ex.representatives(c) for c in cuts)))
                xx = np.zeros((len(reps), ex.n_features))
                xx[:, list(key)] = reps
                values = ex.component(xx, key, times)[:, :, 0]
                percentiles, missing_fraction = [], []
                for j, cc in zip(key, cuts):
                    observed = X[np.isfinite(X[:, j]), j]
                    percentiles.append([float(100*np.mean(observed <= cut)) for cut in cc])
                    missing_fraction.append(float(np.isnan(X[:, j]).mean()))
                funs.append(dict(order=len(key), feature_names=[names[j] for j in key],
                     cut_percentiles=percentiles, training_missing_fraction=missing_fraction,
                     spatial_shape=[len(c)+2 for c in cuts], values=values.tolist()))
            model = dict(fold=fold, seed=seed, days=times, cause=1,
                 baseline=base[:,0].tolist(), functions=funs,
                 value_layout='values[flattened_spatial_cell_C_order][time_index]',
                 interval_convention='Ordered finite bins, upper endpoint inclusive; missing is last bin on each axis',
                 percentile_definition='100 * P_train(X_j <= actual_cut | X_j observed); training fold specific. Raw cutpoints withheld.',
                 source_functions_sha256=sha(sources[fold,seed]/'functions_PRIVATE.npz'))
            # Lossless serialized function values; percentile coordinates do not permit raw-input prediction.
            roundtrip = json.loads(json.dumps(model, allow_nan=False))
            for a,b in zip(model['functions'],roundtrip['functions']):
                assert np.array_equal(np.asarray(a['values']),np.asarray(b['values']))
            shape_models.append(model)
            selection.append(dict(fold=fold, seed=seed, features=[names[j] for j in ex.mains],
                                  pairs=[[names[a],names[b]] for a,b in ex.pairs]))
            checks.append(dict(fold=fold, seed=seed, training_reference_records=512,
                 target_component_reconstruction_error=error, prior_365_day_audit_parity=True,
                 serialized_function_values_exact=True))
    feature_summary = summarize(feature_rows, ['feature','days'],
             ['main_mean_abs','equal_pair_allocated_mean_abs','absolute_magnitude_score'])
    pair_summary = summarize(pair_rows, ['feature1','feature2','days'],
             ['mean_abs','share_of_interaction_absolute_sum','share_of_all_function_absolute_sum'])
    stable_pair_names = {(names[a],names[b]) for a,b in common_pair}
    stable = [r for r in pair_summary if (r['feature1'],r['feature2']) in stable_pair_names]
    csv_write(output/'feature_importance_by_model.csv', feature_rows)
    csv_write(output/'feature_importance_summary.csv', feature_summary)
    csv_write(output/'interaction_importance_by_model.csv', pair_rows)
    csv_write(output/'interaction_importance_summary.csv', pair_summary)
    csv_write(output/'stable_interaction_shares.csv', stable)
    dump(output/'selected_functions.json',selection)
    dump(output/'stable_functions_percentile.json', dict(
         stable_definition='Selected in all six frozen models; NOT proof of function-shape stability',
         stable_main_names=[names[j] for j in common_main],
         stable_pairs=[[names[a],names[b]] for a,b in common_pair],
         display_horizons={'3_month_approx_days':90,'6_month_approx_days':180,'12_month_display_days':365,
                           'leap_year_12_month_alternative_days':366},
         units='CIF probability contribution, not hazard or causal effect',models=shape_models))
    for path,digest in signatures.items():
        assert sha(path) == digest
    dump(output/'export_receipt.json',dict(status='READ_ONLY_FROZEN_FUNCTION_RELEASE',
         new_training=False, models=6, stable_main_count=len(common_main),stable_pair_count=len(common_pair),
         all_pair_count=len(all_pairs),all_input_count=len(names),
         source_training_commit=plan['code_commit'],script_sha256=sha(Path(__file__)),
         reference='Per fold: 512 fixed training rows at linspace(0,n-1,512,dtype=int), identical for all seeds; existing compiled product reference unchanged.',
         horizon_caveat='90/180 day curves are explicitly approximate 3/6 month displays. 365 and 366 both retained. No calendar-month performance metric recomputed.',
         seed_range_caveat='Within-fold min/max across three seeds is algorithmic variation, not a confidence interval. Six-model range also includes fold variation.',
         main_importance='mean(abs(f_j)); equal_pair_allocated_mean_abs = mean(abs(f_j + 0.5*sum incident f_jk)); absolute_magnitude_score = mean(abs(f_j)) + 0.5*sum mean(abs(f_jk)). Distinct measures, not interchangeable.',
         interaction_share='mean(abs(f_jk)) divided by sum of all interaction absolute means (or all main+interaction absolute means), baseline excluded. Not probability share, causal share or deletion utility.',
         privacy='Names and learned function values authorized; cuts replaced with conditional-on-observed training percentile positions. No company IDs/records, raw cuts, per-company predictions, or raw fitted forests exported.',
         checks=checks,source_files_unchanged=len(signatures),source_sha256={str(p):s for p,s in signatures.items()}))
    print(json.dumps(dict(models=6,stable_main=len(common_main),stable_pairs=len(common_pair),
                         source_files_unchanged=len(signatures))),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    main(args.run,args.output)
