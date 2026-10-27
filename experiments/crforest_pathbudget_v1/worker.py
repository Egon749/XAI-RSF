"""One independent scenario/seed, all five paired models; pool-managed outputs."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import time
import traceback
import numpy as np
import pandas as pd
from crforest import ForestConfig, CompetingRiskForest, CIFExpansion, cluster_bootstrap
from simulation import true_cif
from metrics import evaluate, probability_audit

RSCRIPT = '/home/oliver/MainDiskData/financial_diagnosis/csf_joint_sensitivity_v1/environment/r/bin/Rscript'
RLIB = '/home/oliver/MainDiskData/ernet_original_tau0_full_20261015/dynamic_competing_h24_20261015/rsf_er_competing_20261016/environment/rlib'


def write_json(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def reference_fit(train, test, config, out):
    names = [f'x{i}' for i in range(train['X'].shape[1])]
    df = pd.DataFrame(train['X'], columns=names)
    df.insert(0, 'status', train['event']); df.insert(0, 'time', train['time'])
    df.to_csv(out/'train.csv', index=False)
    pd.DataFrame(test['X'], columns=names).to_csv(out/'predict.csv', index=False)
    inbag = np.column_stack([np.bincount(cluster_bootstrap(train['groups'], config.seed, i), minlength=len(df))
                            for i in range(config.n_trees)])
    np.savetxt(out/'inbag.csv', inbag, fmt='%d', delimiter=',')
    cfg = dict(train=str(out/'train.csv'), predict=str(out/'predict.csv'), inbag=str(out/'inbag.csv'),
               output=str(out), trees=config.n_trees, seed=config.seed, nodesize=config.min_leaf,
               depth=config.max_depth, nsplit=config.n_thresholds, mtry=config.mtry or int(np.ceil(np.sqrt(len(names)))))
    write_json(out/'r_config.json', cfg)
    env = os.environ.copy(); env['EW_RLIB'] = RLIB
    with (out/'R.log').open('x') as log:
        subprocess.run([RSCRIPT, '--vanilla', str(Path(__file__).with_name('reference.R')), str(out/'r_config.json')],
                       stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    meta = json.loads((out/'r_meta.json').read_text())
    p = np.fromfile(out/'cif.bin', dtype='<f8').reshape(meta['dims'], order='F')
    grid = np.atleast_1d(np.loadtxt(out/'times.csv'))
    return p, grid, meta


def sample_support(expansion, X_train, x, key):
    valid = np.ones(len(X_train), bool)
    for j in key:
        cuts = expansion.cuts(key, j)
        b = np.searchsorted(cuts, X_train[:, j], side='left')
        b[np.isnan(X_train[:, j])] = len(cuts)+1
        v = len(cuts)+1 if np.isnan(x[j]) else np.searchsorted(cuts, x[j], side='left')
        valid &= b == v
    return int(valid.sum())


def run(args):
    started = time.perf_counter()
    out = Path(args.output or os.environ['OUTPUT_DIR']); out.mkdir(exist_ok=True, parents=True)
    if (out/'summary.json').exists() or (out/'FAILED.json').exists(): raise FileExistsError(out)
    resource.setrlimit(resource.RLIMIT_AS, (6 * 1024**3, 6 * 1024**3))
    source = Path(args.input or os.environ['DATA_DIR'])
    def read_npz(name):
        with np.load(source/name, allow_pickle=False) as a: return {k:a[k] for k in a.files}
    train = read_npz(f'{args.scenario}_{args.seed}_train.npz')
    test = read_npz(f'{args.scenario}_test.npz')
    reference = np.load(source/f'{args.scenario}_reference.npy')
    cfg = ForestConfig(**json.loads((source/'config.json').read_text())['forest'], seed=args.seed)
    prescribed = np.array([.5, 1., 2.])
    metrics, calibration, summaries = [], [], []
    variants = [('A_rfsrc', None), ('B_unrestricted', cfg), ('C_depth2', replace(cfg, max_depth=2)),
                ('D_q2', replace(cfg, path_budget=2)), ('E_q1', replace(cfg, path_budget=1))]
    for variant, config in variants:
        dest = out/variant; dest.mkdir()
        start = time.perf_counter()
        print(json.dumps(dict(event='fit_start', scenario=args.scenario, seed=args.seed, variant=variant)), flush=True)
        row = dict(variant=variant, scenario=args.scenario, seed=args.seed)
        if variant == 'A_rfsrc':
            native, grid, meta = reference_fit(train, test, cfg, dest)
            row.update(fit_predict_seconds=time.perf_counter()-start, implementation='randomForestSRC',
                       candidate_evaluations=None, leaves=int(sum(np.atleast_1d(meta['leaf_counts']))),
                       r_version=meta['version'], structure_audit='library; no path constraint',
                       probability_audit=probability_audit(native))
            ix = np.searchsorted(grid, prescribed, side='right')-1
            predictions = native[:, np.maximum(ix, 0)].copy(); predictions[:, ix<0] = 0.
        else:
            forest = CompetingRiskForest(config).fit(train['X'], train['time'], train['event'], train['groups'])
            row.update(fit_seconds=time.perf_counter()-start, implementation='crforest-reference', **forest.audit_)
            forest.save(dest/'forest_PRIVATE.json')
            predictions = forest.predict(test['X'], prescribed)
            grid = np.unique(np.r_[0., forest.grid_, prescribed])
            row['probability_audit'] = probability_audit(forest.predict(test['X'], grid))
            if variant in ('C_depth2', 'D_q2', 'E_q1'):
                expansion = CIFExpansion.from_forest(forest, reference)
                expansion.save(dest/'functions_PRIVATE.npz')
                exported = CIFExpansion.load(dest/'functions_PRIVATE.npz')
                max_error, checked = 0., 0
                for dataset in (train['X'], test['X']):
                    for start_at in range(0, len(dataset), 128):
                        xx = dataset[start_at:start_at+128]
                        original, reconstructed = forest.predict(xx, grid), exported.predict(xx, grid)
                        max_error = max(max_error, float(np.max(abs(original-reconstructed))))
                        probability_audit(original); probability_audit(reconstructed)
                        checked += original.size
                assert max_error <= 1e-10, max_error
                row.update(reconstruction_error=max_error, reconstruction_scalar_checks=checked,
                           native_time_count=len(forest.grid_), checked_time_count=len(grid))
                complexity = exported.complexity(train['X'])
                write_json(dest/'complexity.json', complexity)
                row.update({k:v for k,v in complexity.items() if k != 'functions'})
                # All components and both events at every specified time, no TopK.
                cases = []
                for case_id, xx in enumerate(test['X'][:3]):
                    mu = exported._coeff(prescribed)[1]
                    total = mu.copy(); components = []
                    for key in [(j,) for j in exported.mains] + exported.pairs:
                        val = exported.component(xx[None, :], key, prescribed)[0]
                        total += val
                        components.append(dict(features=list(map(int, key)), cif_contribution=val.tolist(),
                                               training_cell_support=sample_support(exported, train['X'], xx, key)))
                    assert np.max(abs(total-predictions[case_id])) <= 1e-10
                    cases.append(dict(case=f'S{case_id+1}', synthetic=True, times=prescribed.tolist(),
                                      reference_mean=mu.tolist(), components=components,
                                      reconstructed_cif=total.tolist(), forest_cif=predictions[case_id].tolist()))
                write_json(dest/'complete_cases_PRIVATE.json', cases)
        probability_audit(predictions)
        truth = true_cif(test['X'], prescribed, args.scenario)
        row['integrated_cif_mse_grid'] = float(np.mean((predictions-truth)**2))
        for j, horizon in enumerate(prescribed):
            for k in (1, 2):
                metric, cal = evaluate(test['time'], test['event'], predictions[:, j, k-1], horizon, k,
                                       truth[:, j, k-1])
                identifier = dict(scenario=args.scenario, seed=args.seed, variant=variant, horizon=horizon, cause=k)
                metrics.append({**identifier, **metric})
                calibration.extend([{**identifier, **c} for c in cal])
        np.savez_compressed(dest/'predictions_PRIVATE.npz', cif=predictions, grid=prescribed)
        row.update(total_model_seconds=time.perf_counter()-start)
        summaries.append(row)
        write_json(dest/'summary.json', row)
        print(json.dumps(dict(event='model_complete', **row)), flush=True)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(calibration).to_csv(out/'calibration.csv', index=False)
    write_json(out/'summary.json', dict(status='COMPLETED', scenario=args.scenario, seed=args.seed,
               models=summaries, seconds=time.perf_counter()-started, host=socket.gethostname(),
               peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
               child_peak_rss_MiB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024,
               affinity_count=len(os.sched_getaffinity(0)), python=platform.python_version(), numpy=np.__version__,
               train_n=len(train['X']), test_n=len(test['X']), reference_n=len(reference),
               train_events=np.bincount(train['event'], minlength=3).tolist(),
               test_events=np.bincount(test['event'], minlength=3).tolist(),
               training_early_censor_fraction=float(np.mean((train['event']==0)&(train['time']<2.))),
               source_config_sha256=hashlib.sha256((source/'config.json').read_bytes()).hexdigest()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input'); parser.add_argument('--output')
    parser.add_argument('--scenario', required=True); parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except Exception:
        dest = Path(args.output or os.environ['OUTPUT_DIR']); dest.mkdir(exist_ok=True, parents=True)
        if not (dest/'FAILED.json').exists():
            write_json(dest/'FAILED.json', dict(scenario=args.scenario, seed=args.seed, traceback=traceback.format_exc()))
        raise
