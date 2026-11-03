"""Read saved 74-column/support/shrunk forests and functions; no fitting.

Public outputs contain aggregate feature/function statistics only. Companies,
cutpoints, function coefficients and individual contributions are never exported.
"""
import argparse
from collections import Counter
from itertools import combinations
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from crforest import CompetingRiskForest
from crforest.functions import CIFExpansion


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(run, out):
    start = time.perf_counter()
    out.mkdir(parents=True, exist_ok=False)
    private = Path(tempfile.mkdtemp(prefix='feature_usage_', dir=run / 'private'))
    inputs = {}

    def checked(path, expected=None):
        digest = sha(path)
        if expected is not None:
            assert digest == expected, str(path)
        inputs[str(path)] = digest
        return path

    def readj(path):
        return json.loads(checked(path).read_text())

    frozen = readj(run / 'audit/immutable_input_hashes.json')
    checked(Path(__file__).resolve())
    for name in ['crforest/core.py','crforest/functions.py','crforest/__init__.py']:
        checked(REPO/name, frozen['code/'+name])
    features = readj(checked(run / 'input/features.json', frozen['features.json']))
    schema = pd.read_csv(checked(REPO / '主模型/配置/INPUT_FEATURES.csv'))
    assert schema.feature_name.tolist() == [f['feature_name'] for f in features]
    assert schema.input_position.tolist() == list(range(74))
    formal = readj(run / 'audit/registry_formal.json')
    ensembles = readj(run / 'audit/registry_ensemble.json')
    pointer = checked(REPO / 'CURRENT_POST_ST_MODEL.json')
    pointer_hash = sha(pointer)
    # A runtime guard makes accidental calls to the training API fail immediately.
    def forbidden_fit(*args, **kwargs):
        raise RuntimeError('Fitting forbidden in saved-model analysis')
    CompetingRiskForest.fit = forbidden_fit
    times = np.array([365., 730.])
    usage, function_rows, summaries, checks, seed_rows = [], [], [], [], []

    def load_artifacts(row, sub):
        root = Path(row['path'])
        hashes = readj(root / 'hashes.json')
        paths = {}
        for name in ['forest_PRIVATE.json', 'functions_PRIVATE.npz',
                     'components_PRIVATE.npz', 'complexity_PRIVATE.json', 'verification.json']:
            rel = f'{sub}/{name}'
            paths[name] = checked(root / rel, hashes[rel])
        forest = CompetingRiskForest.load(paths['forest_PRIVATE.json'])
        assert forest.n_features_ == 74
        assert forest.config.path_budget == 2 and forest.config.max_depth == 4
        assert forest.config.split_min_target_groups == 2
        expansion = CIFExpansion.load(paths['functions_PRIVATE.npz'])
        z = np.load(paths['components_PRIVATE.npz'])
        assert np.array_equal(z['times'], [365., 366., 730., 731.])
        terms = {tuple(map(int, k.split(','))): z['terms'][:, j, [0, 2], :]
                 for j, k in enumerate(z['keys'])}
        complexity = readj(paths['complexity_PRIVATE.json'])
        return forest, expansion, terms, complexity

    for year in [2023, 2024, 2025]:
        trrel = f'folds/{year}/train_PRIVATE.npz'
        evrel = f'folds/{year}/eval_PRIVATE.npz'
        X = np.load(checked(run / 'input' / trrel, frozen[trrel]))['X']
        E = np.load(checked(run / 'input' / evrel, frozen[evrel]))['X']
        # Only X is read: neither training nor evaluation outcomes are used.
        rows = sorted([r for r in formal if r['year'] == year], key=lambda r:r['seed'])
        assert [r['seed'] for r in rows] == [42, 101, 202]
        seed_objects = [load_artifacts(r, 'shrunk') for r in rows]
        row = next(r for r in ensembles if r['year'] == year and r['version'] == 'shrunk')
        forest, expansion, terms, complexity = load_artifacts(row, 'representation')
        assert forest.trees_ == sum([s[0].trees_ for s in seed_objects], [])
        assert len(forest.trees_) == 96
        assert set(terms) == set().union(*(set(s[2]) for s in seed_objects))
        coverages = []
        for r, (f, ex, ts, comp) in zip(rows, seed_objects):
            ph = readj(Path(r['path']) / 'hashes.json')
            cv = pd.read_csv(checked(Path(r['path']) / 'feature_coverage_PRIVATE.csv',
                                    ph['feature_coverage_PRIVATE.csv']))
            assert cv.feature.tolist() == list(range(74))
            coverages.append(cv)
            calc = Counter(n['feature'] for t in f.trees_ for n in t if 'feature' in n)
            assert all(calc[j] == cv.iloc[j].split_nodes for j in range(74))
            # Rebuild only the algebraic reference masses, never the forest.
            fresh = CIFExpansion.from_forest(f, X)
            assert np.array_equal(fresh.mass, ex.mass)
            assert np.allclose(fresh.mu, ex.mu, atol=1e-14, rtol=0)
            for key, value in ts.items():
                assert np.max(abs(ex.component(X, key, times) - value)) <= 1e-12
        fresh = CIFExpansion.from_forest(forest, X)
        assert np.array_equal(fresh.mass, expansion.mass)
        assert np.allclose(fresh.mu, expansion.mu, atol=1e-14, rtol=0)
        keysets = [{tuple(v['features']) for v in s[3]['functions'] if v['nonzero']}
                   for s in seed_objects]
        nonzero = {tuple(v['features']) for v in complexity['functions'] if v['nonzero']}
        zero = np.zeros((len(X), 2, 2))
        component_error = mean_error = 0.
        for key, value in terms.items():
            component_error = max(component_error, float(np.max(abs(expansion.component(X, key, times) - value))))
            seedvalues = [s[2].get(key, zero) for s in seed_objects]
            mean_error = max(mean_error, float(np.max(abs(value - np.mean(seedvalues, axis=0)))))
            importance = np.mean(abs(value), axis=0) * 100
            seedimportance = np.array([np.mean(abs(v), axis=0) * 100 for v in seedvalues])
            for ti, h in enumerate([12, 24]):
                for ci, cause in enumerate([1, 2]):
                    corrs = []
                    for a, b in combinations(seedvalues, 2):
                        if np.std(a[:, ti, ci]) > 1e-14 and np.std(b[:, ti, ci]) > 1e-14:
                            corrs.append(float(np.corrcoef(a[:, ti, ci], b[:, ti, ci])[0, 1]))
                    function_rows.append(dict(year=year, cause=cause, horizon_months=h,
                        display_days=int(times[ti]), order=len(key),
                        input_positions=','.join(map(str,key)),
                        feature_names=' × '.join(features[j]['feature_name'] for j in key),
                        labels=' × '.join(features[j]['label'] for j in key),
                        mean_abs_contribution_pp=float(importance[ti,ci]),
                        seed_nonzero_native_count=sum(key in ks for ks in keysets),
                        seed_endpoint_nonzero_count=int(sum(np.max(abs(v[:,ti,ci]))>1e-14 for v in seedvalues)),
                        seed_min_pp=float(seedimportance[:,ti,ci].min()),
                        seed_max_pp=float(seedimportance[:,ti,ci].max()),
                        seed_sd_pp=float(seedimportance[:,ti,ci].std(ddof=1)),
                        shared_seed_correlation_min=min(corrs) if corrs else np.nan,
                        shared_seed_correlation_mean=float(np.mean(corrs)) if corrs else np.nan,
                        defined_seed_pairs=len(corrs),
                        nonzero_all_native_times_both_causes=key in nonzero))
                    for i, seed in enumerate([42,101,202]):
                        seed_rows.append(dict(year=year, seed=seed, cause=cause,
                            horizon_months=h, input_positions=','.join(map(str,key)), order=len(key),
                            mean_abs_contribution_pp=float(seedimportance[i,ti,ci])))
        assert max(component_error, mean_error) <= 1e-12
        for j, meta in enumerate(features):
            usage.append(dict(year=year, input_position=j, **meta,
                eligible_trees=sum(int(c.iloc[j].eligible_trees) for c in coverages),
                split_trees=sum(int(c.iloc[j].split_trees) for c in coverages),
                split_nodes=sum(int(c.iloc[j].split_nodes) for c in coverages),
                used_for_split=any(c.iloc[j].split_nodes>0 for c in coverages),
                split_seed_count=sum(c.iloc[j].split_nodes>0 for c in coverages),
                nonzero_main_native=(j,) in nonzero))
        # Independently sum every component on all train/evaluation rows and native
        # times. No saved predictions or outcome labels are needed for this audit.
        max_error = 0.
        for name, XX in [('train',X),('evaluation',E)]:
            for begin in range(0,len(XX),64):
                xx = XX[begin:begin+64]
                pred = forest.predict(xx)
                total = np.broadcast_to(expansion.mu, pred.shape).copy()
                for key in terms:
                    total += expansion.component(xx,key)
                max_error=max(max_error,float(np.max(abs(pred-total))))
                assert pred.min()>=-1e-12 and pred.sum(axis=2).max()<=1+1e-12
                assert np.diff(pred,axis=1).min()>=-1e-12
        assert max_error<=1e-10
        checks.append(dict(year=year, source_models=3, merged_trees=96,
            verified_training_rows=len(X),verified_evaluation_rows=len(E),
            native_times=len(forest.grid_),causes=2,component_sum_max_error=max_error,
            saved_component_error=component_error,ensemble_seed_mean_error=mean_error,
            max_path_variables=forest.audit()['max_path_variables'],probability_pass=True))
        summaries.append(dict(year=year,inputs=74,used_variables=sum(u['used_for_split'] for u in usage if u['year']==year),
            unused_variables=sum(not u['used_for_split'] for u in usage if u['year']==year),
            main_functions=complexity['main_functions'],pair_functions=complexity['interaction_functions'],
            spatial_cells=complexity['spatial_table_cells'],training_reference_n=len(X)))
        print(json.dumps(summaries[-1]),flush=True)

    ff = pd.DataFrame(function_rows)
    scope=['year','cause','horizon_months']
    ff['share_all_component_magnitude_pct']=ff.mean_abs_contribution_pp / ff.groupby(scope).mean_abs_contribution_pp.transform('sum')*100
    ff['share_within_order_pct']=ff.mean_abs_contribution_pp / ff.groupby(scope+['order']).mean_abs_contribution_pp.transform('sum')*100
    ff['rank_within_order']=ff.groupby(scope+['order']).mean_abs_contribution_pp.rank(method='min',ascending=False).astype(int)
    ff.sort_values(scope+['order','rank_within_order']).to_csv(out/'FUNCTION_IMPORTANCE.csv',index=False)
    pd.DataFrame(seed_rows).to_csv(out/'SEED_IMPORTANCE.csv',index=False)
    uu=pd.DataFrame(usage)
    uu.to_csv(out/'FEATURE_USAGE_LONG.csv',index=False)
    wide=schema.copy()
    for y in [2023,2024,2025]:
        a=uu[uu.year==y].sort_values('input_position')
        for col in ['used_for_split','split_trees','split_seed_count','nonzero_main_native']:
            wide[f'{y}_{col}']=a[col].to_numpy()
    wide.to_csv(out/'ALL_74_USAGE.csv',index=False)
    all_main=[]
    for u in usage:
        for h in [12,24]:
            for cause in [1,2]:
                sub=ff[(ff.year==u['year'])&(ff.horizon_months==h)&(ff.cause==cause)
                       &(ff.order==1)&(ff.input_positions==str(u['input_position']))]
                all_main.append(dict(year=u['year'],horizon_months=h,cause=cause,
                    input_position=u['input_position'],feature_id=u['feature_id'],
                    feature_name=u['feature_name'],label=u['label'],feature_group=u['feature_group'],
                    used_for_split=u['used_for_split'],
                    **{k:(sub.iloc[0][k] if len(sub) else 0.) for k in
                       ['mean_abs_contribution_pp','seed_min_pp','seed_max_pp','seed_sd_pp',
                        'seed_endpoint_nonzero_count']}))
    pd.DataFrame(all_main).to_csv(out/'MAIN_IMPORTANCE_ALL_74.csv',index=False)
    active=ff.assign(active=ff.mean_abs_contribution_pp>1e-12,
                     all_three=ff.seed_endpoint_nonzero_count==3).groupby(scope+['order']).agg(
                         active_functions=('active','sum'),
                         three_seed_nonzero_functions=('all_three','sum')).reset_index()
    active.to_csv(out/'ENDPOINT_FUNCTION_COUNTS.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'COMPLEXITY_SUMMARY.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'VALIDATION.csv',index=False)
    totals=ff.groupby(scope+['order']).mean_abs_contribution_pp.sum().unstack('order')
    totals.columns=['main_sum_pp','pair_sum_pp']
    totals['pair_share_pct']=totals.pair_sum_pp/(totals.main_sum_pp+totals.pair_sum_pp)*100
    totals.reset_index().to_csv(out/'MAIN_PAIR_MAGNITUDE.csv',index=False)
    groups=uu.groupby(['year','feature_group']).agg(input_count=('feature_name','size'),used_count=('used_for_split','sum')).reset_index()
    groups.to_csv(out/'FEATURE_GROUP_USAGE.csv',index=False)
    assert all(sha(p)==s for p,s in inputs.items()), 'Source mutation detected'
    assert sha(pointer)==pointer_hash
    receipt=dict(status='SAVED_MODEL_ANALYSIS_COMPLETE',forest_fits=0,calibrator_fits=0,
        code_commit_at_start=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        model_id='post_st_drop9_support_v1/drop9_support/shrunk',years=[2023,2024,2025],
        seeds=[42,101,202],loaded_seed_models=9,loaded_ensemble_models=3,
        training_labels_read=False,evaluation_labels_read=False,
        reference='annual training empirical marginal product; magnitude averaged on observed joint training rows',
        display_days=[365,730],threshold_for_numerical_nonzero=1e-14,
        current_pointer_sha256=pointer_hash,private_audit=str(private),
        input_sha256=inputs,python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
        elapsed_seconds=time.perf_counter()-start,failures=[],
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (private/'execution_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    (out/'execution_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    print('OUTPUT',out,'PRIVATE_AUDIT',private,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    main(a.run,a.output)
