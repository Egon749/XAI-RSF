from pathlib import Path
from dataclasses import asdict
import argparse, datetime, io, json, shutil, subprocess, tarfile
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from crforest.partition import shared_quantile_cuts, tree_feature_subset
from experiments.post_st_leaf_shallow_v1.explain_company import LABELS
from .common import *


def prepare(run, repo):
    for name in ['input', 'evaluation_input', 'audit', 'reports', 'results', 'private']:
        (run/name).mkdir(exist_ok=False)
    sources = {}
    def copy(src, dst):
        sources[str(src)] = sha(src); dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
    reference = json.loads((repo/'CURRENT_POST_ST_MODEL.json').read_text())
    assert sha(repo/'CURRENT_POST_ST_MODEL.json') == REFERENCE_SHA
    old_protocol = repo/'experiments/post_st_drop9_v1/frozen_protocol.json'
    selection = json.loads(old_protocol.read_text()); sources[str(old_protocol)] = sha(old_protocol)
    assert selection['removed_names'] == DROP_NAMES
    current_hashes = pd.read_csv(repo/'experiments/post_st_2025_shallow_parent_v1/results/model_hashes.csv')
    sources[str(repo/'experiments/post_st_2025_shallow_parent_v1/results/model_hashes.csv')] = sha(repo/'experiments/post_st_2025_shallow_parent_v1/results/model_hashes.csv')
    audits = []; reuse = []; coverage = []; paired = []; maps = []; differences = []
    for year in YEARS:
        history = HISTORIES[year]; src = history/f'input/folds/{year}'; dest = run/f'input/folds/{year}'
        dest.mkdir(parents=True)
        plan = json.loads((src/'plan.json').read_text()); sources[str(src/'plan.json')] = sha(src/'plan.json')
        names = plan['feature_names']; keep = kept_columns(names)
        tr = arrays(src/'train_PRIVATE.npz'); ev = arrays(src/'eval_PRIVATE.npz')
        assert tr['X'].shape == ((309, 83) if year == 2023 else (390, 83))
        assert len(ev['X']) == (36 if year == 2023 else 53) and set(ev) == {'X', 'sample_ids'}
        assert tr['year'].max() == year-1 and len(np.unique(tr['groups'])) == len(tr['X'])
        assert int((tr['event'] == 1).sum()) == (14 if year == 2023 else 49)
        for stem, data in [('train', tr), ('eval', ev)]:
            copy(src/f'{stem}_PRIVATE.npz', dest/f'reference_{stem}_PRIVATE.npz')
            new = reduced(data, keep)
            for k in data:
                np.testing.assert_array_equal(new[k], data[k][:, keep] if k == 'X' else data[k])
            np.savez_compressed(dest/f'{stem}_PRIVATE.npz', **new)
        meta_path = history/f'evaluation_input/{year}_fit_metadata_PRIVATE.parquet'
        sources[str(meta_path)] = sha(meta_path); meta = pd.read_parquet(meta_path)
        np.testing.assert_array_equal(meta.sample_id, tr['sample_ids'])
        cutoff = pd.Timestamp(f'{year}-01-01')
        assert (meta.stop < cutoff).all() and (meta.origin < cutoff).all()
        for field in ['financial_available', 'market_available']:
            assert (meta[field].isna() | (meta[field] <= meta.origin)).all()
        assert (meta.entry_announcement < meta.origin).all()
        for stem in ['test_PRIVATE.npz', 'test_metadata_PRIVATE.parquet']:
            copy(history/f'evaluation_input/{year}_{stem}', run/f'evaluation_input/{year}_{stem}')
        # Labels copied outside the fit inputs, not loaded here.
        for seed, folder in sorted(references(year).items()):
            if year == 2023:
                expected = next(v for v in reference['saved_models'] if v['test_year'] == year and v['seed'] == seed)
            else:
                r = current_hashes[(current_hashes.model == 'xai_q2') & (current_hashes.seed == seed)].iloc[0]
                expected = dict(forest_sha256=r.forest_sha256, predictions_sha256=r.prediction_sha256,
                                functions_sha256=r.functions_sha256)
            for name, key in [('forest_PRIVATE.json', 'forest_sha256'), ('predictions_PRIVATE.npz', 'predictions_sha256')]:
                assert sha(folder/name) == expected[key]
                copy(folder/name, dest/f'reference/{seed}'/name)
            assert sha(folder/'functions_PRIVATE.npz') == expected['functions_sha256']
            sources[str(folder/'functions_PRIVATE.npz')] = expected['functions_sha256']
            f = CompetingRiskForest.load(folder/'forest_PRIVATE.json'); assert f.config == configuration(seed, 'shallow_parent')
            z = arrays(folder/'predictions_PRIVATE.npz')
            np.testing.assert_array_equal(z['sample_ids'], ev['sample_ids'])
            np.testing.assert_array_equal(f.predict(ev['X'], z['times']), z['cif'])
            np.testing.assert_array_equal(z['times'], TIMES)
            reuse.append(dict(year=year, seed=seed, path=str(folder),
                **{k:expected[k] for k in ['forest_sha256','predictions_sha256','functions_sha256']}))
            a, b = pairing(tr['groups'], f.config, 83), pairing(tr['groups'], f.config, 74)
            for i, (aa, bb) in enumerate(zip(a, b)):
                assert aa['bootstrap_sha256'] == bb['bootstrap_sha256'] and aa['multiplicity_sha256'] == bb['multiplicity_sha256']
                oldset = set(map(int, tree_feature_subset(83, 25, seed, i)))
                newset = {keep[int(j)] for j in tree_feature_subset(74, 25, seed, i)}
                paired.append(dict(year=year, seed=seed, tree=i, bootstrap_sha256=aa['bootstrap_sha256'],
                    multiplicity_sha256=aa['multiplicity_sha256'], old_eligible=25, new_eligible=25,
                    common_eligible=len(oldset & newset), old_feature_hash=aa['features_sha256'], new_feature_hash=bb['features_sha256']))
            for j, n in enumerate(names):
                coverage.append(dict(year=year, seed=seed, original_index=j, name=n, removed=n in DROP_NAMES,
                    reference_eligible=sum(j in tree_feature_subset(83, 25, seed, i) for i in range(32)),
                    reduced_eligible=sum(j in [keep[int(k)] for k in tree_feature_subset(74, 25, seed, i)] for i in range(32))))
        before, after = shared_quantile_cuts(tr['X'], 5), shared_quantile_cuts(tr['X'][:, keep], 5)
        for i, j in enumerate(keep): np.testing.assert_array_equal(before[j], after[i])
        fmap = [dict(original_index=j, reduced_index=keep.index(j) if j in keep else -1,
                     name=n, label=LABELS[n], removed=n in DROP_NAMES) for j, n in enumerate(names)]
        write(dest/'feature_map.json', fmap); maps.extend([dict(year=year, **r) for r in fmap])
        write(dest/'protocol.json', dict(year=year, kept_original_indices=keep, removed_names=DROP_NAMES,
            fit_information_before=f'{year}-01-01', configuration=asdict(configuration(42, 'shallow_parent'))))
        audits.append(dict(year=year, training_companies=len(tr['X']), training_target_events=int((tr['event'] == 1).sum()),
            training_recoveries=int((tr['event'] == 2).sum()), training_latest_entry_year=int(tr['year'].max()),
            training_latest_stop=str(meta.stop.max()), evaluation_companies=len(ev['X']),
            old_columns=83, new_columns=74, retained_values_equal=True, non_X_arrays_equal=True,
            retained_shared_cuts_equal=True, paired_bootstrap_trees=96, test_labels_in_pool_input=False))
        for k, v in asdict(configuration(42, 'shallow_parent')).items():
            differences.append(dict(year=year, parameter=k, reference=json.dumps(v), drop9=json.dumps(v), changed=False))
        differences.extend([dict(year=year, parameter=k, reference=a, drop9=b, changed=a != b)
                            for k, a, b in [('input_columns', '83', '74'), ('parent_AJ_strength', '10', '10'), ('case_time_weight', '1', '1')]])
    assert len(reuse) == 6
    protocol = dict(name=NAME, source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        years=YEARS, seeds=SEEDS, forest_fits=6, reused_reference_forests=6, derived_shrunk_models=6, pilot_fits=0,
        removed_names=DROP_NAMES, input_columns=[83, 74], configuration=asdict(configuration(42, 'shallow_parent')),
        shrinkage_strength=10, case_time_weight=1, max_training_followup_calendar_months=24,
        training_info_cutoffs={'2023':'2023-01-01', '2025':'2025-01-01'}, test_2025_observed_through='2026-06-30',
        selection_provenance=dict(prior_protocol_sha256=sha(old_protocol), diagnostic_information_before='2024-01-01',
            same_nine_fields_for_both_years_explicitly_selected_by_user=True, diagnostic_pre2023_redundant_fields=8,
            pre2023_extra_not_supported='total_mktcap_change_3m', no_new_feature_selection=True),
        primary={'2023':24, '2025':12}, secondary={'2023':[12], '2025':[24]},
        unsupported_horizon_policy='If no eligible company reaches horizon and some outcomes remain unknown, suppress AUC/AP/Brier/calibration/coverage; keep support counts and predicted risks only.',
        prediction_aggregation='Arithmetic mean of three seed predictions per company before evaluation; all seed outputs retained',
        metrics=['AUC','AP','IPCW Brier','Top20 captured and coverage','mean risk','observed incidence'],
        top20='ceil(.2*n), descending risk then sample_id; same cohort across arms; retrospective annual ranking',
        uncertainty=dict(repeats=1000, seed=2026110109, unit='company', paired=True,
            scopes='each supported year/horizon, termination; no pooling across horizons',
            interpretation='fixed-model conditional percentile95 intervals; undefined samples counted; no training/selection uncertainty'),
        coverage_caveat='83->74 changes balanced schedule blocks4->3 and eligible identities; same25/tree, mtry75 and bootstrap. Does not isolate removal from its induced coverage change.',
        validation='Every train/eval row, all native times and both causes; q2 reconstruction<=1e-10; reload and CIF constraints, bootstrap multiplicities, raw direct-parentAJ10 replay.',
        resources='6 independent single-thread CPU jobs through existing two-node Slurm/NFS pool;4GiB process cap; no calibration or search',
        keep_current_model=True, data_source_policy='unchanged conservative C; inherited disclosure-version limitations',
        status='FROZEN_BEFORE_NEW_FITS')
    public = repo/'experiments'/NAME/'frozen_protocol.json'
    if public.exists():
        saved = json.loads(public.read_text())
        assert {k:v for k,v in saved.items() if k!='source_commit'} == {k:v for k,v in protocol.items() if k!='source_commit'}
    else: write(public, protocol)
    write(run/'input/protocol.json', protocol); write(run/'audit/source_hashes.json', sources)
    write(run/'audit/data_alignment.json', dict(status='PASS', folds=audits, current_reference_sha256=REFERENCE_SHA))
    for name, rows in [('cohort_audit', audits), ('reused_artifacts', reuse), ('feature_map', maps),
                        ('feature_eligibility', coverage), ('tree_pairing', paired), ('configuration_diff', differences)]:
        pd.DataFrame(rows).to_csv(run/f'reports/{name}.csv', index=False)
    print(json.dumps(audits))


def freeze(run, repo):
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo).strip()
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit],cwd=repo))) as f:
        f.extractall(run/'input/code',filter='data')
    tasks = [dict(kind='cpu',argv=['python3',str(run/'input/code/experiments'/NAME/'launch.py'),'--year',str(y),'--seed',str(s)],
        cwd=str(run/'input/code'),timeout=3600,storage=dict(source=str(run/'input'),mode='nfs',results=str(run/'results/formal')))
        for y in YEARS for s in SEEDS]
    with (run/'formal.jsonl').open('x') as f:f.write(''.join(json.dumps(t)+'\n' for t in tasks))
    (run/'formal.py').write_text('# Six independent annual seed fits.\n')
    write(run/'formal.py.hetero.json',dict(independent=True,manifest='formal.jsonl'))
    write(run/'audit/freeze.json',dict(commit=commit,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),tasks=6,
        protocol_sha256=sha(run/'input/protocol.json'),manifest_sha256=sha(run/'formal.jsonl')))
    write(run/'audit/immutable_input_hashes.json',{str(p.relative_to(run/'input')):sha(p) for p in (run/'input').rglob('*') if p.is_file()})
    print(commit)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['prepare','freeze'],required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[2]
    (prepare if a.stage=='prepare' else freeze)(a.run,repo)
