"""Extend entry years and follow-up without changing any entry-time covariates."""
from pathlib import Path
import argparse
import datetime
import json
import shutil
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays, sha, write, classify_first

BASE = Path('/home/oliver/MainDiskData/xai_rsf_post_st_prognosis_v1_20261027_u6l00ebk')
FEASIBILITY = Path('/home/oliver/MainDiskData/xai_rsf_extend_2023_feasibility_7604vh8n/receipt.json')


def relabel(meta, endpoints, gaps, cutoff):
    """Only endpoint announcements strictly before cutoff may enter fitting."""
    cutoff = pd.Timestamp(cutoff)
    cap = cutoff - pd.Timedelta(days=1)
    end = endpoints[endpoints.announcement < cutoff]
    chain = gaps[gaps.announcement < cutoff]
    labels = []
    for row in meta.itertuples():
        candidates = end[end.group == row.group]
        pairs = [(r.announcement, 1 if r.type.endswith('X') else 2) for r in candidates.itertuples()]
        value = classify_first(row.origin, pairs, cap)
        history = chain[(chain.group == row.group) & (chain.announcement <= value['stop'])]
        value.update(label_eligible=not value['conflict'] and bool(history.resolved_for_first_A_X.all()),
                     chain_gap_present=bool(len(history)),
                     unresolved_chain_gap=not bool(history.resolved_for_first_A_X.all()),
                     effective_followup_cap=cap)
        labels.append(value)
    result = meta.copy().reset_index(drop=True)
    for name in labels[0]:
        result[name] = [v[name] for v in labels]
    assert (result.stop < cutoff).all()
    return result


def save_arrays(path, X, meta):
    np.savez_compressed(path, X=X, time=meta.time.to_numpy(float), event=meta.event.to_numpy(int),
        groups=meta.group.to_numpy(str), sample_ids=meta.sample_id.to_numpy(str),
        year=meta.entry_year.to_numpy(int), h12=meta.h12.to_numpy(int), h24=meta.h24.to_numpy(int),
        label_eligible=meta.label_eligible.to_numpy(bool), chain_gap_present=meta.chain_gap_present.to_numpy(bool))


def prepare(run):
    for name in ['input', 'audit', 'private', 'reports', 'results']:
        (run/name).mkdir(exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    protocol = Path(__file__).with_name('frozen_protocol.json')
    plan = json.loads(protocol.read_text())
    shutil.copy2(protocol, run/'audit/frozen_protocol.json')
    write(run/'audit/pre_fit_protocol_freeze.json', dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol_sha256=sha(protocol), statement='Frozen before new model fitting or predictions; prior 2024 results exposed.'))
    prior = json.loads((repo/'experiments/post_st_prognosis_v1/execution_receipt.json').read_text())
    source = {}
    for name, digest in prior['model_input_hashes'].items():
        path = BASE/'input'/name
        assert sha(path) == digest, name
        source[str(path)] = digest
    feasibility = json.loads(FEASIBILITY.read_text())
    for path, digest in feasibility['source_sha256'].items():
        assert sha(Path(path)) == digest, path
        source[path] = digest
    old = arrays(BASE/'input/train_PRIVATE.npz')
    evaluation = arrays(BASE/'input/eval_PRIVATE.npz')
    oldmeta = pd.read_parquet(BASE/'input/train_metadata_PRIVATE.parquet')
    evmeta = pd.read_parquet(BASE/'input/eval_metadata_PRIVATE.parquet')
    for a, d in [(old, oldmeta), (evaluation, evmeta)]:
        np.testing.assert_array_equal(a['sample_ids'], d.sample_id.to_numpy(str))
        np.testing.assert_array_equal(a['groups'], d.group.to_numpy(str))
    extra = evaluation['year'] == 2023
    test = evaluation['year'] == 2024
    assert extra.sum() == 36 and test.sum() == 45
    meta = pd.concat([oldmeta, evmeta.loc[extra]], ignore_index=True)
    X = np.vstack([old['X'], evaluation['X'][extra]])
    assert meta.group.is_unique and len(meta) == 345
    filters = [('announcement', '<', pd.Timestamp('2024-01-01')), ('group', 'in', meta.group.tolist())]
    end = pd.read_parquet(BASE/'private/endpoint_ledger_PRIVATE.parquet',
                         columns=['group', 'announcement', 'type'], filters=filters)
    gaps = pd.read_parquet(BASE/'private/chain_diagnostics_PRIVATE.parquet',
                          columns=['group', 'announcement', 'resolved_for_first_A_X'], filters=filters)
    assert end.type.str.endswith(('A', 'X')).all()
    reconstructed = relabel(oldmeta, end, gaps, '2023-01-01')
    np.testing.assert_array_equal(reconstructed.event, old['event'])
    np.testing.assert_array_equal(reconstructed.time, old['time'])
    train = relabel(meta, end, gaps, plan['fit_information_cutoff_exclusive'])
    train['split'] = 'train'
    assert train.label_eligible.all() and (train.time > 0).all()
    assert train.entry_year.max() == 2023 and (train.event == 1).sum() == 26 and (train.event == 2).sum() == 198
    assert train.reason.eq('complete_horizon').sum() == 77 and train.reason.eq('administrative_censoring').sum() == 44
    assert set(train.loc[train.entry_year == 2023, 'event']) == {0, 1}
    testmeta = evmeta.loc[test].reset_index(drop=True)
    assert not set(train.group) & set(testmeta.group)
    assert testmeta.label_eligible.all()
    for dd in [train, testmeta]:
        assert (dd.financial_available.isna() | (dd.financial_available < dd.origin)).all()
        assert (dd.market_available.isna() | (dd.market_available < dd.origin)).all()
        assert (dd.source_coverage_end >= dd.stop).all()
    for data, xx, role in [(train, X, 'train'), (testmeta, evaluation['X'][test], 'eval')]:
        save_arrays(run/'input'/f'{role}_PRIVATE.npz', xx, data)
        data.to_parquet(run/'input'/f'{role}_metadata_PRIVATE.parquet', index=False)
    # Compare every retained evaluation field, not just counts.
    final_eval = arrays(run/'input/eval_PRIVATE.npz')
    for name in evaluation:
        np.testing.assert_array_equal(final_eval[name], evaluation[name][test])
    feature = pd.read_csv(BASE/'reports/feature_manifest.csv')
    plan['feature_names'] = feature.name.tolist()
    assert X.shape == (345, 75) and len(plan['feature_names']) == 75
    feature['missing_rate_train'] = np.isnan(X).mean(0)
    feature.to_csv(run/'reports/feature_manifest.csv', index=False)
    counts = []
    for role, dd in [('train', train), ('entry_2023_subset', train[train.entry_year == 2023]), ('2024', testmeta)]:
        for months in [12, 24]:
            h = dd[f'h{months}']; known = (dd.time >= h) | ((dd.event > 0) & (dd.time <= h))
            d = (dd.event == 1) & (dd.time <= h); r = (dd.event == 2) & (dd.time <= h)
            counts.append(dict(split=role, horizon_months=months, companies=len(dd), termination_events=int(d.sum()),
                recovery_events=int(r.sum()), known_no_first_event=int((known & ~d & ~r).sum()), unknown_or_censored=int((~known).sum())))
    pd.DataFrame(counts).to_csv(run/'reports/cohort_counts.csv', index=False)
    pd.DataFrame(dict(entry_year=train.entry_year, termination=train.event.eq(1), recovery=train.event.eq(2),
                     event_zero=train.event.eq(0))).groupby('entry_year').agg(
        companies=('event_zero', 'size'), termination=('termination', 'sum'), recovery=('recovery', 'sum'),
        event_zero=('event_zero', 'sum')).to_csv(run/'reports/training_by_year.csv')
    alignment = dict(status=plan['status'], baseline_label_parity_companies=309, training_companies=345,
        evaluation_companies=45, training_events={'termination':26, 'recovery':198},
        complete_24m_no_event=77, administrative_censoring=44, entry_2023_new_terminations=4,
        prior_entries_new_terminations=8, training_information_cutoff_exclusive='2024-01-01',
        latest_training_endpoint_announcement=str(end.announcement.max()), latest_training_stop=str(train.stop.max()),
        all_entry_covariates_unchanged=True, evaluation_arrays_exact_subset=True, train_test_group_overlap=0,
        original_versions_certified=False, next_trading_day_origin_applied=False, cause_weights_changed=False,
        entry_2025_outcomes_used=False, source_sha256=source, counts=counts,
        note='Joint training-cohort/follow-up extension. Retains original announcement clock and inherited audit limitations.')
    write(run/'audit/data_alignment.json', alignment)
    plan['source_hashes'] = {p.name:sha(p) for p in (run/'input').iterdir() if p.is_file()}
    write(run/'input/plan.json', plan)
    assert all(sha(Path(p)) == digest for p, digest in source.items())
    print(json.dumps(dict(training_companies=345, termination=26, recovery=198, test_companies=45, counts=counts)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True)
    prepare(p.parse_args().run)
