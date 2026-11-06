"""Stage immutable C039 models and own-cutoff history, never outer labels."""
from pathlib import Path
import json
import os
import shutil

import numpy as np
import pandas as pd

from experiments.post_st_paper_features_v1.common import history
from experiments.post_st_prognosis_v1.common import arrays, sha, write

D = Path(__file__).resolve().parent
ROOT = D.parents[1]
YEARS = (2023, 2024, 2025)
SEEDS = (17, 503, 22)
TIMES = np.array([365., 366., 730., 731.])


def load(path):
    return json.loads(Path(path).read_text())


def archive_registry(source):
    """Read the six archived parent10 outputs without report-writing helpers."""
    book = {}
    for path in sorted((source / 'derive_published').glob('*/task-*/output')):
        summary = load(path / 'summary.json')
        if summary.get('strength') != 10:
            continue
        assert summary['status'] == 'COMPLETE' and summary['mode'] == 'derive'
        assert summary['forest_fits'] == 0 and summary['ensemble_mean_error'] < 1e-12
        year = int(summary['year'])
        assert year not in book
        hashes = load(path / 'hashes.json')
        for name in ['mean_predictions_PRIVATE.npz', 'ensemble/forest_PRIVATE.json',
                     'ensemble/functions_PRIVATE.npz']:
            assert sha(path / name) == hashes[name], (year, name)
        book[year] = (path, summary)
    assert set(book) == set(range(2020, 2026))
    return book


def calibration_arrays(outer, source, book, train, metadata):
    """Use complete historical cohorts and labels from this outer snapshot only.

    Unknown endpoints are an error here: the frozen source has fully supported
    cohorts. They are never converted to negatives or silently excluded.
    """
    forecasts, targets, ids, years, provenance = [], [], [], [], []
    paths = sorted((source / f'selection_input/{outer}/12').glob('*_labels_PRIVATE.npz'))
    assert paths
    origin = pd.to_datetime(metadata.origin)
    for path in paths:
        entry = int(path.name.split('_')[0])
        assert 2020 <= entry < outer and entry in book
        labels = arrays(path)
        mask = origin.dt.year.to_numpy() == entry
        assert mask.any()
        assert (origin[mask].map(lambda x: x + pd.DateOffset(months=12))
                < pd.Timestamp(outer, 1, 1)).all()
        assert set(labels) == set(train)
        for key in labels:
            np.testing.assert_array_equal(labels[key], train[key][mask])
        archived = book[entry][0] / 'mean_predictions_PRIVATE.npz'
        prediction = arrays(archived)
        np.testing.assert_array_equal(prediction['sample_ids'], labels['sample_ids'])
        np.testing.assert_array_equal(prediction['times'], TIMES)
        horizons = labels['h12']
        index = np.searchsorted(prediction['times'], horizons)
        np.testing.assert_array_equal(prediction['times'][index], horizons)
        p = prediction['cif'][np.arange(len(horizons)), index]
        event, time = labels['event'], labels['time']
        eligible = labels.get('label_eligible', np.ones(len(time), bool)).astype(bool)
        known = eligible & ((event > 0) | (time >= horizons)) & (time > 0)
        if not known.all():
            raise ValueError(f'{outer}/{entry}: unknown historical endpoints; no zero-filling permitted')
        y = np.column_stack([(event == k) & (time <= horizons) for k in (1, 2)])
        assert np.all(y.sum(axis=1) <= 1) and np.isfinite(p).all()
        assert (p >= -1e-12).all() and (p.sum(axis=1) <= 1 + 1e-12).all()
        forecasts.append(p)
        targets.append(y.astype(float))
        ids.append(labels['sample_ids'])
        years.append(np.full(len(time), entry, dtype=int))
        provenance.append(dict(entry_year=entry, companies=len(time),
            termination=int(y[:, 0].sum()), recovery=int(y[:, 1].sum()), unknown=0,
            labels_sha256=sha(path), predictions_sha256=sha(archived)))
    result = dict(p=np.concatenate(forecasts), y=np.concatenate(targets),
                  sample_ids=np.concatenate(ids), entry_year=np.concatenate(years))
    assert len(set(result['sample_ids'])) == len(result['sample_ids'])
    return result, provenance


def main():
    destination = Path(os.environ['REPAIR_RUN']).resolve()
    assert destination.is_dir(), 'Create a unique private run before staging'
    inp = destination / 'input'
    inp.mkdir(exist_ok=False)
    source = Path(load(D.parent / 'post_st_shrinkage_strength_v1/run_location.json')['private_run'])
    member_source = Path(load(D.parent / 'post_st_seed_explanation_stability_v1/run_location.json')['private_run'])
    pointer_path = ROOT / 'CURRENT_POST_ST_MODEL.json'
    pointer = load(pointer_path)
    assert pointer['arm'] == 'C039' and pointer['version'] == 'shrunk'
    assert pointer['configuration']['seeds'] == list(SEEDS)
    assert pointer['configuration']['n_inputs'] == 74
    protocol_hash = sha(D / 'FROZEN_PROTOCOL.json')
    book = archive_registry(source)
    hashes, audits, full_support = {}, [], []

    def copy(src, dst):
        assert not dst.exists()
        dst.parent.mkdir(parents=True, exist_ok=True)
        before = sha(src)
        shutil.copyfile(src, dst)
        assert before == sha(src) == sha(dst)
        hashes[str(dst.relative_to(destination))] = before

    for year in YEARS:
        out = inp / str(year)
        train_src = source / f'input/folds/{year}/train_PRIVATE.npz'
        eval_src = source / f'input/folds/{year}/eval_PRIVATE.npz'
        train, ev = arrays(train_src), arrays(eval_src)
        assert set(ev) == {'X', 'sample_ids'}
        assert train['X'].shape[1] == ev['X'].shape[1] == 74
        assert not set(train['sample_ids']) & set(ev['sample_ids'])
        metadata_src = history(year) / f'evaluation_input/{year}_fit_metadata_PRIVATE.parquet'
        metadata = pd.read_parquet(metadata_src)
        np.testing.assert_array_equal(metadata.sample_id.to_numpy(), train['sample_ids'])
        np.testing.assert_array_equal(metadata.event.to_numpy(), train['event'])
        np.testing.assert_allclose(metadata.time.to_numpy(), train['time'], rtol=0, atol=1e-12)
        origin, stop = pd.to_datetime(metadata.origin), pd.to_datetime(metadata.stop)
        assert (origin < pd.Timestamp(year, 1, 1)).all()
        assert (stop < pd.Timestamp(year, 1, 1)).all()
        assert (stop >= origin).all()
        assert (stop <= origin.map(lambda x: x + pd.DateOffset(months=24))).all()
        assert len(set(train['groups'])) == len(train['groups'])
        for src, name in [(train_src, 'train_PRIVATE.npz'), (eval_src, 'eval_PRIVATE.npz'),
                          (metadata_src, 'metadata_PRIVATE.parquet')]:
            copy(src, out / name)
        saved, summary = book[year]
        assert summary['source_train_sha256'] == sha(train_src)
        member_predictions = []
        for seed in SEEDS:
            member = member_source / f'input/{year}/{seed}/shrunk'
            expected = next(r for r in pointer['saved_models']
                            if r['seed'] == seed and r['test_year'] == year)
            for name, key in [('forest_PRIVATE.json', 'forest_sha256'),
                              ('functions_PRIVATE.npz', 'functions_sha256'),
                              ('predictions_PRIVATE.npz', 'predictions_sha256')]:
                assert sha(member / name) == expected[key]
                hashes[f'CURRENT_source/{year}/{seed}/{name}'] = expected[key]
            np.testing.assert_equal(load(member / 'forest_PRIVATE.json')['trees'],
                                    load(saved / str(seed) / 'forest_PRIVATE.json')['trees'])
            z = arrays(member / 'predictions_PRIVATE.npz')
            np.testing.assert_array_equal(z['sample_ids'], ev['sample_ids'])
            np.testing.assert_array_equal(z['times'], TIMES)
            member_predictions.append(z['cif'])
        mean = arrays(saved / 'mean_predictions_PRIVATE.npz')
        np.testing.assert_array_equal(mean['sample_ids'], ev['sample_ids'])
        np.testing.assert_array_equal(mean['times'], TIMES)
        np.testing.assert_array_equal(mean['cif'], np.mean(member_predictions, axis=0))
        forest = load(saved / 'ensemble/forest_PRIVATE.json')
        assert len(forest['trees']) == 96 and forest['n_features'] == 74
        for src, name in [(saved / 'ensemble/forest_PRIVATE.json', 'forest_PRIVATE.json'),
                          (saved / 'ensemble/functions_PRIVATE.npz', 'functions_PRIVATE.npz'),
                          (saved / 'mean_predictions_PRIVATE.npz', 'reference_predictions_PRIVATE.npz')]:
            copy(src, out / name)
        cal, cohorts = calibration_arrays(year, source, book, train, metadata)
        np.savez_compressed(out / 'calibration_PRIVATE.npz', **cal)
        hashes[str((out / 'calibration_PRIVATE.npz').relative_to(destination))] = sha(out / 'calibration_PRIVATE.npz')
        full_support.append(dict(outer_cutoff=year, cohorts=cohorts))
        count = lambda n: int(n) if n == 0 or n >= 5 else '<5'
        audits.append(dict(year=year, training_companies=len(train['time']),
            training_termination=int((train['event'] == 1).sum()),
            training_recovery=int((train['event'] == 2).sum()), evaluation_companies=len(ev['X']),
            features=74, trees=96, native_times=len(forest['grid']),
            calibration_entry_years=sorted(set(map(int, cal['entry_year']))),
            calibration_companies=len(cal['p']),
            calibration_termination=count(cal['y'][:, 0].sum()),
            calibration_recovery=count(cal['y'][:, 1].sum()), calibration_unknown=0,
            reference_prediction_error=0., own_cutoff_labels_exact=True))
    write(destination / 'DATA_PROVENANCE_PRIVATE.json', dict(
        source=str(source), members=str(member_source), historical_support=full_support,
        hashes=hashes, protocol_sha256=protocol_hash, outer_evaluation_labels_read=False))
    write(D / 'DATA_AUDIT.json', dict(years=audits, input_hashes=hashes,
        protocol_sha256=protocol_hash, current_pointer_sha256=sha(pointer_path),
        forest_fits=0, outer_evaluation_labels_read=False,
        unknown_outcomes_zero_filled=False, historical_sample_out_predictions=True,
        small_event_cells='Counts 1-4 suppressed as <5; company-level data stay private.'))
    print(json.dumps(dict(staged_years=list(YEARS), outer_labels_read=False,
                          reference_predictions_identical=True)))


if __name__ == '__main__':
    main()
