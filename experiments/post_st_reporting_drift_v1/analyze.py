"""Read saved forecasts, verify them, and recompute reports. No fitting API.

Every invocation owns a new attempt directory. Original inputs are only read;
all hashes are checked again at completion. Full unit tests are a separate step.
"""
from pathlib import Path
from itertools import product
import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
import time
import traceback
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from crforest import CompetingRiskForest
from crforest.core import aalen_johansen
from experiments.post_st_prognosis_v1.common import horizon_predictions, calendar_elapsed
from experiments.post_st_feature_permutation_v1.common import evaluation_context
from experiments.post_st_ablation_v1.uncertainty import fast_metrics
from .metrics import stratified_metrics, paired_auc, paired_draws, alert_metrics, interval

YEARS = [2023, 2024, 2025]
ARMS = ['baseline', 'quadratic', 'support', 'quadratic_support']
VERSIONS = ['raw', 'shrunk']
SEEDS = [42, 101, 202]
TIMES = np.array([365., 366., 730., 731.])
METRIC_NAMES = ['within_year_auc', 'equal_year_auc', 'pooled_auc', 'within_minus_pooled']
REPO = Path(__file__).resolve().parents[2]
PUBLIC = Path(__file__).resolve().parent


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, obj):
    with Path(p).open('x') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False,
                  default=lambda v: v.item() if isinstance(v, np.generic) else str(v))
        f.write('\n')


class Reader:
    def __init__(self):
        self.hashes = {}

    def path(self, p):
        p = Path(p).resolve()
        h = digest(p)
        if str(p) in self.hashes:
            assert h == self.hashes[str(p)], f'Input changed: {p.name}'
        self.hashes[str(p)] = h
        return p

    def json(self, p):
        return json.loads(self.path(p).read_text())

    def arrays(self, p):
        with np.load(self.path(p), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def parquet(self, p):
        return pd.read_parquet(self.path(p))

    def finish(self):
        for p, h in self.hashes.items():
            assert digest(p) == h, f'Input mutated: {Path(p).name}'


def block_fit(*args, **kwargs):
    raise RuntimeError('READ_ONLY_ANALYSIS: forest fitting is forbidden')


def load_evidence(source, reader):
    CompetingRiskForest.fit = block_fit
    reader.path(REPO/'CURRENT_POST_ST_MODEL.json')
    for file in ['frozen_protocol.json', 'RESULTS_ZH.md', 'PAPER_METHODS_RESULTS_ZH.md', 'analyze.py']:
        reader.path(REPO/'experiments/post_st_cause_support_v1'/file)
    immutable = reader.json(source/'audit/immutable_input_hashes.json')
    # All frozen input bytes, including old reference model files, are preserved.
    for name, h in immutable.items():
        assert digest(reader.path(source/'input'/name)) == h
    originals = reader.json(source/'audit/source_hashes.json')
    labels, trains = {}, {}
    for year in YEARS:
        labels[year] = reader.arrays(source/f'evaluation_input/{year}_test_PRIVATE.npz')
        trains[year] = reader.arrays(source/f'input/folds/{year}/train_PRIVATE.npz')
        ev = reader.arrays(source/f'input/folds/{year}/eval_PRIVATE.npz')
        # Worker evaluation input intentionally has X/IDs only, no labels/groups.
        for key in ['X', 'sample_ids']:
            np.testing.assert_array_equal(ev[key], labels[year][key])
        assert len(np.unique(labels[year]['groups'])) == len(ev['X'])
        assert len(np.unique(trains[year]['groups'])) == len(trains[year]['X'])
        assert not set(trains[year]['groups']) & set(labels[year]['groups'])
        assert (trains[year]['year'] < year).all()
        label_hash = digest(source/f'evaluation_input/{year}_test_PRIVATE.npz')
        assert label_hash in originals.values()
    all_groups = np.concatenate([labels[y]['groups'] for y in YEARS])
    assert len(set(all_groups)) == len(all_groups)
    pred = {}
    registry = reader.json(source/'audit/registry_formal.json')
    assert len(registry) == 36
    for row in registry:
        path = Path(row['path'])
        hashes = reader.json(path/'hashes.json')
        for name, h in hashes.items():
            if name.endswith(('forest_PRIVATE.json', 'predictions_PRIVATE.npz')):
                assert digest(reader.path(path/name)) == h
        for version in VERSIONS:
            z = reader.arrays(path/version/'predictions_PRIVATE.npz')
            np.testing.assert_array_equal(z['sample_ids'], labels[row['year']]['sample_ids'])
            np.testing.assert_array_equal(z['times'], TIMES)
            assert np.isfinite(z['cif']).all()
            assert z['cif'].min() >= -1e-10 and z['cif'].sum(-1).max() <= 1+1e-10
            assert np.diff(z['cif'], axis=1).min() >= -1e-10
            pred[row['year'], row['arm'], row['seed'], version] = z['cif']
    saved = reader.arrays(source/'private/risks_PRIVATE.npz')
    risks = {}
    for y, a, v, h in product(YEARS, ARMS, VERSIONS, [12, 24]):
        mean = np.mean([pred[y, a, s, v] for s in SEEDS], axis=0)
        risks[y, a, v, h] = horizon_predictions(mean, TIMES, labels[y][f'h{h}'])
        np.testing.assert_array_equal(risks[y, a, v, h], saved[f'{y}|{a}|ensemble|{v}|{h}'])
    draws = paired_draws({y: len(labels[y]['X']) for y in YEARS})
    old = reader.arrays(source/'private/bootstrap_indices_PRIVATE.npz')
    for y in YEARS:
        np.testing.assert_array_equal(draws[y], old[str(y)])
    return labels, trains, risks, draws


def get_sample(labels, risks, years, arm, version, horizon, cause):
    contexts = [evaluation_context(labels[y], horizon, cause) for y in years]
    assert all(s and k.all() and np.all(w == 1) for _, w, k, s, _ in contexts)
    yy = np.concatenate([z[0] for z in contexts])
    p = np.concatenate([risks[y, arm, version, horizon][:, cause-1] for y in years])
    year = np.concatenate([np.full(len(labels[y]['X']), y) for y in years])
    ids = np.concatenate([labels[y]['sample_ids'] for y in years])
    return yy, p, year, ids


def acceptance_gate(labels, risks):
    expected = {12: [.7584, .7019, .7788, .7055], 24: [.7614, .7360, .7678, .7221]}
    checks = []
    for h in [12, 24]:
        for a, target in zip(ARMS, expected[h]):
            sample = get_sample(labels, risks, YEARS if h == 12 else YEARS[:2], a, 'shrunk', h, 1)
            m = stratified_metrics(*sample)
            assert round(m['within_year_auc'], 4) == target, 'STOP: specified AUC gate failed'
            yy, p, years, _ = sample
            good, pairs = 0., 0
            for y in np.unique(years):
                ix = years == y
                auc, n, numerator = paired_auc(yy[ix], p[ix])
                np.testing.assert_allclose(auc, roc_auc_score(yy[ix], p[ix]), atol=1e-14, rtol=0)
                good += numerator
                pairs += n
            np.testing.assert_allclose(good/pairs, m['within_year_auc'], atol=1e-14, rtol=0)
            checks.append(dict(months=h, arm=a, expected=target, actual=m['within_year_auc'], passed=True))
    return checks


def evaluate(source, out, reader):
    labels, trains, risks, draws = load_evidence(source, reader)
    write_json(out/'ACCEPTANCE_GATE.json', acceptance_gate(labels, risks))
    rows, intervals, oe, alerts, checks, zero_years = [], [], [], [], [], []
    # Save exact reused draws privately, never publish company sampling indices.
    np.savez_compressed(out/'bootstrap_indices_PRIVATE.npz', **{str(y): d for y, d in draws.items()})
    for h, cause, scope, version in product([12, 24], [1, 2], [*YEARS, 'pooled'], VERSIONS):
        years = (YEARS if h == 12 else YEARS[:2]) if scope == 'pooled' else [scope]
        header = dict(scope=str(scope), months=h, cause=cause, version=version,
                      years_included=','.join(map(str, years)))
        if scope == 2025 and h == 24:
            for arm in ARMS:
                rows.append(dict(**header, arm=arm, **{k: np.nan for k in METRIC_NAMES},
                                 supported=False, reason='NO_FOLLOWUP_SUPPORT_AT_HORIZON'))
                oe.append(dict(**header, arm=arm, observed=np.nan, mean_predicted=np.nan,
                               observed_expected=np.nan, reason='NO_FOLLOWUP_SUPPORT_AT_HORIZON'))
                alerts.append(dict(**header, arm=arm, checked=np.nan, captured=np.nan,
                                   precision=np.nan, lift=np.nan, reason='NO_FOLLOWUP_SUPPORT_AT_HORIZON'))
            continue
        samples = {a: get_sample(labels, risks, years, a, version, h, cause) for a in ARMS}
        actual = {}
        alert_actual = {}
        for arm, sample in samples.items():
            yy, p, year, ids = sample
            m = stratified_metrics(*sample)
            actual[arm] = np.array([m[k] for k in METRIC_NAMES])
            fm = fast_metrics(*sample)
            rows.append(dict(**header, arm=arm, **m, companies=len(yy), events=int(yy.sum()),
                             ap=fm[1], brier=fm[2], supported=True, reason=''))
            alert_actual[arm] = alert_metrics(*sample)
            for y in years:
                ix = year == y
                ordinary = roc_auc_score(yy[ix], p[ix]) if len(np.unique(yy[ix])) == 2 else np.nan
                counted, pairs, _ = paired_auc(yy[ix], p[ix])
                fast = fast_metrics(yy[ix], p[ix], year[ix], ids[ix])[0]
                np.testing.assert_allclose([counted, fast], [ordinary, ordinary], rtol=0, atol=1e-14, equal_nan=True)
                checks.append(dict(**header, arm=arm, year=y, sklearn_auc=ordinary,
                                   pair_count_auc=counted, fast_auc=fast, pairs=pairs, passed=True))
                if scope == 'pooled' and arm == 'baseline':
                    zero_years.append(dict(**header, year=y, events=int(yy[ix].sum()),
                                           nonevents=int((~yy[ix]).sum()), pairs=pairs, contributes=pairs > 0))
        boot = np.empty((1000, 4, len(METRIC_NAMES)))
        boe = np.empty((1000, 4))
        blift = np.empty((1000, 4))
        bprecision = np.empty((1000, 4))
        for b in range(1000):
            ix = np.concatenate([draws[y][b]+sum(len(labels[z]['X']) for z in years[:i])
                                 for i, y in enumerate(years)])
            for ai, a in enumerate(ARMS):
                sample = tuple(x[ix] for x in samples[a])
                m = stratified_metrics(*sample)
                boot[b, ai] = [m[k] for k in METRIC_NAMES]
                am = alert_metrics(*sample)
                boe[b, ai], blift[b, ai], bprecision[b, ai] = am['observed_expected'], am['lift'], am['precision']
        for ai, arm in enumerate(ARMS):
            am = alert_actual[arm]
            oe.append(dict(**header, arm=arm, companies=len(samples[arm][0]),
                           events=int(samples[arm][0].sum()), **{k: am[k] for k in ['observed', 'mean_predicted', 'observed_expected']},
                           **interval(boe[:, ai]), reason='ZERO_EVENT_BOOTSTRAP_DEGENERACY' if am['observed'] == 0 else ''))
            alerts.append(dict(**header, arm=arm, **am,
                               precision_lower95=interval(bprecision[:, ai])['lower95'],
                               precision_upper95=interval(bprecision[:, ai])['upper95'],
                               **{f'lift_{k}': v for k, v in interval(blift[:, ai]).items()},
                               reason='ZERO_OBSERVED_INCIDENCE_LIFT_UNDEFINED' if am['observed'] == 0 else ''))
            for j, metric in enumerate(METRIC_NAMES):
                intervals.append(dict(**header, contrast=arm, kind='estimate', metric=metric,
                                      estimate=actual[arm][j], **interval(boot[:, ai, j]), analysis_seed=2026110201))
                if arm != 'baseline':
                    intervals.append(dict(**header, contrast=arm+'_minus_baseline', kind='paired_difference', metric=metric,
                                          estimate=actual[arm][j]-actual['baseline'][j],
                                          **interval(boot[:, ai, j]-boot[:, 0, j]), analysis_seed=2026110201))
        # Interaction is a secondary descriptive contrast, as in the source analysis.
        for j, metric in enumerate(METRIC_NAMES):
            intervals.append(dict(**header, contrast='score_support_interaction', kind='paired_difference', metric=metric,
                                  estimate=actual[ARMS[3]][j]-actual[ARMS[1]][j]-actual[ARMS[2]][j]+actual[ARMS[0]][j],
                                  **interval(boot[:, 3, j]-boot[:, 1, j]-boot[:, 2, j]+boot[:, 0, j]), analysis_seed=2026110201))
        np.savez_compressed(out/f'bootstrap_{scope}_{h}_{cause}_{version}_PRIVATE.npz', auc=boot, oe=boe, lift=blift)
    for name, data in [('STRATIFIED_METRICS', rows), ('STRATIFIED_BOOTSTRAP_95', intervals),
                       ('OBSERVED_EXPECTED', oe), ('TOP20_LIFT', alerts),
                       ('INDEPENDENT_AUC_CHECKS', checks), ('YEAR_PAIR_SUPPORT', zero_years)]:
        pd.DataFrame(data).to_csv(out/f'{name}.csv', index=False)
    old = pd.read_csv(reader.path(source/'reports/METRICS_ENSEMBLE.csv'))
    result = pd.DataFrame(rows)
    for row in result[result.supported].itertuples():
        match = old[(old.scope.astype(str) == row.scope) & (old.arm == row.arm) &
                    (old.version == row.version) & (old.months == row.months) & (old.cause == row.cause)]
        assert len(match) == 1
        np.testing.assert_allclose([row.pooled_auc, row.ap, row.brier], match.iloc[0][['auc', 'ap', 'brier']].to_numpy(float), atol=1e-13, rtol=0, equal_nan=True)
    write_json(out/'metric_validation.json', dict(status='PASS', forest_fits=0, calibrator_fits=0,
               accepted_rounded_values=8, independent_auc_checks=len(checks), metric_rows=len(rows),
               all_supported_IPCW_weights_one=True, bootstrap_draws_exactly_reused=True,
               original_bootstrap_sha256=digest(source/'private/bootstrap_indices_PRIVATE.npz'),
               original_pooled_metrics_reproduced=True, seed=2026110201, repeats=1000))


def calendar_table(frame):
    """Risk time in [origin, first event/censor stop), split at calendar Jan1.

    Entry-day origin at pre-open; times in days, person-month = days/(365.25/12).
    Events on Jan1 are attributed to that year even when exposure there is zero.
    """
    records = []
    for r in frame.itertuples():
        for year in range(r.origin.year, r.stop.year+1):
            start, end = pd.Timestamp(f'{year}-01-01'), pd.Timestamp(f'{year+1}-01-01')
            left, right = max(r.origin, start), min(r.stop, end)
            days = max(0, (right-left).days)
            is_event = r.event > 0 and r.stop.year == year
            if not days and not is_event:
                continue
            records.append(dict(company=r.company, cohort=int(r.cohort), calendar_year=year,
                                person_days=days, termination=int(is_event and r.event == 1),
                                recovery=int(is_event and r.event == 2)))
    return pd.DataFrame(records)


def aggregate_calendar(rows, columns):
    table = rows.groupby(columns).agg(risk_companies=('company', 'nunique'),
                                      person_days=('person_days', 'sum'),
                                      termination_events=('termination', 'sum'),
                                      recovery_events=('recovery', 'sum')).reset_index()
    table['person_months'] = table.person_days/(365.25/12)
    for c in ['termination', 'recovery']:
        table[c+'_rate_per100_person_months'] = 100*table[c+'_events']/table.person_months
    return table


def suppress_calendar(frame):
    frame = frame.copy()
    for c in ['termination', 'recovery']:
        k = c+'_events'
        small = frame[k].between(1, 4)
        frame[k+'_status'] = np.where(small, 'SUPPRESSED_1_TO_4', 'REPORTED')
        frame.loc[small, [k, c+'_rate_per100_person_months']] = np.nan
    return frame


def drift(source, out, reader):
    labels, trains, risks, _ = load_evidence(source, reader)
    acceptance_gate(labels, risks)
    features = reader.json(source/'input/features.json')
    info = reader.parquet(source/'audit/information_alignment_PRIVATE.parquet').set_index('sample_id')
    meta = {y: reader.parquet(source/f'audit/{y}_training_metadata_PRIVATE.parquet') for y in YEARS}
    ajrows, nullrows, series = [], [], []
    for year in YEARS:
        tr = trains[year]
        np.testing.assert_array_equal(meta[year].sample_id.to_numpy(), tr['sample_ids'])
        np.testing.assert_array_equal(meta[year].event.to_numpy(), tr['event'])
        assert (meta[year].stop < pd.Timestamp(f'{year}-01-01')).all()
        calendar_t = np.array([calendar_elapsed(o, s) for o, s in zip(meta[year].origin, meta[year].stop)])
        np.testing.assert_allclose(calendar_t, tr['analysis_time'], atol=1e-12, rtol=0)
        # Descriptive nonparametric training AJ; explicitly requested null comparator,
        # no fitted forest/calibrator/learned parameters or test-label adjustment.
        aj = aalen_johansen(calendar_t, tr['event'], [12, 24], 2)
        for hi, h in enumerate([12, 24]):
            assert (calendar_t >= h).any()
            for cause in [1, 2]:
                row = dict(year=year, cutoff=f'{year}-01-01', months=h, cause=cause,
                           train_companies=len(calendar_t), training_observed_termination=int((tr['event'] == 1).sum()),
                           training_observed_recovery=int((tr['event'] == 2).sum()),
                           training_events_by_h=int(((tr['event'] == cause) & (calendar_t <= h)).sum()),
                           training_at_risk_h=int((calendar_t >= h).sum()), training_aj=float(aj[hi, cause-1]))
                ajrows.append(row)
                yy, w, known, supported, _ = evaluation_context(labels[year], h, cause)
                if supported:
                    assert known.all() and np.all(w == 1)
                null_brier = float(np.mean((aj[hi, cause-1]-yy)**2)) if supported else np.nan
                for a, v in product(ARMS, VERSIONS):
                    p = risks[year, a, v, h][:, cause-1]
                    brier = float(np.mean((p-yy)**2)) if supported else np.nan
                    nullrows.append(dict(**row, arm=a, version=v, supported=supported,
                                         null_brier=null_brier, forest_brier=brier,
                                         forest_minus_null_brier=brier-null_brier,
                                         reason='' if supported else 'NO_FOLLOWUP_SUPPORT_AT_HORIZON'))
                    series.append(dict(**row, arm=a, version=v, mean_predicted=float(p.mean()),
                                       observed=float(yy.mean()) if supported else np.nan, supported=supported))
    ajindex = {(r['year'], r['months'], r['cause']): r['training_aj'] for r in ajrows}
    for h, cause, a, v in product([12, 24], [1, 2], ARMS, VERSIONS):
        years = YEARS if h == 12 else YEARS[:2]
        yy, p, _, _ = get_sample(labels, risks, years, a, v, h, cause)
        constants = np.concatenate([np.full(len(labels[y]['X']), ajindex[y, h, cause]) for y in years])
        nb, fb = float(np.mean((constants-yy)**2)), float(np.mean((p-yy)**2))
        nullrows.append(dict(year='pooled', cutoff='year-specific pre-cutoff AJ constants', months=h,
                             cause=cause, arm=a, version=v, supported=True, training_aj=float(constants.mean()),
                             null_brier=nb, forest_brier=fb, forest_minus_null_brier=fb-nb,
                             reason='pooled Brier recomputed; AJ column is evaluation-weighted mean of annual constants'))
    pd.DataFrame(ajrows).to_csv(out/'TRAINING_AJ.csv', index=False)
    pd.DataFrame(nullrows).to_csv(out/'NULL_AJ_BASELINE.csv', index=False)
    pd.DataFrame(series).to_csv(out/'LEVEL_TIME_SERIES.csv', index=False)
    # One risk spell/company. Latest training snapshot for pre2023 entries;
    # already-frozen outer labels supersede cut-off training labels for2023-25.
    pieces = []
    old = trains[2025]
    take = old['year'] < 2023
    for data, mask in [(old, take), *[(labels[y], np.ones(len(labels[y]['X']), bool)) for y in YEARS]]:
        ids = data['sample_ids'][mask]
        origins = pd.to_datetime(info.loc[ids, 'origin']).reset_index(drop=True)
        times = data['time'][mask]
        assert np.array_equal(times, np.round(times))
        stops = origins + pd.to_timedelta(times, unit='D')
        elapsed = np.array([calendar_elapsed(o, s) for o, s in zip(origins, stops)])
        np.testing.assert_allclose(elapsed, data['analysis_time'][mask], atol=1e-12, rtol=0)
        pieces.append(pd.DataFrame(dict(sample_id=ids, company=data['groups'][mask], origin=origins,
                                       stop=stops, cohort=data['year'][mask], event=data['event'][mask])))
    frame = pd.concat(pieces, ignore_index=True)
    assert len(frame) == frame.company.nunique() == 443
    assert (frame.stop <= frame.origin+pd.DateOffset(months=24)).all()
    assert (frame.stop <= pd.Timestamp('2026-06-30')).all()
    frame.to_parquet(out/'risk_spells_PRIVATE.parquet', index=False)
    exposure = calendar_table(frame)
    assert exposure.termination.sum() == frame.event.eq(1).sum()
    assert exposure.recovery.sum() == frame.event.eq(2).sum()
    assert exposure.person_days.sum() == (frame.stop-frame.origin).dt.days.sum()
    annual = aggregate_calendar(exposure, ['calendar_year'])
    annual.to_csv(out/'calendar_annual_PRIVATE.csv', index=False)
    suppress_calendar(annual).to_csv(out/'CALENDAR_INCIDENCE.csv', index=False)
    exact = aggregate_calendar(exposure, ['cohort', 'calendar_year'])
    exact.to_csv(out/'cohort_calendar_PRIVATE.csv', index=False)
    # Full annual cross table is retained privately. Public output merges cohort
    # blocks and suppresses all cause cells in a calendar year when any is small,
    # avoiding complementary identification via the published annual margin.
    exposure['cohort_group'] = pd.cut(exposure.cohort, [2011, 2019, 2022, 2025], labels=['2012-2019', '2020-2022', '2023-2025']).astype(str)
    merged = aggregate_calendar(exposure, ['cohort_group', 'calendar_year'])
    for cause in ['termination', 'recovery']:
        k = cause+'_events'
        bad_years = merged.loc[merged[k].between(1, 4), 'calendar_year'].unique()
        mask = merged.calendar_year.isin(bad_years)
        merged[k+'_status'] = np.where(mask, 'SUPPRESSED_FOR_SMALL_OR_COMPLEMENTARY_CELL', 'REPORTED')
        merged.loc[mask, [k, cause+'_rate_per100_person_months']] = np.nan
    merged.to_csv(out/'COHORT_CALENDAR_CROSS.csv', index=False)
    cohortrows = []
    for cohort, g in frame.groupby('cohort'):
        t = np.array([calendar_elapsed(o, s) for o, s in zip(g.origin, g.stop)])
        e = g.event.to_numpy()
        aj = aalen_johansen(t, e, [12, 24], 2)
        for hi, h in enumerate([12, 24]):
            known = (t >= h) | ((e > 0) & (t <= h))
            supported = bool((t >= h).any() or known.all())
            for c in [1, 2]:
                count = int(((e == c) & (t <= h)).sum())
                cohortrows.append(dict(cohort=cohort, months=h, cause=c, companies=len(g),
                                       events=count if count == 0 or count >= 5 else np.nan,
                                       count_status='SUPPRESSED_1_TO_4' if 0 < count < 5 else 'REPORTED',
                                       horizon_supported=supported,
                                       aj=float(aj[hi, c-1]) if supported and (count == 0 or count >= 5) else np.nan,
                                       follows_complete=bool(known.all()), observation_clock='saved_latest_available_24m_cap'))
    pd.DataFrame(cohortrows).to_csv(out/'COHORT_AJ.csv', index=False)
    # Covariate summaries never include raw company values or true cutpoints.
    shifts = []
    for year in YEARS:
        for feature in features:
            j = feature['feature_id']
            a, b = trains[year]['X'][:, j], labels[year]['X'][:, j]
            aa, bb = a[np.isfinite(a)], b[np.isfinite(b)]
            sd = np.sqrt((np.var(aa, ddof=1)+np.var(bb, ddof=1))/2) if min(len(aa), len(bb)) >= 2 else np.nan
            difference = float(bb.mean()-aa.mean()) if len(aa) and len(bb) else np.nan
            smd = difference/sd if sd > 0 else (0. if sd == 0 and difference == 0 else np.nan)
            rare = False
            for z in [aa, bb]:
                vals, counts = np.unique(z, return_counts=True)
                if len(vals) <= 2 and len(vals) > 1 and counts.min() < 5:
                    rare = True
            shifts.append(dict(year=year, feature_id=j, feature_name=feature['feature_name'],
                               label=feature['label'], feature_group=feature['feature_group'], smd=smd,
                               abs_smd=abs(smd), train_missing=float(np.isnan(a).mean()),
                               eval_missing=float(np.isnan(b).mean()), missing_difference=float(np.isnan(b).mean()-np.isnan(a).mean()),
                               finite_train=len(aa), finite_eval=len(bb), eligible_public_top=min(len(aa), len(bb)) >= 10 and not rare))
    shifts = pd.DataFrame(shifts)
    shifts.to_csv(out/'all_covariate_shifts_PRIVATE.csv', index=False)
    summary = []
    for (y, group), g in shifts.groupby(['year', 'feature_group']):
        summary.append(dict(year=y, feature_group=group, features=len(g), defined_smd=int(g.smd.notna().sum()),
                            median_abs_smd=float(g.abs_smd.median()), max_abs_smd=float(g.abs_smd.max()),
                            above_abs_025=int(g.abs_smd.gt(.25).sum()), above_abs_05=int(g.abs_smd.gt(.5).sum()),
                            mean_train_missing=float(g.train_missing.mean()), mean_eval_missing=float(g.eval_missing.mean()),
                            mean_abs_missing_change=float(abs(g.missing_difference).mean())))
    pd.DataFrame(summary).to_csv(out/'COVARIATE_SHIFT_SUMMARY.csv', index=False)
    top = shifts[shifts.eligible_public_top].sort_values(['year', 'abs_smd'], ascending=[True, False]).groupby('year').head(5)
    top.drop(columns=['finite_train', 'finite_eval', 'eligible_public_top']).to_csv(out/'COVARIATE_SHIFT_TOP.csv', index=False)
    # Reuse existing reviewed terminal reasons, never infer from the warning type.
    receipt = reader.json(REPO/'experiments/post_st_rule_features_v1/execution_receipt.json')
    reason_root = Path(receipt['private_run'])/'direct_raw_analysis'
    reason_receipt = reader.json(reason_root/'reports/raw_audit_receipt.json')
    reason_path = reason_root/'private/termination_grounds_PRIVATE.parquet'
    assert digest(reader.path(reason_path)) == reason_receipt['artifact_sha256']['private/termination_grounds_PRIVATE.parquet']
    grounds = reader.parquet(reason_path).set_index('sample_id')
    oldids = meta[2025].set_index('sample_id').old_sample_id
    events = frame[frame.event.eq(1)].copy()
    events['old_sample_id'] = events.sample_id.map(oldids)
    events['ground'] = events.old_sample_id.map(grounds.ground).fillna('unavailable_existing_review')
    events['calendar_year'] = events.stop.dt.year
    events.to_parquet(out/'termination_reasons_join_PRIVATE.parquet', index=False)
    # Coarsen year/reason categories; exact old-code categories stay private.
    events['year_group'] = pd.cut(events.calendar_year, [2011, 2022, 2024, 2026], labels=['2012-2022', '2023-2024', '2025-2026H1']).astype(str)
    events['reason_group'] = np.where(events.ground.eq('price_below_one_yuan'), 'reviewed_price_below_one_yuan',
                                     np.where(events.ground.isin(['unavailable_existing_review', 'not_explicit_in_available_terminal_text']),
                                              'unavailable_or_not_explicit', 'reviewed_other_terminal_ground'))
    reasons = events.groupby(['year_group', 'reason_group']).size().rename('events').reset_index()
    small_periods = reasons.loc[reasons.events.between(1, 4), 'year_group'].unique()
    reasons['status'] = np.where(reasons.year_group.isin(small_periods), 'SUPPRESSED_FOR_SMALL_OR_COMPLEMENTARY_CELL', 'REPORTED')
    reasons.loc[reasons.year_group.isin(small_periods), 'events'] = np.nan
    reasons.to_csv(out/'DELISTING_REASON_SUMMARY.csv', index=False)
    write_json(out/'drift_validation.json', dict(status='PASS', unique_companies=443,
               risk_spells_reconcile=True, risk_days=int(exposure.person_days.sum()),
               termination_events=int(frame.event.eq(1).sum()), recovery_events=int(frame.event.eq(2).sum()),
               reason_known=int((~events.ground.isin(['unavailable_existing_review', 'not_explicit_in_available_terminal_text'])).sum()),
               reason_unavailable=int(events.ground.isin(['unavailable_existing_review', 'not_explicit_in_available_terminal_text']).sum()),
               forest_fits=0, calibrator_fits=0, descriptive_AJ_only=True,
               calendar_scope='fixed first-ST cohort; followup ends at first recovery/termination or censor/24m cap, not all listed-firm annual delistings',
               months_unit='actual days / (365.25/12)',
               calendar_limits='2026 through June30; entry cohorts stop2025; pre2023 labels from pre2025 snapshot; no new outcome acquisition',
               reason_limits='only previously reviewed45 terminal events; new/unmatched and2025 reasons unavailable; no new classification'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['metrics', 'drift'])
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--source', required=True, type=Path)
    args = p.parse_args()
    out = args.run/(args.stage+'_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(exist_ok=False)
    snapshot = out/'code_snapshot'
    snapshot.mkdir()
    for name in ['analyze.py', 'metrics.py', '__init__.py', 'analysis_protocol.json']:
        (snapshot/name).write_bytes((PUBLIC/name).read_bytes())
    reader = Reader()
    started = time.perf_counter()
    try:
        (evaluate if args.stage == 'metrics' else drift)(args.source, out, reader)
        reader.finish()
        write_json(out/'INPUT_HASHES_PRIVATE.json', reader.hashes)
        write_json(out/'stage_receipt.json', dict(status='PASS', stage=args.stage, forest_fits=0,
                   calibrator_fits=0, new_models=0, seconds=time.perf_counter()-started,
                   python=sys.version, platform=platform.platform(),
                   input_files=len(reader.hashes), input_hashes_sha256=digest(out/'INPUT_HASHES_PRIVATE.json'),
                   output_hashes={x.name: digest(x) for x in out.iterdir() if x.is_file()},
                   code_sha256=digest(Path(__file__))))
        print(json.dumps(dict(status='PASS', stage=args.stage, output=str(out), seconds=time.perf_counter()-started)))
    except Exception:
        (out/'FAILED.log').write_text(traceback.format_exc())
        write_json(out/'INPUT_HASHES_PRIVATE.json', reader.hashes)
        raise


if __name__ == '__main__':
    main()
