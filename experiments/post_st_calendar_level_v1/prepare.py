"""Verify the frozen reference and estimate calendar slopes from own snapshots."""
from pathlib import Path
import json, subprocess, traceback
import numpy as np
import pandas as pd
from crforest import CompetingRiskForest
from experiments.post_st_reporting_drift_v1.analyze import Reader, write_json, digest, block_fit
from experiments.post_st_reporting_drift_v1.metrics import paired_draws, interval
from experiments.post_st_prognosis_v1.common import horizon_predictions
from experiments.post_st_feature_permutation_v1.common import evaluation_context
from experiments.post_st_ablation_v1.verification import merge
from experiments.post_st_level_coverage_v1.affine import forest_envelope
from .rates import spells, contributions, independent_totals, requested_slope, certified_slopes, DAYS_PER_MONTH

PUBLIC = Path(__file__).resolve().parent
YEARS = [2023, 2024, 2025]
SEEDS = [42, 101, 202]


def main():
    CompetingRiskForest.fit = block_fit
    loc = json.loads((PUBLIC/'run_location.json').read_text())
    source, run = Path(loc['source_run']), Path(loc['private_run'])
    out = run/'prepared'; out.mkdir()
    reader = Reader()
    protocol = reader.json(PUBLIC/'frozen_protocol.json')
    frozen = subprocess.check_output(['git', 'show', 'bea1601877f7efe50227ea817946cede93997ad2:experiments/post_st_calendar_level_v1/frozen_protocol.json'])
    assert frozen == (PUBLIC/'frozen_protocol.json').read_bytes()
    reader.path(PUBLIC.parents[1]/'CURRENT_POST_ST_MODEL.json')
    registry = reader.json(source/'audit/registry_formal.json')
    originals = reader.json(source/'audit/source_hashes.json')
    saved = reader.arrays(source/'private/risks_PRIVATE.npz')
    old_draws = reader.arrays(source/'private/bootstrap_indices_PRIVATE.npz')
    labels, forecasts, trains, models = {}, {}, {}, {}
    gates = []
    for year in YEARS:
        labels[year] = reader.arrays(source/f'evaluation_input/{year}_test_PRIVATE.npz')
        assert digest(source/f'evaluation_input/{year}_test_PRIVATE.npz') in originals.values()
        lab = labels[year]
        trains[year] = reader.arrays(source/f'input/folds/{year}/train_PRIVATE.npz')
        ev = reader.arrays(source/f'input/folds/{year}/eval_PRIVATE.npz')
        for k in ['X', 'sample_ids']: np.testing.assert_array_equal(lab[k], ev[k])
        assert len(set(lab['groups'])) == len(lab['groups'])
        assert not set(lab['groups']) & set(trains[year]['groups'])
        rows = sorted([r for r in registry if r['year'] == year and r['arm'] == 'baseline'], key=lambda r: r['seed'])
        assert [r['seed'] for r in rows] == SEEDS
        ps, fs = [], []
        for row in rows:
            root = Path(row['path']); hashes = reader.json(root/'hashes.json')
            for name in ['shrunk/forest_PRIVATE.json', 'shrunk/predictions_PRIVATE.npz']:
                assert digest(reader.path(root/name)) == hashes[name]
            z = reader.arrays(root/'shrunk/predictions_PRIVATE.npz')
            np.testing.assert_array_equal(z['sample_ids'], lab['sample_ids'])
            ps.append(z['cif']); fs.append(CompetingRiskForest.load(root/'shrunk/forest_PRIVATE.json'))
        models[year] = merge(fs)
        for h in [12, 24]:
            p = horizon_predictions(np.mean(ps, axis=0), z['times'], lab[f'h{h}'])
            forecasts[year, h] = p
            np.testing.assert_array_equal(p, saved[f'{year}|baseline|ensemble|shrunk|{h}'])
            if year == 2025 and h == 24: continue
            y, w, known, supported, _ = evaluation_context(lab, h, 1)
            assert supported and known.all() and (w == 1).all()
            target = protocol['reference_oe_gate'][str(h)][YEARS.index(year)]
            actual = float(y.mean()/p[:, 0].mean())
            gates.append(dict(year=year, months=h, actual=actual, expected=target, passed=round(actual, 4) == target))
            write_json(out/f'gate_{year}_{h}.json', gates[-1])
            assert round(actual, 4) == target, 'STOP: reference O/E mismatch'
    write_json(out/'REFERENCE_GATE.json', gates)
    draws = paired_draws({y: len(labels[y]['X']) for y in YEARS})
    for y in YEARS: np.testing.assert_array_equal(draws[y], old_draws[str(y)])
    np.savez_compressed(out/'reference_risks_PRIVATE.npz', **{f'{y}_{h}':p for (y,h),p in forecasts.items()})
    np.savez_compressed(out/'bootstrap_indices_PRIVATE.npz', **{str(y):v for y,v in draws.items()})
    slope_rows, cert_rows, annual_rows, checks, specifications = [], [], [], [], {}
    for year in YEARS:
        cutoff = pd.Timestamp(f'{year}-01-01')
        meta = reader.parquet(source/f'audit/{year}_training_metadata_PRIVATE.parquet')
        tr = trains[year]
        np.testing.assert_array_equal(meta.sample_id.to_numpy().astype(str), tr['sample_ids'].astype(str))
        np.testing.assert_array_equal(meta.group.to_numpy().astype(str), tr['groups'].astype(str))
        np.testing.assert_array_equal(meta.event.to_numpy(), tr['event'])
        np.testing.assert_allclose(meta.time.to_numpy(), tr['time'], rtol=0, atol=0)
        assert (pd.to_datetime(meta.origin) < cutoff).all() and (pd.to_datetime(meta.stop) < cutoff).all()
        np.testing.assert_allclose((pd.to_datetime(meta.stop)-pd.to_datetime(meta.origin)).dt.days, tr['time'], atol=0, rtol=0)
        m = spells(meta, cutoff)
        starts = {'hist': m.origin.min(), '2y': cutoff-pd.DateOffset(years=2), '1y': cutoff-pd.DateOffset(years=1)}
        starts.update({str(v): pd.Timestamp(f'{v}-01-01') for v in range(year-5, year)})
        parts = {}
        for window, start in starts.items():
            end = cutoff if window in ['hist','2y','1y'] else start+pd.DateOffset(years=1)
            days, events = contributions(m, start, end); parts[window] = days, events
            d2, e2 = independent_totals(meta, cutoff, start, end)
            np.testing.assert_allclose(days.sum(), d2, rtol=0, atol=0)
            np.testing.assert_array_equal(events.sum(0), e2)
            checks.append(dict(year=year, window=window, person_days=days.sum(), events=events.sum(0).tolist(), independent_check=True))
            if window not in ['hist','2y','1y']:
                for k in [1,2]: annual_rows.append(dict(year=year, calendar_year=window, cause=k, person_months=days.sum()/DAYS_PER_MONTH, events=int(events[:,k-1].sum()), rate=events[:,k-1].sum()/(days.sum()/DAYS_PER_MONTH)))
        rng = np.random.default_rng(np.random.SeedSequence([2026110203, year]))
        boot = rng.integers(len(m), size=(1000, len(m)))
        np.savez_compressed(out/f'{year}_training_draws_PRIVATE.npz', indices=boot)
        requested = {}
        hd, he = parts['hist']
        for window in ['2y','1y']:
            wd, we = parts[window]
            requested[window] = []
            for k in [1,2]:
                point = requested_slope(hd, he[:,k-1], wd, we[:,k-1])
                draws_g = [requested_slope(hd,he[:,k-1],wd,we[:,k-1],ix) for ix in boot]
                ci = interval([r['requested'] for r in draws_g]); rawci = interval([r['raw_ratio'] for r in draws_g])
                requested[window].append(point['requested'])
                slope_rows.append(dict(year=year, window=window, cause=k, companies=len(m), historical_events=int(he[:,k-1].sum()), window_events=int(we[:,k-1].sum()), historical_person_months=hd.sum()/DAYS_PER_MONTH, window_person_months=wd.sum()/DAYS_PER_MONTH, **point, **ci, raw_lower95=rawci['lower95'],raw_upper95=rawci['upper95'],raw_undefined=rawci['undefined_repeats'],bootstrap_sparse_fallback=sum(r['sparse_fallback'] for r in draws_g)))
        f = models[year]
        assert f.config.path_budget == 2 and f.config.max_depth == 4 and f.n_features_ == 83 and len(f.trees_) == 96
        vertices = forest_envelope(f)
        np.savez_compressed(out/f'{year}_envelope_PRIVATE.npz', vertices=vertices)
        f.save(out/f'{year}_merged_PRIVATE.json')
        specs = {}
        for arm, req in [('reference',[1.,1.]),('recovery_only',[1.,requested['2y'][1]]),('both_2y',requested['2y']),('both_1y',requested['1y']),('uncertified_2y',requested['2y'])]:
            b, bound = (np.array(req), float(np.max(vertices@req))) if arm == 'uncertified_2y' else certified_slopes(vertices, req)
            if arm in ['reference','recovery_only']: np.testing.assert_array_equal(b, req)
            specs[arm] = dict(b=b.tolist(), c=[0.,0.], shape_grid=[0.], shape_values=[0.], diagnostic=arm=='uncertified_2y')
            cert_rows.append(dict(year=year,arm=arm,requested_termination=req[0],requested_recovery=req[1],certified_termination=b[0],certified_recovery=b[1],termination_reduction=req[0]-b[0],requested_bound=float(np.max(vertices@req)),actual_bound=bound,certificate_pass=bound<=1+1e-12,diagnostic=arm=='uncertified_2y',vertices=len(vertices)))
        annual = [r for r in annual_rows if r['year']==year and r['cause']==1]
        highest = max(annual, key=lambda r: r['rate'])
        hist_rate = he[:,0].sum()/(hd.sum()/DAYS_PER_MONTH)
        high_raw = highest['rate']/hist_rate
        high_requested = float(np.clip(high_raw, .5, 4.))
        b, bound = certified_slopes(vertices, [high_requested,requested['2y'][1]])
        specs['scenario_high'] = dict(b=b.tolist(),c=[0.,0.],shape_grid=[0.],shape_values=[0.],diagnostic=False)
        cert_rows.append(dict(year=year,arm='scenario_high',requested_termination=high_requested,requested_recovery=requested['2y'][1],certified_termination=b[0],certified_recovery=b[1],termination_reduction=high_requested-b[0],requested_bound=float(np.max(vertices@[high_requested,requested['2y'][1]])),actual_bound=bound,certificate_pass=True,diagnostic=False,vertices=len(vertices),high_raw_ratio=high_raw,high_calendar_year=highest['calendar_year']))
        specifications[str(year)] = specs
    pd.DataFrame(slope_rows).to_csv(out/'LEVEL_SLOPES_PRIVATE.csv', index=False)
    pd.DataFrame(annual_rows).to_csv(out/'ANNUAL_RATES_PRIVATE.csv', index=False)
    pd.DataFrame(cert_rows).to_csv(out/'CERTIFICATE_CHECKS.csv', index=False)
    write_json(out/'PERSON_TIME_CHECKS_PRIVATE.json', checks)
    write_json(out/'specifications.json', specifications)
    reader.finish(); write_json(out/'input_hashes_PRIVATE.json', reader.hashes)
    print(pd.DataFrame(slope_rows)[['year','window','cause','requested','sparse_fallback','lower95','upper95']].to_string(index=False))
    print(pd.DataFrame(cert_rows)[['year','arm','certified_termination','certified_recovery','actual_bound']].to_string(index=False))


if __name__ == '__main__':
    try: main()
    except Exception:
        loc=json.loads((PUBLIC/'run_location.json').read_text())
        (Path(loc['private_run'])/'PREPARE_FAILURE.txt').write_text(traceback.format_exc())
        raise
