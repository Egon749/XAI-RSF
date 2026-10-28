"""Fixed-forecast error attribution. No fitting, calibration or model selection."""
from pathlib import Path
import argparse
import json
import re
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from crforest import CIFExpansion
from crforest.partition import tree_feature_subset
from experiments.post_st_prognosis_v1.common import arrays, sha, write, horizon_predictions
from experiments.post_st_cause_weights_h24_v1.reference import forecasts
from experiments.post_st_train2024_h12_v1.evaluate import metric
from experiments.post_st_train2024_h12_v1.validate import direct_metrics
from experiments.post_st_rule_features_v1.inspect_raw import terminal_ground


def period(date):
    """Announcement calendar bands, NEVER a treatment or applicable-rule label."""
    date = pd.Timestamp(date)
    for boundary, label in [('2024-04-12', 'before_20240412'),
                            ('2024-04-30', '20240412_to_20240429'),
                            ('2024-10-30', '20240430_to_20241029')]:
        if date < pd.Timestamp(boundary):
            return label
    return 'from_20241030'


def rule_evidence(ground, text):
    """Describe the explicit trigger, not the counterfactual policy effect."""
    t = re.sub(r'\s+', '', text)
    versions = sorted(set(re.findall(r'上市规则[（(]([^）)]{0,35})[）)]', t)))
    caps = sorted(set(re.findall(r'(?:收盘总市值|收盘市值)均低于(?:人民币)?([35])亿元', t)))
    if ground == 'price_below_one_yuan':
        label = 'preexisting_one_yuan_trigger'
    elif ground == 'market_cap_threshold':
        if caps == ['3']:
            label = 'explicit_3e8_market_cap_trigger'
        elif caps == ['5']:
            label = 'explicit_5e8_market_cap_trigger'
        else:
            raise ValueError('Ambiguous market-cap trigger requires review')
    elif ground in ['financial_or_audit_ground', 'failure_to_file_annual_report']:
        if versions and all('2023' in x for x in versions):
            label = 'explicit_2023_version_financial_audit_or_filing'
        else:
            label = 'other_financial_audit_or_filing_version_increment_unidentified'
    elif ground == 'voluntary':
        label = 'voluntary'
    else:
        label = 'unresolved_terminal_ground'
    return label, ';'.join(versions), ';'.join(caps)


def brier_parts(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    total = float(np.mean((p-y)**2))
    bias = float((p.mean()-y.mean())**2)
    centered = float(np.mean(((p-p.mean())-(y-y.mean()))**2))
    assert abs(total-bias-centered) < 1e-12
    return dict(brier=total, squared_mean_bias=bias, centered_residual=centered,
                mean_bias_fraction_of_brier=bias/total if total else 0.,
                event_case_loss=float(np.mean(((p-y)**2)*y)),
                other_outcome_loss=float(np.mean(((p-y)**2)*(1-y))))


def case_control_auc(event, p, case_mask):
    """Other termination causes are excluded, not turned into controls."""
    event, p, case_mask = np.asarray(event), np.asarray(p), np.asarray(case_mask, bool)
    if np.any(case_mask & (event != 1)):
        raise ValueError('Case mask includes nontermination outcomes')
    controls = event != 1
    a, b = p[case_mask], p[controls]
    if not len(a) or not len(b):
        return np.nan
    return float(np.mean((a[:, None] > b) + .5*(a[:, None] == b)))


def annual_selection(meta, risk, fraction):
    selected = np.zeros(len(meta), bool)
    for year in sorted(meta.entry_year.unique()):
        ix = np.flatnonzero(meta.entry_year.to_numpy() == year)
        order = np.lexsort((meta.sample_id.to_numpy(str)[ix], -risk[ix]))
        selected[ix[order[:int(np.ceil(len(ix)*fraction))]]] = True
    return selected


def analyze(run):
    started = time.perf_counter()
    protocol = json.loads((run/'audit/protocol.json').read_text())
    source = Path(protocol['source_run'])
    raw = Path(protocol['raw_audit_run'])
    out, private = run/'results', run/'private'
    assert not list(out.iterdir()) and not list(private.iterdir())
    sources = {}
    def register(p):
        p = Path(p)
        sources[str(p)] = sha(p)
        return p
    def read_json(p):
        return json.loads(register(p).read_text())
    def read_arrays(p):
        return arrays(register(p))
    def read_frame(p):
        return pd.read_parquet(register(p))
    inherited = read_json(source/'audit/protocol.json')
    for name, digest in read_json(source/'audit/immutable_input_hashes.json').items():
        assert sha(register(source/'input'/name)) == digest
    raw_receipt = read_json(raw/'direct_raw_analysis/reports/raw_audit_receipt.json')
    for name, digest in raw_receipt['artifact_sha256'].items():
        assert sha(register(raw/'direct_raw_analysis'/name)) == digest
    reference, reference_paths, _ = forecasts(inherited)
    for folder in reference_paths.values():
        for name, digest in read_json(folder/'hashes.json').items():
            assert sha(register(folder/name)) == digest
    manifest = pd.read_csv(register(Path(inherited['reference_runs']['2023'])/'reports/feature_manifest.csv'))
    names = manifest.name.tolist()
    assert len(names) == 75 and len(set(names)) == 75
    price_index = names.index('log_price_last')
    tr, ev, tm, em = {}, {}, {}, {}
    for year in protocol['test_years']:
        tr[year] = read_arrays(source/'input/folds'/str(year)/'equal/train_PRIVATE.npz')
        ev[year] = read_arrays(source/'evaluation_input'/str(year)/'test_PRIVATE.npz')
        tm[year] = read_frame(source/'evaluation_input'/str(year)/'train_metadata_PRIVATE.parquet')
        em[year] = read_frame(source/'evaluation_input'/str(year)/'test_metadata_PRIVATE.parquet')
        for data, meta in [(tr[year], tm[year]), (ev[year], em[year])]:
            np.testing.assert_array_equal(data['sample_ids'], meta.sample_id.to_numpy(str))
            np.testing.assert_array_equal(data['event'], meta.event)
            assert meta.sample_id.is_unique and meta.group.is_unique
        assert ((ev[year]['event'] > 0) | (ev[year]['analysis_time'] >= 24)).all()
    meta = pd.concat([em[y] for y in protocol['test_years']], ignore_index=True)
    assert len(meta) == 81 and meta.sample_id.is_unique and meta.group.is_unique
    all_event = meta.event.to_numpy()
    grounds = read_frame(raw/'direct_raw_analysis/private/termination_grounds_PRIVATE.parquet')
    entry = read_frame(raw/'direct_raw_analysis/private/entry_raw_fields_PRIVATE.parquet')
    clock = read_frame(raw/'direct_raw_analysis/private/annual_clock_PRIVATE.parquet')
    market = read_frame(raw/'direct_raw_analysis/private/raw_daily_origin_alignment_PRIVATE.parquet')
    meta = meta.merge(grounds[['sample_id','ground','source_ids','source_text']], on='sample_id', how='left', validate='one_to_one', sort=False)
    assert meta.loc[meta.event.eq(1), 'ground'].notna().all()
    assert meta.loc[meta.event.ne(1), 'ground'].isna().all()
    meta = meta.merge(entry.drop(columns='scope'), on='sample_id', validate='one_to_one', sort=False)
    meta = meta.merge(clock[['sample_id','annual_ann','annual_available','raw_annual_announcement']], on='sample_id', validate='one_to_one', sort=False)
    meta = meta.merge(market[['sample_id','Clsprc','market_cap_yuan','Markettype']], on='sample_id', validate='one_to_one', sort=False)
    x = np.concatenate([ev[y]['X'] for y in protocol['test_years']])
    np.testing.assert_allclose(np.exp(x[:,price_index]), meta.Clsprc, rtol=1e-12)
    np.testing.assert_allclose(np.exp(x[:,names.index('log_total_mktcap_last')])*1000, meta.market_cap_yuan, rtol=1e-12)
    meta['rule_evidence'] = ''
    meta['rule_versions'] = ''
    meta['explicit_cap_threshold_units_1e8'] = ''
    for i in meta.index[meta.event.eq(1)]:
        row = meta.loc[i]
        assert terminal_ground(row.source_text) == row.ground
        meta.loc[i,['rule_evidence','rule_versions','explicit_cap_threshold_units_1e8']] = rule_evidence(row.ground,row.source_text)
    meta['terminal_period'] = meta.stop.map(period)
    meta['terminal_year'] = meta.stop.dt.year
    meta['entry_warning'] = np.where(meta.entry_type.eq('AD'), 'STAR_ST', 'ST')
    meta['annual_version_later_than_origin'] = meta.annual_available.gt(meta.origin)
    meta['annual_notice_same_day'] = meta.annual_ann.eq(meta.origin)
    meta['annual_notice_strictly_before'] = meta.annual_ann.lt(meta.origin)
    meta['annual_report_used'] = meta.report_date.dt.month.eq(12) & meta.report_date.dt.year.eq(meta.entry_year-1)
    meta['day_365_plus_event'] = meta.time > meta.h12
    meta.to_parquet(private/'aligned_cases_PRIVATE.parquet', index=False)

    # Recover additional training terminations at the 2024 cutoff from the same
    # authoritative event ledger. Terminal texts never enter prediction inputs.
    history = pd.read_csv(register('/home/oliver/LLMworkspace/PhD_STARST_Prognosis/runs/phase1_data_freeze_v1/03_event_history.csv'), dtype=str, keep_default_na=False)
    mapping = read_json('/home/oliver/MainDiskData/first_st_followup_audit_20261025_v1/assembly/group_mapping.json')
    history['group'] = history.Symbol.map(lambda z: mapping.get(z,z))
    history['announcement'] = pd.to_datetime(history.adjudicated_announcement_date.where(history.adjudicated_announcement_date.ne(''),history.Annoudt))
    historical = []
    for year in protocol['test_years']:
        for row in tm[year][tm[year].event.eq(1)].itertuples():
            selected = history[history.group.eq(row.group) & history.announcement.gt(row.origin) & history.announcement.le(row.stop) & history.Chgtype.str.endswith(('T','X'))]
            assert len(selected) and selected.announcement.lt(pd.Timestamp(year,1,1)).all()
            text = '\n'.join(selected.Chgtype+':'+selected.Chgrsdis+':'+selected.Content+selected.supplemental_content)
            ground = terminal_ground(text)
            previous = grounds[grounds.sample_id.eq(row.sample_id)]
            if len(previous):
                assert previous.iloc[0].ground == ground
            historical.append(dict(test_year=year,sample_id=row.sample_id,ground=ground,terminal_year=row.stop.year,
                                   terminal_period=period(row.stop),source_ids=';'.join(selected.event_id),source_text=text))
    historical = pd.DataFrame(historical)
    historical.to_parquet(private/'training_terminal_evidence_PRIVATE.parquet',index=False)
    historical.groupby(['test_year','ground']).size().rename('termination_events').reset_index().to_csv(out/'training_termination_grounds.csv',index=False)

    # Load every fixed forest prediction, preserving seed identity and hashes.
    predictions, folders = {}, {}
    formal_job = read_json(source/'audit/formal_gate.json')['job']
    tasks = [json.loads(line) for line in register(source/'formal.jsonl').read_text().splitlines()]
    for task_id, task in enumerate(tasks):
        args = task['argv']
        year = int(args[args.index('--year')+1])
        variant = args[args.index('--variant')+1]
        seed = str(args[args.index('--seed')+1])
        folder = source/'results/formal'/str(formal_job)/f'task-{task_id}'/'output'
        for name,digest in read_json(folder/'hashes.json').items():
            assert sha(register(folder/name)) == digest
        p = read_arrays(folder/'predictions_PRIVATE.npz')
        predictions[(year,variant,seed)] = p
        folders[(year,variant,seed)] = folder
    for (year,model,seed), p in reference.items():
        if model in ['unrestricted','standard_rsf']:
            predictions[(year,model,seed)] = p
    assert len(predictions) == 30
    for year in protocol['test_years']:
        for model in protocol['models']:
            pieces = [predictions[(year,model,str(s))] for s in protocol['seeds']]
            predictions[(year,model,'ensemble')] = {
                'sample_ids':pieces[0]['sample_ids'], 'times':pieces[0]['times'],
                'cif':np.mean([p['cif'] for p in pieces],axis=0),
                'training_cif':np.mean([p['training_cif'] for p in pieces],axis=0)}
    risks, train_metrics, replay, brier, errors, subgroups, support_summary, tree_audit = {}, [], [], [], [], [], [], []
    support_by_key = {}
    for (year,model,seed), p in predictions.items():
        np.testing.assert_array_equal(p['sample_ids'], ev[year]['sample_ids'])
        risks[(year,model,seed)] = horizon_predictions(p['cif'],p['times'],ev[year]['h24'])
        trainrisk = horizon_predictions(p['training_cif'],p['times'],tr[year]['h24'])
        for k,cause in [(1,'termination'),(2,'recovery')]:
            train_result,_,_,_ = metric(tr[year]['analysis_time'],tr[year]['event'],24.,trainrisk[:,k-1],k,tr[year]['label_eligible'])
            train_metrics.append(dict(test_year=year,model=model,seed=seed,cause=cause,scope='training_apparent_not_validation',**train_result))
            result,w,y,known = metric(ev[year]['analysis_time'],ev[year]['event'],24.,risks[(year,model,seed)][:,k-1],k,ev[year]['label_eligible'])
            assert known.all() and (w==1).all()
            np.testing.assert_allclose([result[z] for z in ['auc','ap','brier']],direct_metrics(y,risks[(year,model,seed)][:,k-1]),atol=1e-12,rtol=0)
            replay.append(dict(scope=str(year),model=model,seed=seed,cause=cause,**result))
        if (year,model,seed) not in folders:
            continue
        folder = folders[(year,model,seed)]
        f = read_json(folder/'forest_PRIVATE.json')
        per_tree_risk, zero_support, target_support = [], [], []
        for tid,nodes in enumerate(f['trees']):
            eligible = price_index in tree_feature_subset(f['n_features'],f['config']['tree_feature_count'],f['config']['seed'],tid)
            used = any(node.get('feature')==price_index for node in nodes)
            assert not used or eligible
            tree_audit.append(dict(test_year=year,model=model,seed=seed,tree=tid,price_eligible=eligible,price_used=used))
            values, supports = [], []
            for i,row in enumerate(ev[year]['X']):
                at = 0
                while 'cif' not in nodes[at]:
                    node=nodes[at];v=row[node['feature']]
                    left = v <= node['threshold'] or (np.isnan(v) and node['missing_left'])
                    at = node['left' if left else 'right']
                node=nodes[at];idx=np.searchsorted(f['grid'],ev[year]['h24'][i],side='right')-1
                values.append(node['cif'][idx] if idx>=0 else [0.,0.])
                supports.append(node['cause_company_support'][0])
            per_tree_risk.append(values);target_support.append(supports);zero_support.append(np.asarray(supports)==0)
        exact = float(np.max(np.abs(np.mean(per_tree_risk,axis=0)-risks[(year,model,seed)])))
        explanation = CIFExpansion.load(folder/'functions_PRIVATE.npz')
        rebuilt = horizon_predictions(explanation.predict(ev[year]['X'],p['times']),p['times'],ev[year]['h24'])
        reconstruction = float(np.max(abs(rebuilt-risks[(year,model,seed)])))
        assert exact < 1e-12 and reconstruction < 1e-10
        support_by_key[(year,model,seed)] = np.asarray(zero_support).mean(axis=0)
        support_summary.append(dict(test_year=year,model=model,seed=seed,
            trees=len(f['trees']),price_eligible_trees=sum(z['price_eligible'] for z in tree_audit[-len(f['trees']):]),
            price_used_trees=sum(z['price_used'] for z in tree_audit[-len(f['trees']):]),
            reached_zero_target_leaf_fraction=float(np.mean(zero_support)),reached_target_company_support_mean=float(np.mean(target_support)),
            tree_replay_max_error=exact,complete_function_reconstruction_error=reconstruction))
    oldmetrics=pd.read_csv(register(source/'evaluation/metrics.csv'),dtype={'scope':str,'seed':str})
    actual=pd.DataFrame(replay)
    joined=actual.merge(oldmetrics,on=['scope','model','seed','cause'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(joined)==len(actual)==80
    for col in ['auc','ap','brier','risk_mean']:
        np.testing.assert_allclose(joined[col+'_new'],joined[col+'_old'],atol=1e-12,rtol=0,equal_nan=True)
    actual.to_csv(out/'metric_replay.csv',index=False)
    pd.DataFrame(train_metrics).to_csv(out/'training_apparent_metrics.csv',index=False)
    pd.DataFrame(support_summary).to_csv(out/'saved_tree_support.csv',index=False)
    pd.DataFrame(tree_audit).to_csv(private/'tree_price_usage_PRIVATE.csv',index=False)

    prediction_rows=[]
    for model in protocol['models']:
        for seed in [*map(str,protocol['seeds']),'ensemble']:
            fullrisk=np.concatenate([risks[(year,model,seed)] for year in protocol['test_years']])
            chosen={f:annual_selection(meta,fullrisk[:,0],f) for f in [.2,.25]}
            zero=np.concatenate([np.mean([support_by_key[(year,model,str(s))] for s in protocol['seeds']],axis=0)
                                 if seed=='ensemble' else support_by_key[(year,model,seed)]
                                 for year in protocol['test_years']]) if model in inherited['weights'] else np.full(len(meta),np.nan)
            for scope,mask in [('2023',meta.entry_year.eq(2023)),('2024',meta.entry_year.eq(2024)),('pooled',np.ones(len(meta),bool))]:
                yy=all_event[mask]==1;pp=fullrisk[mask,0]
                brier.append(dict(scope=scope,model=model,seed=seed,n=int(np.sum(mask)),events=int(yy.sum()),
                                  predicted_mean=float(pp.mean()),observed_mean=float(yy.mean()),**brier_parts(yy,pp)))
                submeta=meta.loc[mask];e=all_event[mask]
                for axis in ['ground','terminal_period','terminal_year','rule_evidence','day_365_plus_event']:
                    for value in sorted(submeta.loc[submeta.event.eq(1),axis].unique(),key=str):
                        cases=submeta.event.eq(1).to_numpy() & submeta[axis].eq(value).to_numpy()
                        global_cases=np.zeros(len(meta),bool);global_cases[np.flatnonzero(mask)[cases]]=True
                        support=zero[global_cases]
                        errors.append(dict(scope=scope,model=model,seed=seed,axis=axis,segment=str(value),cases=int(cases.sum()),
                            original_nontermination_controls=int((e!=1).sum()),mean_case_risk=float(pp[cases].mean()),
                            mean_case_squared_error=float(np.mean((1-pp[cases])**2)),
                            loss_contribution_to_full_cohort=float(np.sum((1-pp[cases])**2)/len(e)),
                            auc_cases_vs_original_nontermination=case_control_auc(e,pp,cases),
                            captured_top20=int((chosen[.2]&global_cases).sum()),captured_top25=int((chosen[.25]&global_cases).sum()),
                            missed_top20=int((~chosen[.2]&global_cases).sum()),
                            mean_reached_zero_target_leaf_fraction=float(support.mean()) if np.isfinite(support).any() else np.nan,
                            interpretation='outcome-defined descriptive diagnostic; other terminations excluded from AUC controls; annual selections unchanged'))
                if seed=='ensemble':
                    for axis in ['entry_warning','annual_notice_same_day','annual_version_later_than_origin','going_concern','internal_control_mentioned']:
                        for value in sorted(submeta[axis].unique(),key=str):
                            use=submeta[axis].eq(value).to_numpy();target=e[use]==1;score=pp[use]
                            subgroups.append(dict(scope=scope,model=model,axis=axis,segment=str(value),n=int(use.sum()),events=int(target.sum()),
                                risk_mean=float(score.mean()),observed=float(target.mean()),brier=float(np.mean((score-target)**2)),
                                auc=float(roc_auc_score(target,score)) if len(np.unique(target))==2 else np.nan))
            if seed=='ensemble':
                for i,row in meta.iterrows():
                    prediction_rows.append(dict(sample_id=row.sample_id,entry_year=row.entry_year,model=model,event=row.event,
                        ground=row.ground,terminal_period=row.terminal_period,risk=fullrisk[i,0],recovery_risk=fullrisk[i,1],
                        top20=chosen[.2][i],top25=chosen[.25][i],zero_target_leaf_fraction=zero[i]))
    pd.DataFrame(prediction_rows).to_parquet(private/'fixed_forecast_audit_PRIVATE.parquet',index=False)
    pd.DataFrame(brier).to_csv(out/'brier_decomposition.csv',index=False)
    pd.DataFrame(errors).to_csv(out/'event_error_segments.csv',index=False)
    pd.DataFrame(subgroups).to_csv(out/'entry_information_subgroups.csv',index=False)
    endpoints=meta[meta.event.eq(1)]
    endpoints.groupby(['entry_year','terminal_year','terminal_period','ground','rule_evidence']).size().rename('events').reset_index().to_csv(out/'terminal_clock_and_rule_counts.csv',index=False)

    info=[];composition=[];signals=[];risksets=[]
    for year in protocol['test_years']:
        for role,m,data in [('train',tm[year],tr[year]),('test',em[year],ev[year])]:
            c=m.merge(clock[['sample_id','annual_ann','annual_available']],on='sample_id',validate='one_to_one')
            e=m.merge(entry.drop(columns='scope'),on='sample_id',validate='one_to_one')
            info.append(dict(test_year=year,role=role,n=len(m),prior_annual_before_origin=int(c.annual_ann.lt(c.origin).sum()),
                prior_annual_same_day=int(c.annual_ann.eq(c.origin).sum()),prior_annual_after_origin=int(c.annual_ann.gt(c.origin).sum()),
                prior_annual_used=int((c.report_date.dt.month.eq(12)&c.report_date.dt.year.eq(c.entry_year-1)).sum()),
                latest_annual_version_after_origin=int(c.annual_available.gt(c.origin).sum()),
                selected_q3=int(c.report_date.dt.month.eq(9).sum()),financial_age_median=float(c.financial_age.median()),
                warning_type_source_present=int(m.entry_type.notna().sum()),audit_opinion_source_present=int(e.opinion.ne('').sum()),
                going_concern_source_positive=int(e.going_concern.eq('1').sum()),internal_control_text_mentions=int(e.internal_control_mentioned.sum()),
                warning_type_in_fitted_inputs=False,audit_opinion_in_fitted_inputs=False,rule_calendar_in_fitted_inputs=False,
                current_price_in_fitted_inputs=True,original_financial_vintage_PIT_certified=False))
            for kind,g in m.groupby('entry_type'):
                composition.append(dict(test_year=year,role=role,entry_type=kind,n=len(g),share=len(g)/len(m),
                    termination=int(g.event.eq(1).sum()),recovery=int(g.event.eq(2).sum()),
                    complete_no_event=int((g.event.eq(0)&g.time.ge(g.h24)).sum()),unknown=int((g.event.eq(0)&g.time.lt(g.h24)).sum())))
            for signal,index in [('lower_price',price_index),('lower_market_cap',names.index('log_total_mktcap_last'))]:
                score=-data['X'][:,index];assert np.isfinite(score).all()
                result,_,_,_=metric(data['analysis_time'],data['event'],24.,score,1,data['label_eligible'])
                signals.append(dict(test_year=year,role=role,signal=signal,n=len(m),auc=result['auc'],ap=result['ap'],
                    known_events=result['known_target_events'],unknown=result['unknown_outcomes'],
                    interpretation='fixed monotone raw-input ranking; post-exposure diagnostic, not a calibrated prediction'))
        mm=meta[meta.entry_year.eq(year)]
        for boundary in protocol['policy_dates']:
            date=pd.Timestamp(boundary)
            future=mm.origin.ge(date);at=mm.origin.lt(date)&mm.stop.ge(date);ended=mm.origin.lt(date)&mm.stop.lt(date)
            assert np.all(future.astype(int)+at.astype(int)+ended.astype(int)==1)
            risksets.append(dict(entry_year=year,date=boundary,n=len(mm),not_entered_before_date=int(future.sum()),
                first_event_free_before_date=int(at.sum()),prior_termination=int((ended&mm.event.eq(1)).sum()),
                prior_recovery=int((ended&mm.event.eq(2)).sum()),prior_horizon_end=int((ended&mm.event.eq(0)).sum()),
                interpretation='calendar risk-set description, no untreated control or policy causal estimate'))
    pd.DataFrame(info).to_csv(out/'information_set_audit.csv',index=False)
    pd.DataFrame(composition).to_csv(out/'warning_type_composition.csv',index=False)
    pd.DataFrame(signals).to_csv(out/'raw_signal_diagnostics.csv',index=False)
    pd.DataFrame(risksets).to_csv(out/'policy_date_risk_sets.csv',index=False)
    for name,digest in sources.items():
        assert sha(name)==digest,name
    write(run/'audit/source_hashes.json',sources)
    write(out/'calculation_receipt.json',dict(status='COMPLETE_FIXED_FORECAST_DIAGNOSTIC',new_research_fits=0,
        new_calibration_fits=0,existing_forests_referenced=30,q2_forests_traversed=18,q2_trees_traversed=len(tree_audit),
        fixed_forecast_metric_rows_verified=80,unique_test_companies=len(meta),test_terminations=int((all_event==1).sum()),
        unknown_test_endpoints=0,terminal_reason_unresolved=int(endpoints.ground.eq('not_explicit_in_available_terminal_text').sum()),
        max_tree_replay_error=float(pd.DataFrame(support_summary).tree_replay_max_error.max()),
        max_function_reconstruction_error=float(pd.DataFrame(support_summary).complete_function_reconstruction_error.max()),
        no_source_or_model_mutation=True,no_2025_entry_evaluation=True,no_new_inputs=True,
        wall_seconds=time.perf_counter()-started,source_files_hashed=len(sources),
        source_manifest_sha256=sha(run/'audit/source_hashes.json'),protocol_sha256=sha(run/'audit/protocol.json'),
        script_sha256=sha(Path(__file__))))
    print((out/'calculation_receipt.json').read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    analyze(parser.parse_args().run)
