"""Recover the entry-warning feature without fitting or changing old inputs.

Legal distances remain blocked until their complete historical rule and data
semantics have been verified. Outcomes are used only in descriptive audit tables.
"""
from pathlib import Path
import argparse
import json
import re
import numpy as np
import pandas as pd
from experiments.post_st_prognosis_v1.common import arrays, sha, write


NEW_FEATURE = 'entry_is_star_st'


def recover_entry(row, events):
    """Use only the actual entry announcement, never later status changes."""
    if row['entry_type'] not in ('AB', 'AD'):
        raise ValueError('Only audited first normal-to-ST/*ST entries are supported')
    ids = str(row['entry_source_ids']).split(';')
    if not ids or any(x not in events.index for x in ids):
        raise ValueError('Missing entry announcement source')
    g = events.loc[ids]
    for event in g.itertuples():
        date = event.adjudicated_announcement_date or event.Annoudt
        if not date or pd.Timestamp(date) != pd.Timestamp(row['origin']):
            raise ValueError('Entry announcement date mismatch; no execution-date fallback')
        if event.Chgtype != row['entry_type']:
            raise ValueError('Entry transition mismatch')
        name = event.ShortNameac.upper()
        state = 'D' if '*ST' in name else 'B' if 'ST' in name else None
        if state != event.Chgtype[-1]:
            raise ValueError('Source warning type conflicts with announced name')
    codes = sorted({c.strip() for v in g.Chgreas for c in re.split('[,，;；]', v) if c.strip()})
    return dict(entry_is_star_st=int(row['entry_type']=='AD'),
                available_at=pd.Timestamp(row['origin']),source_ids=';'.join(ids),
                raw_trigger_codes=','.join(codes),
                raw_prior_year_audit_opinions=';'.join(sorted(set(g.TypeAuditOpinbc))),
                raw_going_concern_flags=';'.join(sorted(set(g.ISSustManaNonStandExpl))),
                input_rule='entry announcement must already have been published')


def prepare(base_run, run, original_receipt):
    for name in ['input','private','reports']:
        (run/name).mkdir(exist_ok=False,parents=True)
    original = json.loads(original_receipt.read_text())
    for name,digest in original['model_input_hashes'].items():
        assert sha(base_run/'input'/name)==digest, name
    event_path=Path('/home/oliver/LLMworkspace/PhD_STARST_Prognosis/runs/phase1_data_freeze_v1/03_event_history.csv')
    dictionary=Path('/home/oliver/LLMworkspace/download/BDT_Trdchg[DES][csv].txt')
    assert sha(event_path)==original['source_hashes'][str(event_path)]
    assert sha(dictionary)==original['source_hashes'][str(dictionary)]
    events=pd.read_csv(event_path,dtype=str,keep_default_na=False).set_index('event_id')
    assert events.index.is_unique
    manifest=pd.read_csv(base_run/'reports/feature_manifest.csv')
    names=manifest.name.tolist();assert len(names)==75 and NEW_FEATURE not in names
    metadata=[];feature_counts=[]
    for role in ['train','eval']:
        d=pd.read_parquet(base_run/'input'/f'{role}_metadata_PRIVATE.parquet')
        old=arrays(base_run/'input'/f'{role}_PRIVATE.npz')
        assert np.array_equal(old['sample_ids'],d.sample_id.to_numpy(str))
        recovered=pd.DataFrame([recover_entry(row,events) for row in d.to_dict('records')])
        new={**old,'X':np.column_stack([old['X'],recovered[NEW_FEATURE].to_numpy(float)]),
             'feature_names':np.array(names+[NEW_FEATURE],dtype=str)}
        np.testing.assert_array_equal(new['X'][:,:75],old['X'])
        for key in old:
            if key!='X':np.testing.assert_array_equal(old[key],new[key])
        assert new['X'].shape==(len(d),76)
        out=run/'input'/f'{role}_76_columns_UNTRAINED_PRIVATE.npz'
        np.savez_compressed(out,**new)
        reread=arrays(out);np.testing.assert_array_equal(reread['X'],new['X'])
        recovered.insert(0,'sample_id',d.sample_id.to_numpy())
        recovered.to_parquet(run/'private'/f'{role}_entry_evidence_PRIVATE.parquet',index=False)
        metadata.append(d)
        feature_counts.append(dict(role=role,rows=len(d),input_columns=76,added_fields=1,
                                   missing_entry_type=0,source_date_mismatches=0,old_input_columns_unchanged=True,
                                   outcome_arrays_unchanged=True))
    d=pd.concat(metadata,ignore_index=True)
    counts=[];mix=[]
    for scope,g in d.groupby('split'):
        mix.append(dict(scope=scope,n=len(g),st=int(g.entry_type.eq('AB').sum()),
                        star_st=int(g.entry_type.eq('AD').sum()),st_fraction=float(g.entry_type.eq('AB').mean())))
        for typ,h in g.groupby('entry_type'):
            known=(h.event>0)|(h.time>=h.h24)
            counts.append(dict(scope=scope,entry_type='ST' if typ=='AB' else '*ST',n=len(h),
                termination=int(h.event.eq(1).sum()),recovery=int(h.event.eq(2).sum()),
                known_no_first_event=int((h.event.eq(0)&known).sum()),unknown=int((~known).sum()),
                termination_fraction_only_if_complete=float(h.event.eq(1).mean()) if known.all() else np.nan))
    pd.DataFrame(counts).to_csv(run/'reports/entry_type_outcomes_24m.csv',index=False)
    pd.DataFrame(mix).to_csv(run/'reports/entry_type_composition.csv',index=False)
    pd.DataFrame(feature_counts).to_csv(run/'reports/feature_recovery.csv',index=False)
    pd.DataFrame(dict(index=range(76),name=names+[NEW_FEATURE])).to_csv(run/'reports/proposed_feature_manifest.csv',index=False)
    x=d.loc[d.entry_year.eq(2024),'origin'];ann=pd.Timestamp('2024-04-30')
    audit=dict(status='TYPE_FEATURE_PREPARED_NO_TRAINING',data_status='PROVISIONAL',
        original_fit_commit=original['training_commit'],original_delivery_commit='33b60aa512b82aceacb5b905a4f5bab65e67bfac',
        base_run=str(base_run),new_private_run=str(run),new_fits=0,old_models_changed=False,old_predictions_changed=False,
        new_numeric_fields=[NEW_FEATURE],proposed_input_columns=76,original_fitted_input_columns=75,
        entry_feature_clock='at publication of the entry announcement, not before it',
        raw_trigger_metadata_recovered=True,raw_trigger_metadata_added_to_model=False,
        rule_distance_features_generated=False,all_scope_rules_resolved=False,
        train_original_ST=38,train_original_ST_unknown_h24=24,
        entries_2024_before_rule_announcement=int((x<ann).sum()),
        entries_2024_on_rule_announcement=int((x==ann).sum()),
        entries_2024_after_rule_announcement=int((x>ann).sum()),
        entries_2024_on_or_after_market_rule_effective=int((x>=pd.Timestamp('2024-10-30')).sum()),
        event_source_sha256=sha(event_path),dictionary_sha256=sha(dictionary),
        old_inputs_sha256=original['model_input_hashes'],prepare_source_sha256=sha(Path(__file__)),
        private_outputs_sha256={str(p.relative_to(run)):sha(p) for p in sorted(run.rglob('*')) if p.is_file()})
    write(run/'reports/execution_receipt.json',audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ['old_inputs_sha256','private_outputs_sha256']},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-run',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--original-receipt',type=Path,required=True)
    a=p.parse_args();prepare(a.base_run,a.run,a.original_receipt)
